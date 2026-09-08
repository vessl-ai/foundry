# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Disk caching for SGLang's breakable (BCG) prefill CUDA graphs.

A BCG shape is a chain of plain CUDA graph segments with eager break
functions between them (one break per attention layer).  The segments are
ordinary stream captures — the same thing foundry already persists for
decode — and the break closures capture only addresses, all of which are
deterministic under foundry's VMM region.  So:

SAVE  — capture each segment into a foundry graph instead of a torch graph
        and write it to the archive as ``bcg_{size}_s{index}.json``.  The
        rest of the BCG flow (breaks, output buffer, bookkeeping) is
        untouched; foundry graphs are API-compatible with torch's.

LOAD  — (next stage) skip the warmup/capture forwards, restore the segment
        graphs from the archive, and rebuild the break closures by running
        the break bodies once against the restored segments' addresses.

Gated behind FOUNDRY_BCG_CACHE=1 while under development.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import time
from typing import Any

import torch

from foundry.graph import CUDAGraph as FoundryCUDAGraph
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    get_config,
    get_graph_extension_mode,
)

logger = logging.getLogger(__name__)

_BCG_FILENAME_RE = re.compile(r"^bcg_(?P<size>\d+)_s(?P<seg>\d+)\.json$")

# Active only between capture_one entry and exit on the BCG backend; the
# segment hooks consult it to know which shape they belong to.
_ctx: dict[str, Any] | None = None


def _segment_filename(size: int, seg_index: int) -> str:
    return f"bcg_{size}_s{seg_index}.json"


def _workspace_dir() -> str:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry workspace_dir is not initialized")
    return cfg.workspace_dir


def scan_segment_files(workspace_dir: str) -> dict[int, list[str]]:
    """size -> [filename ordered by segment index]."""
    by_size: dict[int, list[tuple[int, str]]] = {}
    for filename in os.listdir(workspace_dir):
        m = _BCG_FILENAME_RE.match(filename)
        if not m:
            continue
        by_size.setdefault(int(m.group("size")), []).append(
            (int(m.group("seg")), filename)
        )
    return {
        size: [f for _, f in sorted(entries)] for size, entries in by_size.items()
    }


def install_bcg_save_hooks() -> None:
    """Route BCG segment captures through foundry graphs and archive them."""
    from sglang.srt.model_executor.runner_backend import (
        breakable_cuda_graph_backend as bcg_mod,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        breakable_cuda_graph as bcgraph_mod,
    )

    backend_cls = bcg_mod.BreakableCudaGraphBackend
    capture_cls = bcgraph_mod.BreakableCUDAGraphCapture
    graph_cls = bcgraph_mod.BreakableCUDAGraph

    orig_capture_one = backend_cls.capture_one
    orig_begin = capture_cls._begin_new_segment
    orig_append = graph_cls._append_segment

    @functools.wraps(orig_capture_one)
    def patched_capture_one(self, shape_key, forward_fn, *args, **kwargs):
        global _ctx
        if get_graph_extension_mode() != CUDAGraphExtensionMode.SAVE:
            return orig_capture_one(self, shape_key, forward_fn, *args, **kwargs)
        _ctx = {"size": shape_key.size, "seg": 0, "save_time": 0.0}
        try:
            result = orig_capture_one(self, shape_key, forward_fn, *args, **kwargs)
        finally:
            ctx, _ctx = _ctx, None
        graph = self._graphs.get(shape_key)
        if graph is not None:
            _record_break_metadata(graph, shape_key.size)
            record_shared_buffer_metadata(self, shape_key, self._outputs.get(shape_key))
        logger.info(
            "[Foundry] BCG saved %d segments for size=%d (save %.3fs)",
            ctx["seg"], ctx["size"], ctx["save_time"],
        )
        return result

    @functools.wraps(orig_begin)
    def patched_begin(self):
        # Only the plain path is supported; dedup wraps segments in registry
        # objects foundry cannot serialize. Dedup is off by default
        # (SGLANG_ENABLE_CUDA_GRAPH_DEDUP=False); refuse loudly if enabled.
        if _ctx is None:
            return orig_begin(self)
        if self.cuda_graph._deduped_cuda_graph is not None:
            raise RuntimeError(
                "FOUNDRY_BCG_CACHE does not support SGLANG_ENABLE_CUDA_GRAPH_DEDUP"
            )
        graph = FoundryCUDAGraph()
        graph.capture_begin(
            pool=self._pool, capture_error_mode=self._capture_error_mode
        )
        self._current_graph = graph
        self._current_graph_needs_instantiate = False

    @functools.wraps(orig_append)
    def patched_append(self, graph, needs_instantiate):
        orig_append(self, graph, needs_instantiate)
        if _ctx is None or not isinstance(graph, FoundryCUDAGraph):
            return
        t0 = time.perf_counter()
        path = os.path.join(
            _workspace_dir(), _segment_filename(_ctx["size"], _ctx["seg"])
        )
        graph.save(path, None)
        _ctx["seg"] += 1
        _ctx["save_time"] += time.perf_counter() - t0

    backend_cls.capture_one = patched_capture_one
    capture_cls._begin_new_segment = patched_begin
    graph_cls._append_segment = patched_append
    logger.info("[Foundry] BCG save hooks installed")


# ---------------------------------------------------------------------------
# Break-closure metadata: what LOAD needs to rebuild each replay_fn without
# running the capture. Tensor leaves are recorded as (ptr, shape, stride,
# dtype); non-tensor args are runtime Python objects the LOAD-side eager run
# provides. Paths let LOAD substitute tensors positionally.
# ---------------------------------------------------------------------------

def _tensor_meta(t: torch.Tensor) -> dict[str, Any]:
    return {
        "ptr": t.data_ptr(),
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype).removeprefix("torch."),
        "device": t.device.index or 0,
    }


def _leaf_metas(obj: Any, path: str = "") -> list[dict[str, Any]]:
    """Tensor leaves of args/output structures, with their access path."""
    out: list[dict[str, Any]] = []
    if isinstance(obj, torch.Tensor):
        m = _tensor_meta(obj)
        m["path"] = path
        out.append(m)
    elif isinstance(obj, (tuple, list)):
        for i, e in enumerate(obj):
            out.extend(_leaf_metas(e, f"{path}[{i}]"))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_leaf_metas(v, f"{path}[{k!r}]"))
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if isinstance(v, torch.Tensor) or isinstance(v, (tuple, list, dict)):
                out.extend(_leaf_metas(v, f"{path}.{k}"))
    return out


def _nontensor_leaves(obj: Any, path: str = "") -> list[dict[str, Any]]:
    """Non-tensor leaves of args/kwargs, for deciding whether closures can be
    cloned across shapes instead of collected per shape."""
    out: list[dict[str, Any]] = []
    if isinstance(obj, torch.Tensor):
        return out
    if isinstance(obj, (tuple, list)):
        for i, e in enumerate(obj):
            out.extend(_nontensor_leaves(e, f"{path}[{i}]"))
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_nontensor_leaves(v, f"{path}[{k!r}]"))
        return out
    entry = {"path": path, "type": type(obj).__qualname__}
    if isinstance(obj, (int, float, bool, str, type(None))):
        entry["value"] = obj
    else:
        entry["id"] = id(obj)
        entry["repr"] = repr(obj)[:80]
    out.append(entry)
    return out


def _struct_spec(obj: Any) -> Any:
    """JSON-serializable skeleton of args/kwargs: tensors become {"t": n}
    (numbered in _leaf_metas order), scalars keep their value. Raises on
    anything else so unsupported models fail at SAVE, not silently at LOAD."""
    counter = [0]

    def walk(o):
        if isinstance(o, torch.Tensor):
            spec = {"t": counter[0]}
            counter[0] += 1
            return spec
        if isinstance(o, (bool, int, float, str)) or o is None:
            return {"v": o}
        if isinstance(o, tuple):
            return {"tuple": [walk(e) for e in o]}
        if isinstance(o, list):
            return {"list": [walk(e) for e in o]}
        if isinstance(o, dict):
            return {"dict": {k: walk(v) for k, v in o.items()}}
        raise TypeError(
            f"BCG break closure holds a non-scalar object {type(o).__qualname__}; "
            "closure synthesis cannot archive it"
        )

    return walk(obj)


def _build_from_spec(spec: Any, tensors: list[torch.Tensor]) -> Any:
    if "t" in spec:
        return tensors[spec["t"]]
    if "v" in spec:
        return spec["v"]
    if "tuple" in spec:
        return tuple(_build_from_spec(e, tensors) for e in spec["tuple"])
    if "list" in spec:
        return [_build_from_spec(e, tensors) for e in spec["list"]]
    if "dict" in spec:
        return {k: _build_from_spec(v, tensors) for k, v in spec["dict"].items()}
    raise ValueError(f"bad spec node: {spec!r}")


def _closure_cells(fn) -> dict[str, Any]:
    return dict(zip(fn.__code__.co_freevars, [c.cell_contents for c in fn.__closure__]))


def _record_break_metadata(graph, size: int) -> None:
    """After capture_one, introspect each break closure and archive tensor
    metadata for LOAD-side reconstruction."""
    import json

    breaks = []
    for fn in graph._break_fns:
        cells = _closure_cells(fn)
        breaks.append({
            "inner": getattr(cells.get("captured_inner"), "__qualname__", "?"),
            "args": _leaf_metas(cells.get("captured_args")),
            "kwargs": _leaf_metas(cells.get("captured_kwargs")),
            "output": _leaf_metas(cells.get("captured_output")),
            "nontensor_args": _nontensor_leaves(cells.get("captured_args")),
            "nontensor_kwargs": _nontensor_leaves(cells.get("captured_kwargs")),
            "inner_module": getattr(cells.get("captured_inner"), "__module__", None),
            "inner_name": getattr(cells.get("captured_inner"), "__name__", None),
            "args_spec": _struct_spec(cells.get("captured_args")),
            "kwargs_spec": _struct_spec(cells.get("captured_kwargs")),
            "output_is_none": cells.get("captured_output") is None,
        })
    path = os.path.join(_workspace_dir(), f"bcg_{size}_meta.json")
    with open(path, "w") as f:
        json.dump({"size": size, "segments": len(graph._segments), "breaks": breaks}, f)


def record_shared_buffer_metadata(backend, shape_key, stored) -> None:
    """Archive the shared output buffer and per-shape output-slice metadata."""
    import json

    buf = backend._shared_output_buffer
    meta = {
        "buffer": _leaf_metas(buf),
        "output": _leaf_metas(stored),
    }
    path = os.path.join(
        _workspace_dir(), f"bcg_{shape_key.size}_out.json"
    )
    with open(path, "w") as f:
        json.dump(meta, f)


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def _rebuild_from_meta(metas: list[dict[str, Any]]):
    from foundry import ops

    tensors = []
    for m in metas:
        tensors.append(
            ops.tensor_from_ptr(
                m["ptr"], m["shape"], m["stride"],
                getattr(torch, m["dtype"]), m["device"],
            )
        )
    return tensors


def _substitute_leaves(obj: Any, replacements: list[torch.Tensor], idx: list[int]):
    """Walk obj in _leaf_metas order, replacing each tensor leaf with the next
    archived-address tensor. Non-tensor leaves pass through untouched."""
    if isinstance(obj, torch.Tensor):
        t = replacements[idx[0]]
        idx[0] += 1
        return t
    if isinstance(obj, tuple):
        return tuple(_substitute_leaves(e, replacements, idx) for e in obj)
    if isinstance(obj, list):
        return [_substitute_leaves(e, replacements, idx) for e in obj]
    if isinstance(obj, dict):
        return {k: _substitute_leaves(v, replacements, idx) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        for k, v in list(vars(obj).items()):
            if isinstance(v, (torch.Tensor, tuple, list, dict)):
                setattr(obj, k, _substitute_leaves(v, replacements, idx))
        return obj
    return obj


def _import_inner(module_name: str, fn_name: str):
    import importlib

    # torch custom ops live under the synthetic torch._ops.<namespace> module,
    # which importlib cannot import; go through torch.ops instead.
    if module_name.startswith("torch._ops."):
        namespace = module_name.split(".", 2)[2]
        return getattr(getattr(torch.ops, namespace), fn_name)
    mod = importlib.import_module(module_name)
    return getattr(mod, fn_name)


def _synthesize_break_fn(bmeta: dict[str, Any]):
    """Assemble a replay_fn equivalent to the one eager_on_graph builds,
    entirely from archived metadata: tensor leaves at their archived
    addresses, scalar state from the spec, the break body re-imported."""
    inner = _import_inner(bmeta["inner_module"], bmeta["inner_name"])
    arg_tensors = _rebuild_from_meta(bmeta["args"])
    kw_tensors = _rebuild_from_meta(bmeta["kwargs"])
    args = _build_from_spec(bmeta["args_spec"], arg_tensors)
    kwargs = _build_from_spec(bmeta["kwargs_spec"], kw_tensors)
    if not bmeta.get("output_is_none", False):
        raise TypeError("closure synthesis requires out-parameter breaks")

    def replay_fn():
        return inner(*args, **kwargs)

    return replay_fn


def _can_synthesize(meta: dict[str, Any]) -> bool:
    return all(
        b.get("args_spec") is not None
        and b.get("inner_module")
        and b.get("output_is_none", False)
        for b in meta["breaks"]
    )


class _CollectOnlyCapture:
    """Stands in for BreakableCUDAGraphCapture during LOAD's closure-collect
    run: the eager_on_graph wrapper fires (so break closures are built) but
    no stream capture happens."""

    def __init__(self, cuda_graph, barrier_fn):
        self.cuda_graph = cuda_graph
        self._barrier_fn = barrier_fn

    def _end_current_segment(self):
        pass

    def _begin_new_segment(self):
        pass


def install_bcg_load_hooks() -> None:
    import json

    from sglang.srt.model_executor.runner_backend import (
        breakable_cuda_graph_backend as bcg_mod,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        breakable_cuda_graph as bcgraph_mod,
    )

    backend_cls = bcg_mod.BreakableCudaGraphBackend
    graph_cls = bcgraph_mod.BreakableCUDAGraph
    orig_capture_one = backend_cls.capture_one

    stats = {"shapes": 0, "eager": 0.0, "restore": 0.0}

    @functools.wraps(orig_capture_one)
    def patched_capture_one(self, shape_key, forward_fn, capture_inputs=None,
                            post_warmup_hook=None):
        if get_graph_extension_mode() != CUDAGraphExtensionMode.LOAD:
            return orig_capture_one(self, shape_key, forward_fn, capture_inputs,
                                    post_warmup_hook)

        from foundry.integration.sglang import runtime as rt

        # BCG restore runs before the decode runner's preallocation; the
        # archived addresses must be mapped before tensor_from_ptr views them.
        rt.preallocate_for_load_mode()

        size = shape_key.size
        ws = _workspace_dir()
        meta = json.load(open(os.path.join(ws, f"bcg_{size}_meta.json")))
        out_meta = json.load(open(os.path.join(ws, f"bcg_{size}_out.json")))

        graph = graph_cls(None)
        if _can_synthesize(meta):
            # Fast path: break closures hold only tensors (archived addresses)
            # and scalars (archived values) — assemble them directly, no
            # forward at all.
            t0 = time.perf_counter()
            for bmeta in meta["breaks"]:
                graph._break_fns.append(_synthesize_break_fn(bmeta))
            stats["synth"] = stats.get("synth", 0.0) + time.perf_counter() - t0
        else:
            # Fallback: one eager forward with a collect-only capture context,
            # then swap the closures' tensor state for archived addresses.
            token = bcgraph_mod._current_capture_var.set(
                _CollectOnlyCapture(graph, self._tp_group.barrier)
            )
            t0 = time.perf_counter()
            try:
                self._device_module.synchronize()
                self._tp_group.barrier()
                forward_fn()
                if post_warmup_hook is not None:
                    post_warmup_hook()
            finally:
                bcgraph_mod._current_capture_var.reset(token)
            stats["eager"] += time.perf_counter() - t0

            if len(graph._break_fns) != len(meta["breaks"]):
                raise RuntimeError(
                    f"BCG LOAD size={size}: collected {len(graph._break_fns)} break "
                    f"closures but archive has {len(meta['breaks'])}"
                )
            for fn, bmeta in zip(graph._break_fns, meta["breaks"]):
                names = fn.__code__.co_freevars
                cells = fn.__closure__
                for cell_name, meta_key in (
                    ("captured_args", "args"),
                    ("captured_kwargs", "kwargs"),
                    ("captured_output", "output"),
                ):
                    if cell_name not in names:
                        continue
                    i = names.index(cell_name)
                    repl = _rebuild_from_meta(bmeta[meta_key])
                    idx = [0]
                    new_val = _substitute_leaves(cells[i].cell_contents, repl, idx)
                    if idx[0] != len(repl):
                        raise RuntimeError(
                            f"BCG LOAD size={size}: {cell_name} leaf count mismatch "
                            f"(walked {idx[0]}, archive {len(repl)})"
                        )
                    cells[i].cell_contents = new_val

        t0 = time.perf_counter()

        # Restore the archived segments in order.
        pool = self._pool
        for seg_idx in range(meta["segments"]):
            path = os.path.join(ws, _segment_filename(size, seg_idx))
            loaded = FoundryCUDAGraph.load(path, pool)
            seg = loaded[0] if isinstance(loaded, tuple) else loaded
            graph._segments.append(seg)

        # Shared output buffer + per-shape output slice at archived addresses.
        if self._shared_output_buffer is None:
            bufs = _rebuild_from_meta(out_meta["buffer"])
            self._shared_output_buffer = bufs[0] if len(bufs) == 1 else bufs
        outs = _rebuild_from_meta(out_meta["output"])
        stored = outs[0] if len(outs) == 1 else outs

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = stored
        self._capture_inputs[shape_key] = capture_inputs
        stats["restore"] += time.perf_counter() - t0
        stats["shapes"] += 1
        if stats["shapes"] % 20 == 0:
            logger.info(
                "[Foundry] BCG LOAD progress: %d shapes (synth %.2fs, eager %.2fs, restore %.2fs)",
                stats["shapes"], stats.get("synth", 0.0), stats["eager"], stats["restore"],
            )

    backend_cls.capture_one = patched_capture_one
    logger.info("[Foundry] BCG load hooks installed")
