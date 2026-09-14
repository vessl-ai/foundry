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


def _draft_bypass() -> bool:
    """See runtime.draft_bypass_active(): draft model runners are not cached."""
    from foundry.integration.sglang import runtime as rt

    return rt.draft_bypass_active()

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


_ar_patched = False


def _raise_allreduce_push_thresholds() -> None:
    """Keep pull-algo allreduce out of BCG-covered graph captures.

    The graph-mode pull kernels read capture-time side state (a per-capture
    device pointer-table row plus peer input registrations made during
    capture) that exists only in the process that captured the graph, so a
    restored replay hits unmapped/garbage pointers.  The push kernel carries
    no such state.  Raise the graph-context push threshold so every size a
    BCG prefill graph can carry picks push at capture time.
    """
    global _ar_patched
    if _ar_patched:
        return
    try:
        from sglang.srt.distributed.device_communicators.configs import (
            custom_all_reduce_v2 as ar_cfg,
        )
    except ImportError:
        return
    floor = int(os.environ.get("FOUNDRY_BCG_AR_PUSH_MB", "16")) * 1024 * 1024
    orig = ar_cfg.get_all_reduce_config

    @functools.wraps(orig)
    def patched(world_size: int):
        cfg = orig(world_size)
        graph = cfg.graph
        if graph.one_shot_push_threshold >= floor:
            return cfg
        logger.info(
            "[Foundry] BCG cache: raising graph allreduce push threshold "
            "%d -> %d bytes (pull algos bake capture-time state)",
            graph.one_shot_push_threshold, floor,
        )
        return cfg._replace(
            graph=graph._replace(
                one_shot_push_threshold=floor,
                one_shot_pull_threshold=max(graph.one_shot_pull_threshold, floor),
            )
        )

    ar_cfg.get_all_reduce_config = functools.cache(patched)
    _ar_patched = True


def _check_no_pull_kernels(size: int) -> None:
    """Refuse to archive a shape whose segments captured a pull allreduce —
    its replay in another process would read capture-time-only state."""
    import json

    ws = _workspace_dir()
    for filename in os.listdir(ws):
        m = _BCG_FILENAME_RE.match(filename)
        if not m or int(m.group("size")) != size:
            continue
        d = json.load(open(os.path.join(ws, filename)))
        for node in d.get("nodes", []):
            name = node.get("params", {}).get("function_name", "")
            if "AllReducePull" in name:
                raise RuntimeError(
                    f"BCG SAVE size={size}: segment {filename} captured a "
                    f"pull-algo allreduce ({name[:60]}); raise "
                    f"FOUNDRY_BCG_AR_PUSH_MB above this shape's allreduce "
                    f"message size so the push algo is selected"
                )


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
        if get_graph_extension_mode() != CUDAGraphExtensionMode.SAVE or _draft_bypass():
            return orig_capture_one(self, shape_key, forward_fn, *args, **kwargs)
        _ctx = {"size": shape_key.size, "seg": 0, "save_time": 0.0}
        _record_forward_closure(shape_key.size, forward_fn)
        try:
            result = orig_capture_one(self, shape_key, forward_fn, *args, **kwargs)
        finally:
            ctx, _ctx = _ctx, None
        graph = self._graphs.get(shape_key)
        if graph is not None:
            _check_no_pull_kernels(shape_key.size)
            _record_break_metadata(graph, shape_key.size)
            _wrap_break_checksums(graph, shape_key.size)
            _record_capture_inputs_metadata(shape_key.size, kwargs.get("capture_inputs") or (args[0] if args else None))
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

    _raise_allreduce_push_thresholds()
    _install_param_map_probe()
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

def _closure_leaf_metas(fn: Any) -> list[dict[str, Any]]:
    """Tensor leaves reachable from a function's closure cells (one level of
    object attributes), keyed by freevar-rooted path. Used to compare the
    runner's static buffers between SAVE and LOAD."""
    out: list[dict[str, Any]] = []
    names = getattr(fn.__code__, "co_freevars", ())
    cells = fn.__closure__ or ()
    for name, cell in zip(names, cells):
        try:
            out.extend(_leaf_metas(cell.cell_contents, name))
        except ValueError:
            continue
    # Host tensor addresses are not deterministic across processes and are
    # never baked into device graphs — only device buffers matter here.
    return [m for m in out if m.get("is_cuda", True)]


def _record_forward_closure(size: int, forward_fn: Any) -> None:
    import json

    metas = _closure_leaf_metas(forward_fn)
    path = os.path.join(_workspace_dir(), f"bcg_{size}_fwdclosure.json")
    with open(path, "w") as f:
        json.dump(metas, f)


def _verify_forward_closure(size: int, forward_fn: Any) -> None:
    import json

    path = os.path.join(_workspace_dir(), f"bcg_{size}_fwdclosure.json")
    if not os.path.exists(path):
        return
    saved = {m["path"]: m for m in json.load(open(path))}
    live = {m["path"]: m for m in _closure_leaf_metas(forward_fn)}
    moved = []
    for p, m in saved.items():
        lv = live.get(p)
        if lv is not None and lv["ptr"] != m["ptr"]:
            moved.append((p, hex(m["ptr"]), hex(lv["ptr"])))
    missing = [p for p in saved if p not in live]
    if moved or missing:
        logger.warning(
            "[Foundry] BCG closure drift size=%d: moved=%d missing=%d first=%s",
            size, len(moved), len(missing), (moved + [(p, "-", "-") for p in missing])[:4],
        )
    else:
        logger.info(
            "[Foundry] BCG closure verified size=%d (%d leaves)", size, len(saved)
        )


def _tensor_meta(t: torch.Tensor) -> dict[str, Any]:
    return {
        "ptr": t.data_ptr(),
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype).removeprefix("torch."),
        "device": t.device.index or 0,
        "is_cuda": t.is_cuda,
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
    # Preserve the nesting of the captured output so LOAD hands the tail the
    # same shape SGLang produced at SAVE (e.g. DSPARK targets return
    # ``(hidden_states, [aux_hidden_states...])``; a flat tensor list breaks
    # the model's 2-tuple unpack).  Specs are best-effort: exotic containers
    # fall back to the legacy flat form.
    try:
        meta["buffer_spec"] = _struct_spec(buf)
        meta["output_spec"] = _struct_spec(stored)
    except TypeError as exc:
        logger.warning("[Foundry] BCG output structure not archived: %s", exc)
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


def _wrap_break_checksums(graph, size: int) -> None:
    """FOUNDRY_BCG_SUM=1: print a checksum of each break's first tensor input
    before running it, on both SAVE and LOAD replay — diffing the two logs
    pinpoints the first layer whose activations diverge."""
    _sum_mode = os.environ.get("FOUNDRY_BCG_SUM")
    if _sum_mode not in ("1", "2"):
        return
    import json

    meta_path = os.path.join(_workspace_dir(), f"bcg_{size}_meta.json")
    if not os.path.exists(meta_path):
        return
    meta = json.load(open(meta_path))
    for i, (fn, bmeta) in enumerate(zip(graph._break_fns, meta["breaks"])):
        tensors = _rebuild_from_meta(bmeta["args"])
        probe = tensors[0] if tensors else None

        def wrapped(fn=fn, probe=probe, i=i, tensors=tensors):
            rank0 = (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            )
            if probe is not None and rank0:
                torch.cuda.synchronize()
                s = probe.float().abs().sum().item()
                print(f"[BCG-SUM] size={size} break={i} in={s:.6e}", flush=True)
            if _sum_mode == "2" and i == 0:
                torch.cuda.synchronize()
                _check_ep_map_watch(f"size={size}")
            if (
                i == 0
                and rank0
                and os.environ.get("FOUNDRY_MEMSNAP") == "1"
                and "memsnap" not in _dumped
            ):
                import pickle

                _dumped.add("memsnap")
                torch.cuda.synchronize()
                snap = torch.cuda.memory._snapshot()
                _mode = get_graph_extension_mode().value
                with open(f"/work/logs/memsnap_{_mode}.pickle", "wb") as f:
                    pickle.dump(snap, f)
                print(f"[MEMSNAP] dumped {_mode} size={size}", flush=True)
            _dump_keys = os.environ.get("FOUNDRY_BCG_DUMP", "")
            _dk = f"{size}:{i}"
            _allr = os.environ.get("FOUNDRY_BCG_DUMP_ALLRANKS") == "1"
            _rkid = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            _do_dump = (rank0 or _allr) and _dk in _dump_keys.split(",") and _dk not in _dumped
            if _do_dump:
                torch.cuda.synchronize()
                _pre = [t.detach().clone().cpu() for t in tensors]
            r = fn()
            _pd = os.environ.get("FOUNDRY_BCG_PTRDUMP", "")
            if (rank0 or _allr) and _pd and _pd.split(":")[0] + ":" + _pd.split(":")[1] == _dk and ("ptr" + _dk) not in _dumped:
                import json as _j

                from foundry import ops as _fops

                _dumped.add("ptr" + _dk)
                torch.cuda.synchronize()
                _plist = _j.load(open(_pd.split(":", 2)[2]))
                _blob = {}
                for _e in _plist:
                    try:
                        _t = _fops.tensor_from_ptr(_e["ptr"], [262144], [1], torch.uint8, torch.cuda.current_device())
                        _blob[_e["ptr"]] = _t.clone().cpu()
                    except Exception as _exc:  # peer-device IPC buffers etc.
                        _blob[_e["ptr"]] = str(_exc)[:80]
                torch.save(_blob, f"/work/logs/ptrdump_{get_graph_extension_mode().value}_{_dk.replace(':', '_')}" + (f"_r{_rkid}" if _allr else "") + ".pt")
                print(f"[PTRDUMP] {_dk} {len(_blob)} ptrs", flush=True)
            if _do_dump:
                torch.cuda.synchronize()
                _post = [t.detach().clone().cpu() for t in tensors]
                _dumped.add(_dk)
                _mode = get_graph_extension_mode().value
                torch.save(
                    {"pre": _pre, "post": _post},
                    f"/work/logs/bcgdump_{_mode}_{size}_{i}" + (f"_r{_rkid}" if _allr else "") + ".pt",
                )
            if _sum_mode == "2" and rank0:
                torch.cuda.synchronize()
                outs = " ".join(f"{t.float().abs().sum().item():.6e}" for t in tensors)
                print(f"[BCG-SUM-OUT] size={size} break={i} args={outs}", flush=True)
            return r

        graph._break_fns[i] = wrapped


def _ep_dispatcher_map_path() -> str:
    return os.path.join(_workspace_dir(), "ep_dispatcher_map.json")


def _iter_moe_dispatchers(model):
    for name, module in model.named_modules():
        disp = getattr(module, "dispatcher", None)
        if disp is not None and hasattr(disp, "local_expert_mapping"):
            yield name, disp


def _region_bounds():
    from foundry.integration.sglang import runtime as rt
    from foundry.allocation_region import parse_size

    cfg = rt.get_config()
    base = int(cfg.base_addr)
    return base, base + parse_size(cfg.region_size)


def _ensure_ep_dispatcher_maps_in_region(model) -> None:
    """SAVE (before capture): create every dispatcher's ``local_expert_mapping``
    *now*, on this thread, inside foundry's deterministic region.

    SGLang builds the mapping lazily on the first EP dispatch.  In the
    production tree that first dispatch can happen in a context whose
    allocations are not redirected into the region (observed: Solar-Pro-4
    W4AFP8 tp2/ep2 + DSPARK, mapping at 0x7af9... while the region is
    0x6000...), so the archived address is meaningless in the LOAD process
    and ``_restore_ep_dispatcher_maps`` fails with "pointer resides on host
    memory".  Pre-creating the (tiny, content-deterministic) mapping here
    makes SGLang skip its lazy path, so the captured graphs bake an in-region
    address that LOAD can rebuild."""
    lo, hi = _region_bounds()
    created = replaced = 0
    if os.environ.get("FOUNDRY_EP_MAP_DEBUG") == "1":
        _rk = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        for _k, (name, disp) in enumerate(_iter_moe_dispatchers(model)):
            if _k < 3 or _k in (13, 47):
                logger.info(
                    "[Foundry] EPMAP-DEBUG rank=%d %s num_experts=%s local=%s routed=%s shared=%s ep_rank=%s ep_size=%s skip=%s has_map=%s",
                    _rk, name, disp.num_experts, getattr(disp, "num_local_experts", None),
                    disp.num_local_routed_experts, disp.num_local_shared_experts,
                    disp.moe_ep_rank, getattr(disp, "moe_ep_size", None),
                    getattr(disp, "skip_local_expert_mapping", None),
                    disp.local_expert_mapping is not None,
                )
    for name, disp in _iter_moe_dispatchers(model):
        if getattr(disp, "moe_ep_size", 1) <= 1 or getattr(
            disp, "skip_local_expert_mapping", False
        ):
            continue
        old = disp.local_expert_mapping
        if old is not None and lo <= old.data_ptr() < hi:
            continue
        n_routed = disp.num_local_routed_experts
        n_shared = disp.num_local_shared_experts
        dev = torch.cuda.current_device()
        new = torch.full((disp.num_experts,), -1, dtype=torch.int32, device=dev)
        start = disp.moe_ep_rank * n_routed
        new[start : start + n_routed] = torch.arange(
            0, n_routed, dtype=torch.int32, device=dev
        )
        if n_shared > 0:
            new[-n_shared:] = torch.arange(
                n_routed, n_routed + n_shared, dtype=torch.int32, device=dev
            )
        if not (lo <= new.data_ptr() < hi):
            raise RuntimeError(
                f"[Foundry] EP dispatcher map for {name} allocated outside the "
                f"region: {new.data_ptr():#x} not in [{lo:#x}, {hi:#x})"
            )
        if old is not None:
            if not torch.equal(old.to(new.device), new):
                logger.warning(
                    "[Foundry] EP dispatcher map content differs from SGLang's "
                    "lazily built one at %s; keeping SGLang's values", name
                )
                new.copy_(old.to(new.device))
            replaced += 1
        else:
            created += 1
        disp.local_expert_mapping = new
    if created or replaced:
        logger.info(
            "[Foundry] EP dispatcher maps pre-created in region "
            "(created=%d, replaced=%d, first=%s)",
            created, replaced,
            next((hex(d.local_expert_mapping.data_ptr()) for _, d in _iter_moe_dispatchers(model) if d.local_expert_mapping is not None), None),
        )


def _prebuild_multimem_gatherers(model) -> None:
    """Build SGLang's lazily-created multimem logits all-gather state *now*
    (before prefill capture), identically on SAVE and LOAD.

    ``MultimemAllGatherer`` builds its symmetric-memory buffer on the first
    *eager* call.  Upstream that happens in the decode-capture warmup
    forwards; foundry SAVE skips those warmups and LOAD skips capture, and on
    LOAD the build inside the decode runner's private MemPool fails
    ("CUDA driver error: invalid argument").  Either way the decode graphs
    bake the NCCL ring all-gather fallback (~0.4 ms per step vs a few us) —
    measured as -3.5%% conc-8 throughput on Solar-Pro-4 W4AFP8 + DSPARK.

    Building here — same point, same order on both paths, before LOAD's region
    preallocation — lets SAVE capture the multimem kernel and gives LOAD the
    same symmetric-memory addresses (verified against the archive)."""
    if os.environ.get("FOUNDRY_PREBUILD_MULTIMEM", "1") != "1":
        return
    try:
        from sglang.srt.distributed.device_communicators import (
            triton_symm_mem_ag as ag,
        )
    except Exception:
        return
    import json as _json

    lm = getattr(model, "lm_head", None)
    w = getattr(lm, "weight", None)
    if w is None:
        return
    rec = {}
    for name, mod in model.named_modules():
        g = getattr(mod, "_logits_gatherer", None)
        if not isinstance(g, ag.MultimemAllGatherer):
            continue
        if g._state is not ag.MultimemAllGatherer._UNINIT:
            continue
        x = torch.empty((1, w.shape[0]), dtype=torch.bfloat16, device=w.device)
        st = g._build(x)
        del x
        if st is g._UNINIT:
            continue
        g._state = st
        if st is None:
            rec[name] = None
            continue
        h = st.symm_mem_hdl
        rec[name] = {
            "comm_buff": st.comm_buff.data_ptr(),
            "buffer_ptrs": list(h.buffer_ptrs),
            "signal_pad_ptrs": list(h.signal_pad_ptrs),
            "buffer_ptrs_dev": int(h.buffer_ptrs_dev),
            "signal_pad_ptrs_dev": int(h.signal_pad_ptrs_dev),
            "multicast_ptr": int(h.multicast_ptr),
        }
    if not rec:
        return
    path = os.path.join(_workspace_dir(), "multimem_state.json")
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.SAVE:
        with open(path, "w") as f:
            _json.dump(rec, f)
        logger.info("[Foundry] multimem all-gather prebuilt (%d): %s", len(rec),
                    {k: (hex(v["comm_buff"]) if v else None) for k, v in rec.items()})
    elif mode == CUDAGraphExtensionMode.LOAD and os.path.exists(path):
        ref = _json.load(open(path))
        if ref == rec:
            logger.info("[Foundry] multimem all-gather prebuilt: addresses match SAVE (%d)", len(rec))
        else:
            logger.error(
                "[Foundry] multimem all-gather state DIFFERS from SAVE — restored "
                "decode graphs would read wrong symmetric memory: save=%s load=%s",
                ref, rec,
            )
            raise RuntimeError("foundry: multimem all-gather state mismatch vs archive")


def _record_ep_dispatcher_maps(model) -> None:
    """SAVE (after capture): archive the address of each dispatcher's lazily
    created ``local_expert_mapping`` — captured graphs read it in place."""
    import json as _json

    import hashlib

    out = {}
    for name, disp in _iter_moe_dispatchers(model):
        t = disp.local_expert_mapping
        if t is not None:
            digest = hashlib.sha1(
                t.detach().contiguous().cpu().numpy().tobytes()
            ).hexdigest()[:16]
            out[name] = {
                "ptr": t.data_ptr(),
                "numel": t.numel(),
                "hash": digest,
                # Verbatim contents: LOAD restores these instead of
                # re-deriving the mapping (the derivation drifted from
                # SGLang's own on the production tree — content mismatch).
                "values": t.detach().cpu().tolist(),
            }
    if out:
        with open(_ep_dispatcher_map_path(), "w") as f:
            _json.dump(out, f)
        logger.info("[Foundry] EP dispatcher maps recorded (%d layers)", len(out))


def _restore_ep_dispatcher_maps(model) -> None:
    """LOAD (before replay): make every dispatcher's ``local_expert_mapping``
    live at its archived address with its archived contents.

    The maps are small torch allocations made (SAVE) just before prefill
    capture, from the default caching-allocator pool.  Viewing the archived
    address with ``tensor_from_ptr`` is NOT enough: the LOAD process's
    allocator still considers that block free and later hands it to other
    tensors (draft-model init, runtime temporaries), which silently overwrite
    the routing table the captured graphs read — every MoE layer then routes
    part of the tokens to wrong experts (observed: garbage on the W4AFP8 +
    DSPARK production tree).  So LOAD re-runs the exact SAVE allocation
    (``_ensure_ep_dispatcher_maps_in_region`` at the same point in the same
    order, before region preallocation) and only verifies the address; the
    torch-owned tensor keeps the block reserved.  A raw view is kept only as
    a logged fallback."""
    import json as _json

    from foundry import ops as foundry_ops
    from foundry.integration.sglang import runtime as rt

    path = _ep_dispatcher_map_path()
    if not os.path.exists(path):
        return
    ref = _json.load(open(path))
    if os.environ.get("FOUNDRY_EP_MAP_SYMMETRIC", "1") == "1":
        _ensure_ep_dispatcher_maps_in_region(model)
    rt.preallocate_for_load_mode()
    restored = owned = 0
    fallback = []
    for name, disp in _iter_moe_dispatchers(model):
        meta = ref.get(name)
        if meta is None:
            continue
        live = disp.local_expert_mapping
        if live is not None and live.data_ptr() == meta["ptr"] and live.numel() == meta["numel"]:
            t = live
            owned += 1
        else:
            fallback.append(
                (name, hex(meta["ptr"]), hex(live.data_ptr()) if live is not None else None)
            )
            t = foundry_ops.tensor_from_ptr(
                meta["ptr"], [meta["numel"]], [1], torch.int32,
                torch.cuda.current_device(),
            )
        if meta.get("values") is not None:
            if t is live and os.environ.get("FOUNDRY_EP_MAP_DEBUG") == "1":
                _cur = t.detach().cpu().tolist()
                _nd = sum(1 for _x, _y in zip(_cur, meta["values"]) if _x != _y)
                if _nd:
                    logger.warning(
                        "[Foundry] EPMAP-DEBUG %s: LOAD-derived map differs from SAVE in %d/%d entries",
                        name, _nd, len(_cur),
                    )
            t.copy_(torch.tensor(meta["values"], dtype=torch.int32, device=t.device))
        else:
            t.fill_(-1)
            n_routed = disp.num_local_routed_experts
            start = disp.moe_ep_rank * n_routed
            t[start : start + n_routed] = torch.arange(
                0, n_routed, dtype=torch.int32, device=t.device
            )
            n_shared = disp.num_local_shared_experts
            if n_shared > 0:
                t[-n_shared:] = torch.arange(
                    n_routed, n_routed + n_shared, dtype=torch.int32, device=t.device
                )
        disp.local_expert_mapping = t
        restored += 1
        if "hash" in meta:
            import hashlib

            digest = hashlib.sha1(
                t.detach().contiguous().cpu().numpy().tobytes()
            ).hexdigest()[:16]
            if digest != meta["hash"]:
                logger.warning(
                    "[Foundry] EP dispatcher map content mismatch at %s: "
                    "save=%s load=%s", name, meta["hash"], digest,
                )
    if fallback:
        logger.warning(
            "[Foundry] EP dispatcher maps NOT allocator-owned at archived address "
            "(%d/%d; raw view fallback, may be overwritten): first=%s",
            len(fallback), restored, fallback[:3],
        )
    if restored:
        logger.info(
            "[Foundry] EP dispatcher maps restored (%d layers, %d allocator-owned)",
            restored, owned,
        )
    global _ep_map_watch
    _ep_map_watch = [
        (name, disp.local_expert_mapping, ref[name].get("values"))
        for name, disp in _iter_moe_dispatchers(model)
        if name in ref and ref[name].get("values") is not None
    ][:2]


_ep_map_watch: list = []
_dumped: set = set()


def _check_ep_map_watch(tag: str) -> None:
    """FOUNDRY_BCG_SUM=2 diagnostic: detect later overwrites of the maps."""
    for name, t, values in _ep_map_watch:
        cur = t.detach().cpu().tolist()
        if cur != values:
            bad = sum(1 for a, b in zip(cur, values) if a != b)
            print(f"[EP-MAP-CLOBBERED] {tag} {name} {bad}/{len(values)} entries differ", flush=True)


def _install_param_map_probe() -> None:
    """Record every model parameter/buffer address at SAVE and verify them at
    LOAD.  Restored graphs read these tensors at their captured addresses, so
    any drift (e.g. nondeterministic expert-shard loading under EP) silently
    computes with the wrong weights."""
    from sglang.srt.model_executor.runner import prefill_cuda_graph_runner as pr

    orig = pr.PrefillCudaGraphRunner.capture
    if getattr(orig, "_foundry_param_probe", False):
        return

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        import json as _json

        mode = get_graph_extension_mode()
        try:
            import hashlib

            model = self.model_runner.model
            snap = {n: t.data_ptr() for n, t in model.named_parameters()}
            snap.update({"buf:" + n: t.data_ptr() for n, t in model.named_buffers()})
            # Plain tensor attributes (not registered as parameter/buffer) —
            # e.g. Marlin ``layer.workspace``, dispatcher maps — are read by
            # captured kernels at their SAVE-time address too.
            _seen = {id(t) for _, t in model.named_parameters()} | {
                id(t) for _, t in model.named_buffers()
            }
            for mn, mod in model.named_modules():
                for holder_name, holder in (("", mod), (".dispatcher", getattr(mod, "dispatcher", None))):
                    if holder is None:
                        continue
                    for an, av in list(vars(holder).items()):
                        if isinstance(av, torch.Tensor) and av.is_cuda and id(av) not in _seen:
                            key = f"attr:{mn}{holder_name}.{an}"
                            snap[key] = av.data_ptr()
                            if 0 < av.numel() * av.element_size() <= (1 << 20):
                                raw = av.detach().contiguous().cpu().view(torch.uint8)
                                snap["hash:" + key] = hashlib.sha1(bytes(raw.numpy().tobytes())).hexdigest()[:16]
            # Content hashes for small buffers — restored graphs read them at
            # captured addresses, so stale contents silently corrupt compute.
            for n, t in model.named_buffers():
                if 0 < t.numel() * t.element_size() <= (1 << 20):
                    raw = t.detach().contiguous().cpu().view(torch.uint8)
                    snap["hash:" + n] = hashlib.sha1(bytes(raw.numpy().tobytes())).hexdigest()[:16]
            # Memory-pool tensors (KV / mamba / req_to_token) — captured
            # segments may read them at baked addresses.
            def _walk_pool(obj, prefix, depth, seen_ids):
                if depth > 4 or obj is None or id(obj) in seen_ids:
                    return
                seen_ids.add(id(obj))
                if isinstance(obj, torch.Tensor):
                    if obj.is_cuda:
                        snap[prefix] = obj.data_ptr()
                    return
                if isinstance(obj, (list, tuple)):
                    for k, v in enumerate(obj[:512]):
                        _walk_pool(v, f"{prefix}[{k}]", depth + 1, seen_ids)
                    return
                if isinstance(obj, dict):
                    for k, v in list(obj.items())[:512]:
                        _walk_pool(v, f"{prefix}[{k!r}]", depth + 1, seen_ids)
                    return
                if hasattr(obj, "__dict__") and not isinstance(obj, torch.nn.Module):
                    for k, v in list(vars(obj).items()):
                        if isinstance(v, (torch.Tensor, list, tuple, dict)) or (
                            hasattr(v, "__dict__") and type(v).__module__.startswith("sglang")
                        ):
                            _walk_pool(v, f"{prefix}.{k}", depth + 1, seen_ids)

            _mr = self.model_runner
            _pseen: set = set()
            for _pn in ("token_to_kv_pool", "req_to_token_pool", "token_to_kv_pool_allocator"):
                _walk_pool(getattr(_mr, _pn, None), f"pool:{_pn}", 0, _pseen)
            path = os.path.join(_workspace_dir(), "model_ptr_map.json")
            if mode == CUDAGraphExtensionMode.SAVE:
                with open(path, "w") as f:
                    _json.dump(snap, f)
                logger.info("[Foundry] model ptr map saved (%d tensors)", len(snap))
            elif mode == CUDAGraphExtensionMode.LOAD and os.path.exists(path):
                ref = _json.load(open(path))
                moved = [
                    (n, ref[n] if n.startswith("hash:") else hex(ref[n]),
                     snap.get(n, 0) if n.startswith("hash:") else hex(snap.get(n, 0)))
                    for n in ref
                    if snap.get(n) != ref[n]
                ]
                attr_n = sum(1 for n in ref if n.startswith("attr:"))
                pool_n = sum(1 for n in ref if n.startswith("pool:"))
                logger.info(
                    "[Foundry] model ptr probe: %d attr + %d pool tensors compared",
                    attr_n, pool_n,
                )
                if moved:
                    logger.warning(
                        "[Foundry] model ptr drift: %d/%d tensors moved; first=%s",
                        len(moved), len(ref), moved[:5],
                    )
                else:
                    logger.info(
                        "[Foundry] model ptr map verified (%d tensors)", len(ref)
                    )
        except Exception as exc:
            logger.warning("[Foundry] model ptr probe failed: %s", exc)
        shared_capture_setup_pre(self.model_runner, mode)
        result = orig(self, *args, **kwargs)
        shared_capture_setup_post(self.model_runner, mode, "after prefill capture")
        finalize_archive_if_last_capture(self.model_runner, mode, "decode")
        return result

    patched._foundry_param_probe = True
    pr.PrefillCudaGraphRunner.capture = patched


def _install_skip_sizes_filter() -> None:
    """Drop FOUNDRY_BCG_SKIP_SIZES from the prefill runner's bucket list so
    dispatch pads those sizes up to the next available graph instead of
    looking up a graph that was never restored."""
    skip = {
        int(s) for s in os.environ.get("FOUNDRY_BCG_SKIP_SIZES", "").split(",") if s
    }
    if not skip:
        return
    from sglang.srt.model_executor.runner import prefill_cuda_graph_runner as pr

    orig_capture = pr.PrefillCudaGraphRunner.capture

    @functools.wraps(orig_capture)
    def patched_capture(self, *args, **kwargs):
        kept = [t for t in self.capture_num_tokens if t not in skip]
        if len(kept) != len(self.capture_num_tokens):
            logger.info(
                "[Foundry] BCG LOAD: dropping skip sizes %s from prefill "
                "buckets", sorted(set(self.capture_num_tokens) - set(kept)),
            )
            self.capture_num_tokens = kept
        return orig_capture(self, *args, **kwargs)

    pr.PrefillCudaGraphRunner.capture = patched_capture


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
        if get_graph_extension_mode() != CUDAGraphExtensionMode.LOAD or _draft_bypass():
            return orig_capture_one(self, shape_key, forward_fn, capture_inputs,
                                    post_warmup_hook)

        # Escape hatch for shapes whose archived kernels cannot replay in a
        # fresh process (e.g. cuBLAS variants relying on host-initialized
        # module globals).  Capturing live mid-restore would desync the pool
        # trajectory, so skip the shape entirely — dispatch pads such sizes
        # up to the next available graph.
        skip = os.environ.get("FOUNDRY_BCG_SKIP_SIZES", "")
        if skip and str(shape_key.size) in skip.split(","):
            logger.info(
                "[Foundry] BCG LOAD size=%d: skipped (FOUNDRY_BCG_SKIP_SIZES); "
                "dispatch will pad to the next larger graph", shape_key.size,
            )
            return None

        from foundry.integration.sglang import runtime as rt

        # BCG restore runs before the decode runner's preallocation; the
        # archived addresses must be mapped before tensor_from_ptr views them.
        rt.preallocate_for_load_mode()

        size = shape_key.size
        _verify_forward_closure(size, forward_fn)
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
            if out_meta.get("buffer_spec") is not None:
                self._shared_output_buffer = _build_from_spec(out_meta["buffer_spec"], bufs)
            else:
                self._shared_output_buffer = bufs[0] if len(bufs) == 1 else bufs
        outs = _rebuild_from_meta(out_meta["output"])
        if out_meta.get("output_spec") is not None:
            stored = _build_from_spec(out_meta["output_spec"], outs)
        else:
            stored = outs[0] if len(outs) == 1 else outs

        _wrap_break_checksums(graph, size)
        verify_capture_inputs(size, capture_inputs)
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

    _raise_allreduce_push_thresholds()
    _install_skip_sizes_filter()
    _install_param_map_probe()
    backend_cls.capture_one = patched_capture_one
    logger.info("[Foundry] BCG load hooks installed")


def _record_capture_inputs_metadata(size: int, capture_inputs: Any) -> None:
    """Archive the runner's static input-buffer addresses so LOAD can verify
    that its own buffers landed at the same place — the graph has these
    addresses baked in, so a mismatch means garbage replay."""
    import json

    metas = _leaf_metas(capture_inputs) if capture_inputs is not None else []
    path = os.path.join(_workspace_dir(), f"bcg_{size}_inputs.json")
    with open(path, "w") as f:
        json.dump({"inputs": metas}, f)


def verify_capture_inputs(size: int, capture_inputs: Any) -> None:
    import json

    path = os.path.join(_workspace_dir(), f"bcg_{size}_inputs.json")
    if not os.path.exists(path):
        return
    want = json.load(open(path))["inputs"]
    have = _leaf_metas(capture_inputs) if capture_inputs is not None else []
    if len(want) != len(have):
        logger.error(
            "[Foundry] BCG size=%d input-buffer count mismatch: archive=%d run=%d",
            size, len(want), len(have),
        )
        return
    bad = [
        (w["path"], hex(w["ptr"]), hex(h["ptr"]))
        for w, h in zip(want, have)
        if w["ptr"] != h["ptr"]
    ]
    if bad:
        logger.error(
            "[Foundry] BCG size=%d INPUT-ADDR MISMATCH (%d/%d): %s",
            size, len(bad), len(want), bad[:4],
        )
    else:
        logger.info("[Foundry] BCG size=%d input buffers verified (%d)", size, len(want))


# --- shared capture setup -------------------------------------------------
#
# These three steps must run exactly once per process, around whichever graph
# capture actually happens.  In a standalone server that is the prefill (BCG)
# capture, which runs first.  Under PD disaggregation sglang disables one
# phase per role, so a decode-role server never enters the prefill hook at
# all -- the decode hook calls these instead (see hooks.py).


def shared_capture_setup_pre(model_runner, mode) -> None:
    """Multimem prebuild + autotune/EP-map restore, before graph capture."""
    from foundry.integration.sglang import autotune_ops

    from foundry.integration.sglang import runtime as _rt

    model = model_runner.model
    if mode in (CUDAGraphExtensionMode.SAVE, CUDAGraphExtensionMode.LOAD):
        _rt.pin_alloc_offset("before multimem prebuild")
        _prebuild_multimem_gatherers(model)
    if mode == CUDAGraphExtensionMode.LOAD:
        autotune_ops.load(_workspace_dir())
        _restore_ep_dispatcher_maps(model)
    elif mode == CUDAGraphExtensionMode.SAVE:
        _ensure_ep_dispatcher_maps_in_region(model)


def shared_capture_setup_post(model_runner, mode, tag: str) -> None:
    """Record EP maps and pin autotune decisions, after graph capture."""
    from foundry.integration.sglang import autotune_ops

    if mode != CUDAGraphExtensionMode.SAVE:
        return
    _record_ep_dispatcher_maps(model_runner.model)
    autotune_ops.save(_workspace_dir(), tag)


def finalize_archive_if_last_capture(model_runner, mode, other_phase: str) -> None:
    """Write the archive's completion marker when no further capture follows.

    ``final_alloc_offset.json`` is what makes an archive complete for auto
    mode, and it is normally written at the end of decode capture.  A PD
    prefill-role server disables decode graphs, so without this the archive
    stays incomplete forever and auto mode re-SAVEs on every start.
    """
    if mode != CUDAGraphExtensionMode.SAVE:
        return
    from foundry.integration.sglang import runtime as _rt

    server_args = getattr(model_runner, "server_args", None)
    if server_args is None or not _rt.phase_graph_disabled(server_args, other_phase):
        return
    from foundry.integration.sglang.graph_ops import pack_fatbins, save_graph_manifest

    save_graph_manifest()
    pack_fatbins()
    _rt.capture_final_alloc_offset()
    ws = _workspace_dir()
    if ws:
        _rt.check_no_nccl_in_cached_graphs(ws, f"{other_phase} graphs disabled")
    logger.info(
        "[Foundry] archive finalized after capture (%s graphs disabled by PD role)",
        other_phase,
    )
