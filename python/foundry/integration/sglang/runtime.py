# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime state and VMM setup for the Foundry SGLang integration."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from foundry import ops as cge
from foundry.allocation_region import parse_size
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    compute_workspace_rank,
    get_config,
    get_graph_extension_mode,
    get_hook_library_path,
    get_nvshmem_host_path,
)

logger = logging.getLogger(__name__)


@dataclass
class WarmupState:
    sglang_version: str = ""
    timestamp: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_total_memory: int = 0
    memory_pool_config: dict = field(default_factory=dict)
    final_alloc_offset: int = 0
    # Hybrid linear-attention (mamba/KDA) models: the mamba pool size is
    # computed from free memory during SAVE's init and written back into
    # server_args — a path LOAD skips. Persist the resolved values so LOAD
    # can restore them before _apply_memory_pool_config.
    server_args_overrides: dict = field(default_factory=dict)
    # Everything an archive is pinned to (GPU arch, driver/torch/sglang
    # versions, model, parallel layout, graph batch sizes, VMM region).
    # AUTO mode refuses to restore an archive whose fingerprint differs.
    fingerprint: dict = field(default_factory=dict)


@dataclass
class CUDAGraphExtensionState:
    capture_index: int = 0
    rank: int = 0
    loaded_graphs: dict = field(default_factory=dict)


_state: CUDAGraphExtensionState | None = None
_final_alloc_offset: int = 0


def get_state() -> CUDAGraphExtensionState | None:
    return _state


def _workspace_dir() -> str | None:
    cfg = get_config()
    return None if cfg is None else cfg.workspace_dir


def _workspace_root() -> str | None:
    cfg = get_config()
    return None if cfg is None else cfg.workspace_root


_SERVER_ARGS_PERSIST_KEYS = (
    # Resolved by init_memory_pool from free memory on SAVE; LOAD skips that
    # computation, so they must be replayed from the archive (hybrid
    # linear-attention models crash in MambaPool(size=None) otherwise).
    "max_mamba_cache_size",
    "max_running_requests",
    "max_total_tokens",
)


def collect_server_args_overrides(server_args) -> dict:
    out = {}
    for key in _SERVER_ARGS_PERSIST_KEYS:
        val = getattr(server_args, key, None)
        if val is not None:
            out[key] = val
    return out


def apply_server_args_overrides(server_args, overrides: dict) -> None:
    if server_args is None:
        return
    for key, val in (overrides or {}).items():
        if getattr(server_args, key, None) == val:
            continue  # already resolved to the same value (frozen forks)
        try:
            setattr(server_args, key, val)
        except AttributeError:
            try:
                from sglang.srt.server_args import get_context

                get_context().override("foundry", **{key: val})
            except Exception:
                logger.warning(
                    "[Foundry] cannot override %s on frozen server_args", key
                )


def _graph_signature(server_args) -> str:
    """Stable description of which CUDA graphs a run will build.

    v3 forks carry a `cuda_graph_config` dataclass; older trees expose the
    batch-size list and cap directly. Either way the point is that an archive
    baked for one set of shapes cannot serve another.
    """
    cfg = getattr(server_args, "cuda_graph_config", None)
    if cfg is not None:
        return str(cfg)
    return str(
        [
            getattr(server_args, name, None)
            for name in (
                "cuda_graph_bs",
                "cuda_graph_max_bs",
                "disable_cuda_graph",
                "disable_cuda_graph_padding",
            )
        ]
    )


def compute_archive_fingerprint(server_args) -> dict:
    """Identify the (hardware, stack, model, layout) an archive belongs to."""
    try:
        from sglang.version import __version__ as sglang_version
    except Exception:
        sglang_version = "unknown"

    props = torch.cuda.get_device_properties(0)
    cfg = get_config()

    def arg(name, default=None):
        value = getattr(server_args, name, default)
        return default if value is None else value

    tp = int(arg("tp_size", 1))
    pp = int(arg("pp_size", 1))
    dp = int(arg("dp_size", 1))
    world = pp * tp if arg("enable_dp_attention", False) else dp * tp * pp

    return {
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_count": torch.cuda.device_count(),
        "cuda": torch.version.cuda or "unknown",
        "torch": torch.__version__,
        "sglang": sglang_version,
        "model_path": str(arg("model_path", "")),
        "quantization": str(arg("quantization", "")),
        "kv_cache_dtype": str(arg("kv_cache_dtype", "")),
        "attention_backend": str(arg("attention_backend", "")),
        "speculative_algorithm": str(arg("speculative_algorithm", "")),
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "ep": int(arg("ep_size", 1)),
        "world_size": world,
        "graphs": _graph_signature(server_args),
        "region": None
        if cfg is None
        else f"{cfg.base_addr:#x}/{cfg.region_size}/{cfg.scratch_space_size}",
    }


def _auto_decision(server_args) -> tuple[CUDAGraphExtensionMode, str]:
    cfg = get_config()
    root = Path(cfg.workspace_root)
    if not (root / "warmup_state.json").exists():
        return CUDAGraphExtensionMode.SAVE, f"no archive at {root}"

    try:
        state = load_warmup_state()
    except Exception as exc:
        return CUDAGraphExtensionMode.SAVE, f"unreadable archive ({exc})"

    want = compute_archive_fingerprint(server_args)
    have = state.fingerprint
    if not have:
        return CUDAGraphExtensionMode.SAVE, "archive predates fingerprinting"

    mismatched = [k for k, v in want.items() if have.get(k) != v]
    if mismatched:
        detail = ", ".join(
            f"{k}: archive={have.get(k)!r} run={want[k]!r}" for k in mismatched[:3]
        )
        return CUDAGraphExtensionMode.SAVE, f"fingerprint mismatch — {detail}"

    # final_alloc_offset.json is written at the very end of a rank's capture,
    # so its presence for every rank is what makes the archive complete.
    incomplete = [
        i
        for i in range(int(want["world_size"]))
        if not (root / f"rank_{i}" / "final_alloc_offset.json").exists()
    ]
    if incomplete:
        return (
            CUDAGraphExtensionMode.SAVE,
            f"incomplete archive — ranks {incomplete[:4]} never finished capture",
        )

    return CUDAGraphExtensionMode.LOAD, f"complete matching archive ({state.timestamp})"


_AUTO_RESOLVED_ENV = "FOUNDRY_AUTO_RESOLVED"


def resolve_auto_mode(server_args) -> CUDAGraphExtensionMode:
    """Turn AUTO into SAVE or LOAD.

    Call this only where ``server_args`` is fully resolved: the fingerprint
    covers derived fields (attention backend, CUDA-graph shapes) that are
    still unset while ``ServerArgs.__post_init__`` is running, and a decision
    taken there would never match one taken at save time.

    The first process to decide exports the result, so every rank in the tree
    agrees — the parent probes the archive before any rank has written to it,
    and workers must not re-probe a half-written one.
    """
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.AUTO:
        return None if cfg is None else cfg.mode

    inherited = os.environ.get(_AUTO_RESOLVED_ENV)
    if inherited in (CUDAGraphExtensionMode.SAVE.value, CUDAGraphExtensionMode.LOAD.value):
        cfg.mode = CUDAGraphExtensionMode(inherited)
        logger.info("[Foundry] auto mode inherited from parent: %s", inherited)
        return cfg.mode

    try:
        decision, reason = _auto_decision(server_args)
    except Exception as exc:  # never let mode resolution take the server down
        decision, reason = CUDAGraphExtensionMode.SAVE, f"probe failed ({exc})"

    cfg.mode = decision
    os.environ[_AUTO_RESOLVED_ENV] = decision.value
    logger.info("[Foundry] auto mode resolved to %s — %s", decision.value, reason)
    return decision


def create_warmup_state(
    memory_pool_config: dict | None = None,
    server_args_overrides: dict | None = None,
    fingerprint: dict | None = None,
) -> WarmupState:
    try:
        from sglang.version import __version__ as sglang_version
    except Exception:
        sglang_version = "unknown"

    props = torch.cuda.get_device_properties(0)
    return WarmupState(
        sglang_version=sglang_version,
        timestamp=datetime.now().isoformat(),
        cuda_version=torch.version.cuda or "unknown",
        gpu_name=props.name,
        gpu_total_memory=props.total_memory,
        memory_pool_config=memory_pool_config or {},
        server_args_overrides=server_args_overrides or {},
        fingerprint=fingerprint or {},
    )


def save_warmup_state(state: WarmupState) -> None:
    workspace_root = _workspace_root()
    if workspace_root is None:
        return
    ext_state = get_state()
    path = os.path.join(workspace_root, "warmup_state.json")
    if ext_state is not None and ext_state.rank != 0 and os.path.exists(path):
        return
    os.makedirs(workspace_root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2)
    logger.info("[Foundry] Saved SGLang warmup state to %s", path)


def load_warmup_state() -> WarmupState:
    workspace_root = _workspace_root()
    if workspace_root is None:
        raise RuntimeError("Foundry workspace_root is not initialized")
    path = os.path.join(workspace_root, "warmup_state.json")
    if not os.path.exists(path):
        raise RuntimeError(f"Foundry warmup state file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    valid = set(WarmupState.__dataclass_fields__.keys())
    return WarmupState(**{k: v for k, v in data.items() if k in valid})


def setup_graph_extension(server_args, tp_rank: int, pp_rank: int, dp_rank: int | None) -> None:
    """Set up the VMM region before SGLang initializes NCCL/process groups."""
    global _state
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    # server_args is fully resolved by now; safe to decide AUTO here for runs
    # whose spawn sites the integration did not patch.
    resolve_auto_mode(server_args)

    t0 = time.perf_counter()
    rank = compute_workspace_rank(server_args, tp_rank, pp_rank, dp_rank)
    Path(cfg.workspace_root).mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(cfg.workspace_root) / f"rank_{rank}"
    cfg.workspace_dir = str(workspace_dir)
    logger.info("[Foundry] SGLang rank=%d workspace_dir=%s", rank, workspace_dir)

    if cfg.mode == CUDAGraphExtensionMode.SAVE:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
    elif cfg.mode == CUDAGraphExtensionMode.LOAD:
        cge.set_skip_fatbin_processing(True)
        if not workspace_dir.exists():
            raise RuntimeError(f"Foundry workspace for rank {rank} does not exist: {workspace_dir}")
        cge.load_cuda_modules_and_libraries(str(workspace_dir))

    region_size = parse_size(cfg.region_size)
    cge.set_allocation_region(cfg.base_addr, region_size)
    _ = torch._C._cuda_getCurrentBlasHandle()
    _state = CUDAGraphExtensionState(rank=rank)
    logger.info(
        "[Foundry] SGLang graph extension setup completed in %.3f s",
        time.perf_counter() - t0,
    )


def skip_to_scratch_boundary() -> None:
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    scratch = parse_size(cfg.scratch_space_size)
    current = cge.get_current_alloc_offset()
    if current > scratch:
        logger.warning(
            "[Foundry] Current allocation offset %d exceeds scratch size %d",
            current,
            scratch,
        )
        return
    cge.set_current_alloc_offset(scratch)
    logger.info("[Foundry] SGLang skipped allocator to scratch boundary %d", scratch)


def capture_final_alloc_offset() -> int:
    global _final_alloc_offset
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return 0
    _final_alloc_offset = cge.get_current_alloc_offset()
    if cfg.workspace_dir is not None:
        path = os.path.join(cfg.workspace_dir, "final_alloc_offset.json")
        with open(path, "w") as f:
            json.dump({"final_alloc_offset": _final_alloc_offset}, f)
    if cfg.workspace_root is not None:
        warmup_state_path = os.path.join(cfg.workspace_root, "warmup_state.json")
        if os.path.exists(warmup_state_path):
            state = load_warmup_state()
            state.final_alloc_offset = _final_alloc_offset
            save_warmup_state(state)
    logger.info("[Foundry] SGLang final_alloc_offset=%d", _final_alloc_offset)
    return _final_alloc_offset


_preallocated = False


def preallocate_for_load_mode() -> None:
    """Map the region up to SAVE's watermark. Idempotent: BCG prefill restore
    runs before the decode runner and needs the mapping first; the second
    caller must not tear down and rebuild the preallocation under live
    tensor views."""
    global _preallocated
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        return
    if _preallocated:
        return
    _preallocated = True
    final = 0
    if cfg.workspace_dir is not None:
        path = os.path.join(cfg.workspace_dir, "final_alloc_offset.json")
        if os.path.exists(path):
            with open(path) as f:
                final = json.load(f).get("final_alloc_offset", 0)
    if final <= 0:
        final = load_warmup_state().final_alloc_offset
    remaining = final - cge.get_current_alloc_offset()
    if remaining > 0:
        cge.preallocate_region(remaining)


def disable_sync_on_free() -> None:
    """Drop the hook's device-sync-before-unmap once the multi-threaded
    weight-loading phase is over — graph capture's alloc/free churn would
    otherwise pay a full device sync per freed allocation."""
    setter = getattr(cge, "set_sync_on_free", None)
    if setter is not None:
        setter(False)


def log_hook_stats(label: str) -> None:
    """Dump the hook's per-entry-point call counts and time at a phase edge."""
    fn = getattr(cge, "report_hook_stats", None)
    if fn is not None:
        try:
            fn(label)
        except Exception:
            pass


def log_alloc_offset(label: str) -> None:
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    offset = cge.get_current_alloc_offset()
    logger.info(
        "[Foundry] SGLang alloc_offset[%s]=%d (%.2f MB)",
        label,
        offset,
        offset / (1024 * 1024),
    )
    log_hook_stats(label)


def setup_ld_preload_env() -> None:
    current = os.environ.get("LD_PRELOAD", "")
    for path in (get_hook_library_path(), get_nvshmem_host_path()):
        if path and path not in current:
            current = f"{path}:{current}" if current else path
    if current:
        os.environ["LD_PRELOAD"] = current
    mode = get_graph_extension_mode()
    if mode != CUDAGraphExtensionMode.NONE:
        os.environ["FOUNDRY_MODE"] = mode.value
    os.environ["FOUNDRY_SPAWN_T0_NS"] = str(time.perf_counter_ns())
