# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Pin the decode runner's graph-registry slots to fixed addresses so a
restored decode graph reads the slots the runner actually writes.

A decode CUDA graph bakes in the addresses of the runner's ``CudaGraphBuffer
Registry`` slots (input_ids, positions, out_cache_loc, ...) — the exact tensors
``slice_for`` hands to the model on every replay.  Those slots are backed by
the torch caching allocator, whose layout differs between SAVE (graphs
captured) and LOAD (graphs restored): on Solar TP4 the decode slots land tens
of GB apart because the decode runner initializes *after* prefill, whose
capture-vs-restore memory footprint differs.  The graph then reads the SAVE
address while the runner writes the LOAD address → garbage from the first (or
second) decode step.  Prefill(BCG) is immune because its buffers belong to the
first runner, whose layout matches.

Fix: record each registry slot's address on SAVE (right after capture, so it
is exactly what the graph baked in); on LOAD, before the graphs are restored,
replace each slot's backing tensor with a ``tensor_from_ptr`` view at the
recorded SAVE address (foundry's deterministic region is extended to cover it).
``slice_for`` then slices the SAVE-address buffer and the restored graph reads
what the runner populates.

Driven from hooks.py patched_capture; env FOUNDRY_DECODE_PIN_BUFFERS=0 disables.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

_ARCHIVE_NAME = "decode_buffers.json"
_load_records: dict[str, dict[str, Any]] | None = None
_region_extended = False


def enabled() -> bool:
    return os.environ.get("FOUNDRY_DECODE_PIN_BUFFERS", "1") == "1"


def _workspace_dir() -> str:
    from foundry.integration.sglang.config import get_config

    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry workspace_dir is not initialized")
    return cfg.workspace_dir


def _iter_slots(registry: Any):
    slots = getattr(registry, "_slots", None)
    if not slots:
        return
    for name, slot in slots.items():
        buf = getattr(slot, "buffer", None)
        yield name, slot, buf


def record_registry_slots(registry: Any) -> None:
    """Archive each CUDA slot's address (call right after SAVE capture)."""
    meta: dict[str, Any] = {}
    for name, _slot, buf in _iter_slots(registry):
        if isinstance(buf, torch.Tensor) and buf.is_cuda:
            meta[name] = {
                "ptr": buf.data_ptr(),
                "shape": list(buf.shape),
                "stride": list(buf.stride()),
                "dtype": str(buf.dtype).removeprefix("torch."),
                "device": buf.device.index or 0,
            }
    path = os.path.join(_workspace_dir(), _ARCHIVE_NAME)
    with open(path, "w") as fh:
        json.dump(meta, fh)
    logger.info("[Foundry] decode buffers: archived %d registry slots", len(meta))


def _load_saved() -> dict[str, dict[str, Any]]:
    global _load_records
    if _load_records is None:
        path = os.path.join(_workspace_dir(), _ARCHIVE_NAME)
        _load_records = json.load(open(path)) if os.path.exists(path) else {}
        if not _load_records:
            logger.warning("[Foundry] decode buffers: no archive at %s", path)
    return _load_records


def _ensure_region_mapped(saved: dict[str, dict[str, Any]]) -> None:
    """Extend the mapped VMM region to cover the highest SAVE slot end."""
    global _region_extended
    if _region_extended or not saved:
        return
    from foundry import ops as cge
    from foundry.integration.sglang.config import get_config

    base = get_config().base_addr
    max_end = 0
    for m in saved.values():
        numel = 1
        for s in m["shape"]:
            numel *= s
        elt = torch.empty(0, dtype=getattr(torch, m["dtype"])).element_size()
        max_end = max(max_end, (m["ptr"] - base) + numel * elt)
    cur = cge.get_current_alloc_offset()
    if max_end > cur:
        cge.preallocate_region(max_end - cur)
        logger.info(
            "[Foundry] decode buffers: extended region %d -> %d (+%d MB)",
            cur, max_end, (max_end - cur) >> 20,
        )
    _region_extended = True


def pin_registry_slots(registry: Any) -> None:
    """Replace each registry slot's backing tensor with a view at its SAVE
    address (call on LOAD before the decode graphs are restored)."""
    saved = _load_saved()
    if not saved:
        return
    _ensure_region_mapped(saved)
    from foundry import ops

    pinned = 0
    for name, slot, buf in _iter_slots(registry):
        m = saved.get(name)
        if m is None or not isinstance(buf, torch.Tensor) or not buf.is_cuda:
            continue
        try:
            view = ops.tensor_from_ptr(
                m["ptr"], m["shape"], m["stride"],
                getattr(torch, m["dtype"]), m["device"],
            )
            view.copy_(buf)
            slot.buffer = view
            pinned += 1
        except Exception:
            logger.exception("[Foundry] decode buffers: pin failed for %s", name)
    logger.info(
        "[Foundry] decode buffers: pinned %d/%d registry slots to SAVE addresses",
        pinned, len(saved),
    )


# ---------------------------------------------------------------------------
# (A) Graph-side rebind: instead of forcing the runner's slots to the SAVE
# address (which the 4-stage buffer abstraction defeats), rewrite the archived
# decode-graph node params so the restored graph reads the runner's *actual*
# LOAD slot addresses.  Only bytes that fall inside a known SAVE slot range are
# shifted by (LOAD_base - SAVE_base); graph-internal (allocator-region) tensors
# are left untouched so the allocator-event cursor still validates.
# ---------------------------------------------------------------------------

def build_slot_remap(registry: Any) -> dict[int, int]:
    """{save_base: load_base} per CUDA slot — exact base addresses only.

    Decode graphs reference each slot at its base (slice_for gives buffer[:n],
    offset 0), so we match the exact SAVE base and never a range: a range would
    also catch graph-internal (allocator) tensors that happen to land inside a
    large slot's extent (e.g. input_embeds' 8 MB) and corrupt them."""
    saved = _load_saved()
    remap: dict[int, int] = {}
    dbg = os.environ.get("FOUNDRY_DECODE_PIN_DEBUG") == "1"
    for name, _slot, buf in _iter_slots(registry):
        m = saved.get(name)
        if m is None or not isinstance(buf, torch.Tensor) or not buf.is_cuda:
            continue
        remap[m["ptr"]] = buf.data_ptr()
        if dbg:
            logger.info(
                "[Foundry] remap %s: 0x%x -> 0x%x", name, m["ptr"], buf.data_ptr()
            )
    return remap


def _rebind_hex(hexstr: str, remap: dict[int, int]) -> tuple[str, int]:
    if not hexstr:
        return hexstr, 0
    b = bytearray.fromhex(hexstr)
    changed = 0
    for off in range(0, len(b) - 7, 8):
        v = int.from_bytes(b[off:off + 8], "little")
        nv = remap.get(v)
        if nv is not None:
            b[off:off + 8] = nv.to_bytes(8, "little")
            changed += 1
    return b.hex(), changed


def rebind_graph_dict(gjson: dict, remap: dict[int, int]) -> int:
    """Rewrite every kernel param / arg buffer address that lands in a slot
    range. Returns the number of 8-byte words shifted."""
    changed = 0
    for node in gjson.get("nodes", []):
        p = node.get("params", {})
        for kp in p.get("kernelParams", []):
            if isinstance(kp, dict) and kp.get("value_hex"):
                kp["value_hex"], c = _rebind_hex(kp["value_hex"], remap)
                changed += c
        ab = p.get("extra_argBuffer_hex")
        if ab:
            p["extra_argBuffer_hex"], c = _rebind_hex(ab, remap)
            changed += c
    return changed


def rebind_decode_graphs_inplace(registry: Any, graph_paths: list[str]) -> list[str]:
    """Rebind each decode graph json to the runner's LOAD slot addresses,
    writing back to the *original* path (so graph_manifest's filename
    references and on-demand template matching still resolve).  The original
    bytes are moved to ``<path>.bak``; call restore_decode_graphs after load.
    Returns the list of paths that were rewritten."""
    if not enabled():
        return []
    remap = build_slot_remap(registry)
    if not remap:
        logger.warning("[Foundry] decode rebind: empty slot remap; skipping")
        return []
    backed: list[str] = []
    total = 0
    hidden = 0
    for path in graph_paths:
        gjson = json.load(open(path))
        total += rebind_graph_dict(gjson, remap)
        os.replace(path, path + ".bak")
        with open(path, "w") as fh:
            json.dump(gjson, fh)
        backed.append(path)
        # start_graph_builds_impl prefers the sibling ``.cugraph`` binary and
        # ignores the JSON whenever the binary is valid, so the rebound JSON
        # only takes effect if the binary is hidden.  Move it aside; restore
        # puts it back.
        cug = _cugraph_sibling(path)
        if cug and os.path.exists(cug):
            os.replace(cug, cug + ".hidden")
            hidden += 1
    logger.info(
        "[Foundry] decode rebind: shifted %d slot refs across %d graphs "
        "(%d slots); hid %d .cugraph binaries so JSON is used",
        total, len(graph_paths), len(remap), hidden,
    )
    return backed


def _cugraph_sibling(json_path: str) -> str | None:
    if json_path.endswith(".json"):
        return json_path[:-5] + ".cugraph"
    return None


def _rebind_bytes(data: bytearray, remap: dict[int, int]) -> int:
    """Substitute each SAVE base address (8-byte little-endian) with its LOAD
    base wherever it appears in the raw binary.  Returns words changed."""
    changed = 0
    for save, load in remap.items():
        sb = save.to_bytes(8, "little")
        lb = load.to_bytes(8, "little")
        start = 0
        while True:
            i = data.find(sb, start)
            if i < 0:
                break
            data[i:i + 8] = lb
            changed += 1
            start = i + 8
    return changed


def rebind_decode_binaries_inplace(
    registry: Any, graph_paths: list[str]
) -> list[str]:
    """Rebind the decode ``.cugraph`` *binaries* to the runner's LOAD slot
    addresses.  start_graph_builds_impl reads the sibling ``.cugraph`` binary
    (not the JSON) whenever it is valid, so the binary is the real load source.
    Each slot's absolute SAVE pointer is stored raw in the binary, so a plain
    8-byte substitution of SAVE->LOAD base is exact and self-validating (the
    SAVE addresses live in the 0x60xx… VMM range and never collide with the
    file's small structural offsets).  Originals move to ``<cug>.bak``; call
    restore_decode_binaries after load.  Returns the .cugraph paths rewritten."""
    if not enabled():
        return []
    remap = build_slot_remap(registry)
    if not remap:
        logger.warning("[Foundry] decode binary rebind: empty slot remap; skipping")
        return []
    backed: list[str] = []
    total = 0
    for path in graph_paths:
        cug = _cugraph_sibling(path)
        if not cug or not os.path.exists(cug):
            continue
        data = bytearray(open(cug, "rb").read())
        total += _rebind_bytes(data, remap)
        os.replace(cug, cug + ".bak")
        with open(cug, "wb") as fh:
            fh.write(data)
        backed.append(cug)
    logger.info(
        "[Foundry] decode binary rebind: shifted %d refs across %d cugraphs "
        "(%d slots)",
        total, len(backed), len(remap),
    )
    return backed


def restore_decode_binaries(paths: list[str]) -> None:
    """Restore original ``.cugraph`` bytes from the .bak siblings."""
    for cug in paths:
        bak = cug + ".bak"
        if os.path.exists(bak):
            os.replace(bak, cug)


def restore_decode_graphs(paths: list[str]) -> None:
    """Restore original graph json bytes and the hidden .cugraph binaries."""
    for path in paths:
        bak = path + ".bak"
        if os.path.exists(bak):
            os.replace(bak, path)
        cug = _cugraph_sibling(path)
        if cug and os.path.exists(cug + ".hidden"):
            os.replace(cug + ".hidden", cug)


_replay_probe_installed = False


def install_replay_probe() -> None:
    """Log the registry slot addresses at the FIRST decode replay, to compare
    against the load-time addresses build_slot_remap saw."""
    global _replay_probe_installed
    if _replay_probe_installed or os.environ.get("FOUNDRY_DECODE_REPLAY_PROBE") != "1":
        return
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    orig = DecodeCudaGraphRunner.execute
    done = {"n": 0}

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        if done["n"] < 1:
            reg = getattr(self, "buffer_registry", None)
            if reg is not None:
                for name, _slot, buf in _iter_slots(reg):
                    if isinstance(buf, torch.Tensor) and buf.is_cuda:
                        logger.info(
                            "[Foundry] REPLAY slot %s: 0x%x", name, buf.data_ptr()
                        )
            done["n"] += 1
        return orig(self, *args, **kwargs)

    DecodeCudaGraphRunner.execute = patched
    _replay_probe_installed = True
    logger.info("[Foundry] replay probe installed")
