"""Pin Triton autotune decisions across SAVE and LOAD.

Kernels that run *eagerly* on a restored graph path (BCG break closures such
as linear-attention / KDA ops, eager logits tails) still pick their Triton
config at runtime.  Many autotuners key only on constexpr tile sizes (e.g.
``key=["BC", "IS_VARLEN"]``), so the config is chosen by benchmarking at the
*first* call in the process and then reused for every sequence length.

SAVE makes that first call during prefill capture (first capture size); LOAD
skips capture and makes it at the first user request (a different length,
different timings) -> a different config -> different accumulation order ->
1-ulp differences in a few tokens that grow into divergent greedy output
(observed: Solar-Pro-4 W4AFP8 + DSPARK, KDA break on rank 1, token 12).

Fix: SAVE records every live Autotuner's ``cache`` (key -> config index) per
rank; LOAD installs those decisions before any eager kernel runs, so eager
kernels use exactly the configs the SAVE process used.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys

logger = logging.getLogger(__name__)

_FILE = "autotune_cache.pkl"


def enabled() -> bool:
    return os.environ.get("FOUNDRY_AUTOTUNE_PIN", "1") == "1"


def _iter_autotuners():
    try:
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import KernelInterface
    except Exception:  # triton not available
        return
    seen: set = set()
    for mname, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            items = list(vars(mod).items())
        except Exception:
            continue
        for aname, obj in items:
            # Unwrap Heuristics(...Autotuner...) chains; only touch triton
            # KernelInterface objects (arbitrary objects can raise on getattr).
            cur, depth = obj, 0
            while depth < 4:
                try:
                    if not isinstance(cur, KernelInterface):
                        break
                    if isinstance(cur, Autotuner):
                        if id(cur) not in seen:
                            seen.add(id(cur))
                            yield f"{mname}.{aname}", cur
                        break
                    cur = cur.fn
                except Exception:
                    break
                depth += 1


def _cfg_sig(cfg) -> tuple:
    return (
        tuple(sorted((cfg.kwargs or {}).items())),
        cfg.num_warps,
        cfg.num_stages,
        getattr(cfg, "num_ctas", 1),
        getattr(cfg, "maxnreg", None),
    )


def save(workspace_dir: str, tag: str = "") -> None:
    if not enabled():
        return
    out: dict = {}
    n = 0
    for name, at in _iter_autotuners():
        sigs = {_cfg_sig(c): i for i, c in enumerate(at.configs)}
        ent = {}
        for key, cfg in at.cache.items():
            idx = sigs.get(_cfg_sig(cfg))
            if idx is None:
                continue
            ent[key] = idx
        if ent:
            out[name] = ent
            n += len(ent)
    path = os.path.join(workspace_dir, _FILE)
    try:
        with open(path, "wb") as f:
            pickle.dump(out, f)
    except Exception as exc:  # unpicklable key element
        logger.warning("[Foundry] autotune pin: save failed (%s)", exc)
        return
    logger.info(
        "[Foundry] autotune pin: saved %d decisions across %d autotuners%s",
        n, len(out), f" ({tag})" if tag else "",
    )


def load(workspace_dir: str) -> None:
    if not enabled():
        return
    path = os.path.join(workspace_dir, _FILE)
    if not os.path.exists(path):
        return
    try:
        ref = pickle.load(open(path, "rb"))
    except Exception as exc:
        logger.warning("[Foundry] autotune pin: load failed (%s)", exc)
        return
    installed = missing = 0
    live = dict(_iter_autotuners())
    for name, ent in ref.items():
        at = live.get(name)
        if at is None:
            missing += len(ent)
            continue
        for key, idx in ent.items():
            if 0 <= idx < len(at.configs):
                at.cache[key] = at.configs[idx]
                installed += 1
    logger.info(
        "[Foundry] autotune pin: installed %d SAVE decisions (%d for autotuners not "
        "loaded yet)", installed, missing,
    )
