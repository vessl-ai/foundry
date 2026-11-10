# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Persistent on-disk cache for FlashInfer's SM90 GDN prefill CuTe-DSL kernels.

FlashInfer's ``gdn_kernels/delta_rule_dsl`` compiles its SM90 kernels with
``cute.compile`` behind a process-local dict. :func:`install` routes that
compile through FlashInfer's ``build_and_load_cute_dsl_kernel`` (its on-disk
``.o`` cache under ``~/.cache/flashinfer/<version>/<arch>/cached_ops/``), so
one process compiles and every later one loads. Applied on SAVE and LOAD
before the capture loop when SGLang's FlashInfer GDN kernel module is loaded;
``FOUNDRY_FLASHINFER_GDN_CUTE_CACHE=0`` turns it off (in-process compile,
8-10 s per scheduler at the first extend).

SGLang's own CuTe AOT cache (``_enable_gdn_aot_cache`` in its
``gdn_flashinfer`` module) replaces ``_in_mem_compile_cache`` with a dict
that exports every stored kernel to ``SGLANG_CUTE_AOT_CACHE_DIR``. When that
cache is present, it already persists these kernels, and a kernel from
FlashInfer's ``.o`` cache cannot go into it: ``build_and_load_cute_dsl_kernel``
returns the kernel reloaded from the ``.o`` (a TVM-FFI function without
``export_to_c``), so SGLang's export would raise at the first extend. Foundry
then leaves ``cached_compile`` to FlashInfer: :func:`install` does not patch
when the cache is already SGLang's, and the patched function hands the call
back to FlashInfer's own when SGLang installs its cache later.
"""

from __future__ import annotations

import glob
import hashlib
import importlib
import logging
import os
import re
import sys
from typing import Any

logger = logging.getLogger(__name__)

CACHE_MODULE = "flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache"
CACHE_PACKAGE = "flashinfer.gdn_kernels.delta_rule_dsl"
SGLANG_GDN_FLASHINFER = "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer"
# cached_ops/<this>_<arch>_cute_dsl/<kernel name>.o
CACHED_OPS_MODULE = "gdn_delta_rule_dsl"
_UNSTABLE_KEY = re.compile(r"0x[0-9a-fA-F]{6,}")

_installed: str | None = None
OPT_OUT_ENV = "FOUNDRY_FLASHINFER_GDN_CUTE_CACHE"


def enabled() -> bool:
    """On unless ``FOUNDRY_FLASHINFER_GDN_CUTE_CACHE=0``."""
    return os.environ.get(OPT_OUT_ENV) != "0"


def kernel_name(func: Any, key_text: str) -> str:
    """Per-kernel cache name: the kernel class plus a digest of the compile
    key (dtypes, static shapes, options), the sole per-kernel cache key of
    FlashInfer's ``.o`` cache."""
    return f"{type(func).__qualname__}_{hashlib.sha256(key_text.encode()).hexdigest()[:16]}"


def gdn_flashinfer_in_use() -> bool:
    """SGLang imports its FlashInfer GDN kernel module only when a hybrid GDN
    model runs with the FlashInfer prefill or decode kernel."""
    return SGLANG_GDN_FLASHINFER in sys.modules


def sglang_owns_cache(cache: Any) -> bool:
    """``cache`` (FlashInfer's ``_in_mem_compile_cache``) was put there by
    SGLang, i.e. SGLang persists the compiled kernels itself."""
    return type(cache).__module__.startswith("sglang.")


def install() -> str | None:
    """Route ``custom_compile_cache.cached_compile`` (and the name as already
    imported by the delta_rule modules) through FlashInfer's persistent
    ``.o`` cache, unless SGLang's CuTe AOT cache already holds the dict (see
    the module docstring). Returns a status string for the log, or None when
    this FlashInfer has no such module or helper (nothing to do). Idempotent."""
    global _installed
    if _installed is not None:
        return _installed
    try:
        ccc = importlib.import_module(CACHE_MODULE)
        cute_dsl_core = importlib.import_module("flashinfer.jit.cute_dsl_core")
        cute = importlib.import_module("cutlass.cute")
    except Exception as exc:
        logger.debug("[Foundry] FlashInfer CuTe-DSL cache patch not applicable: %r", exc)
        return None
    build_and_load = getattr(cute_dsl_core, "build_and_load_cute_dsl_kernel", None)
    original = getattr(ccc, "cached_compile", None)
    in_mem = getattr(ccc, "_in_mem_compile_cache", None)
    options_key = getattr(ccc, "_compile_options_key", None)
    if build_and_load is None or original is None or in_mem is None or options_key is None:
        logger.debug("[Foundry] FlashInfer CuTe-DSL cache patch not applicable (API)")
        return None
    if sglang_owns_cache(in_mem):
        cache_type = type(in_mem)
        _installed = (
            f"not patched: SGLang's CuTe AOT cache ({cache_type.__module__}."
            f"{cache_type.__qualname__}) persists the delta_rule_dsl kernels"
        )
        return _installed
    sources = sorted(glob.glob(os.path.join(os.path.dirname(ccc.__file__), "*.py")))

    def cached_compile(func, *args, compile_options=None, **kwargs):
        cache = ccc._in_mem_compile_cache
        if sglang_owns_cache(cache):
            return original(func, *args, compile_options=compile_options, **kwargs)
        key = (func._get_compile_key(), options_key(compile_options))
        compiled = cache.get(key)
        if compiled is not None:
            return compiled
        key_text = repr(key)

        def compile_fn():
            return cute.compile[compile_options](func, *args, **kwargs)

        if _UNSTABLE_KEY.search(key_text):
            logger.info(
                "[Foundry] CuTe-DSL cache: %s key holds an address, compiled in process",
                type(func).__qualname__,
            )
            compiled = compile_fn()
        else:
            name = kernel_name(func, key_text)
            logger.info("[Foundry] CuTe-DSL cache: %s <- %s", name, key_text)
            compiled = build_and_load(CACHED_OPS_MODULE, name, compile_fn, sources)
        cache[key] = compiled
        return compiled

    ccc.cached_compile = cached_compile
    rebound = []
    for name, module in list(sys.modules.items()):
        if (
            name.startswith(CACHE_PACKAGE + ".")
            and module is not ccc
            and getattr(module, "cached_compile", None) is original
        ):
            module.cached_compile = cached_compile
            rebound.append(name.rsplit(".", 1)[-1])
    _installed = "delta_rule_dsl cached_compile -> build_and_load_cute_dsl_kernel" + (
        f" (rebound in {', '.join(rebound)})" if rebound else ""
    )
    return _installed
