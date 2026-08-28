# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime monkey-patch installer for the Foundry SGLang integration."""

from __future__ import annotations

import functools
import logging
import os
import time
from dataclasses import asdict

from foundry.integration.sglang import runtime as rt
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    get_graph_extension_mode,
    get_workspace_root,
    load_graph_extension_config,
)

logger = logging.getLogger(__name__)
_INSTALLED = False


def _ep_lazy_init_needed() -> bool:
    """True when the DeepEP all-to-all backend is active, so pre-capture lazy
    init (NVSHMEM buffer, DeepGEMM JIT) must be warmed up outside stream capture."""
    try:
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        return get_moe_a2a_backend().is_deepep()
    except Exception:
        return False


def _pre_capture_warmup_needed(cuda_graph_runner) -> bool:
    """True when foundry's suppressed warmup forwards would have triggered
    lazy init that must not fire inside stream capture.

    Two known sources:
    - DeepEP EP: NVSHMEM buffer creation + DeepGEMM per-shape JIT.
    - TP > 1: the tensor-parallel path runs @torch.compile'd helpers that
      single-GPU/DP never hit (e.g. vocab_parallel_embedding's
      get_masked_input_and_mask). Their first call inductor-compiles and
      copies CPU constants to device — illegal inside the capture window
      ("Cannot copy between CPU and CUDA tensors during CUDA graph
      capture"). NCCL/pynccl first-collective lazy init rides along too.
    """
    if _ep_lazy_init_needed():
        return True
    tp_size = getattr(
        getattr(cuda_graph_runner, "model_runner", None), "tp_size", None
    )
    if tp_size is None:
        tp_size = getattr(cuda_graph_runner, "tp_size", 1)
    return (tp_size or 1) > 1


def _resolve_dp_rank(model_runner) -> int | None:
    dp_rank = getattr(model_runner, "dp_rank", None)
    if dp_rank is not None:
        return dp_rank

    server_args = model_runner.server_args
    if getattr(server_args, "enable_dp_attention", False):
        from sglang.srt.layers.dp_attention import compute_dp_attention_world_info

        # Returns (attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size) — a
        # 4-tuple. Take attn_dp_rank (3rd) and discard the rest. (Unpacking into
        # 3 targets would raise ValueError.)
        _, _, dp_rank, _ = compute_dp_attention_world_info(
            server_args.enable_dp_attention,
            model_runner.tp_rank,
            server_args.tp_size,
            server_args.dp_size,
            server_args.attn_cp_size,
        )
        return dp_rank

    if getattr(server_args, "dp_size", 1) > 1:
        raise RuntimeError(
            "Foundry SGLang integration cannot derive regular DP rank because "
            "ModelRunner.dp_rank is absent. Preserve the constructor dp_rank on "
            "ModelRunner before initializing torch distributed."
        )

    return None


def install_hooks(server_args) -> None:
    global _INSTALLED
    cfg_path = getattr(server_args, "foundry_graph_extension_config_path", None)
    if not cfg_path:
        return
    if _INSTALLED:
        return

    t0_ns = os.environ.get("FOUNDRY_SPAWN_T0_NS")
    if t0_ns:
        logger.info(
            "[Foundry] SGLang spawn -> install_hooks: %.1f ms",
            (time.perf_counter_ns() - int(t0_ns)) / 1e6,
        )

    load_graph_extension_config(cfg_path)
    # AUTO is deliberately NOT resolved here: in the parent this runs from
    # ServerArgs.__post_init__, before the resolution pipeline fills in the
    # derived fields the fingerprint covers. It is resolved at the spawn sites
    # and in setup_graph_extension, where server_args is complete.
    logger.info(
        "[Foundry] SGLang hooks installing: mode=%s workspace=%s",
        get_graph_extension_mode().value,
        get_workspace_root(),
    )

    era = _detect_engine_era()
    _patch_init_torch_distributed()
    if era == "v3":
        # Post runner-backend refactor (runner/ + runner_backend/ split, e.g.
        # the vessl-ai production fork): memory pool resolution lives in
        # KVCacheConfigurator and graph capture behind FullCudaGraphBackend.
        _patch_resolve_memory_pool_v3()
        _patch_cuda_graph_capture_v3()
    else:
        _patch_init_memory_pool()
        _patch_load_model()
        _patch_kernel_warmup()
        _patch_cuda_graph_capture()
    _patch_spawn_sites()

    _INSTALLED = True
    logger.info("[Foundry] SGLang hooks installed (engine era: %s)", era)


def _detect_engine_era() -> str:
    """v3 = runner/runner_backend refactor; v2 = monolithic CudaGraphRunner."""
    try:
        import sglang.srt.model_executor.runner_backend.full_cuda_graph_backend  # noqa: F401

        return "v3"
    except Exception:
        return "v2"


def _patch_resolve_memory_pool_v3() -> None:
    """SAVE persists the profiled MemoryPoolConfig; LOAD replays it.

    In the v3 engine, `KVCacheConfigurator.configure` calls
    `_resolve_memory_pool_config(pre_model_load_memory)` which profiles free
    GPU memory — non-reproducible across runs. Everything downstream of the
    resolved config (`_derive_pool_sizes` -> `_init_pools`) is deterministic,
    so replaying just the resolved config on LOAD keeps the allocation
    trajectory identical to SAVE.
    """
    from dataclasses import asdict

    from sglang.srt.mem_cache import kv_cache_configurator as kcc
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

    cls = kcc.KVCacheConfigurator
    orig = cls._resolve_memory_pool_config

    @functools.wraps(orig)
    def patched(self, pre_model_load_memory):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self, pre_model_load_memory)

        # Weight loading (the multi-threaded phase the free-side device sync
        # protects) is complete by the time the memory pool is resolved.
        rt.disable_sync_on_free()

        if mode == CUDAGraphExtensionMode.LOAD:
            import torch

            rt.log_alloc_offset("before_resolve_memory_pool")
            state = rt.load_warmup_state()
            if not state.memory_pool_config:
                raise RuntimeError("Foundry LOAD requires memory_pool_config")
            rt.apply_server_args_overrides(
                getattr(self, "server_args", None) or _current_server_args(),
                state.server_args_overrides,
            )
            # Mirror SAVE's free-memory profile side effect on the caching
            # allocator (see the v2 hook for the full rationale).
            torch.cuda.empty_cache()
            logger.info("[Foundry] SGLang reused saved memory pool config (v3)")
            return MemoryPoolConfig(**state.memory_pool_config)

        rt.log_alloc_offset("before_resolve_memory_pool")
        config = orig(self, pre_model_load_memory)
        rt.log_alloc_offset("after_resolve_memory_pool")
        sa = getattr(self, "server_args", None) or _current_server_args()
        state = rt.create_warmup_state(
            asdict(config),
            rt.collect_server_args_overrides(sa),
            rt.compute_archive_fingerprint(sa),
        )
        rt.save_warmup_state(state)
        return config

    cls._resolve_memory_pool_config = patched


def _current_server_args():
    from sglang.srt.server_args import get_global_server_args

    try:
        return get_global_server_args()
    except Exception:
        return None


def _patch_cuda_graph_capture_v3() -> None:
    """v3 capture/load seams.

    SAVE: wrap `FullCudaGraphBackend.capture_one` — skip the two warmup
    forwards (their non-deterministic activation allocations poison the
    caching allocator relative to LOAD), capture into a foundry graph, and
    save it keyed by ShapeKey.size. Post-capture, write the manifest, pack
    fatbins and record the final VMM watermark.

    LOAD: let the upstream `capture()` run unchanged — `warmup()`, buffer
    prep, per-shape `capture_prepare` and `init_forward_metadata_out_graph`
    all execute exactly as on SAVE (symmetric allocation trajectory for
    free) — but `capture_one` neither runs the forward nor captures: after
    the loop, all archived graphs are loaded in one pass and slotted into
    the backend's `_graphs`/`_outputs` maps, which `replay()` reads.
    """
    from sglang.srt.model_executor.runner import decode_cuda_graph_runner as dcgr
    from sglang.srt.model_executor.runner_backend import (
        full_cuda_graph_backend as fcgb,
    )

    backend_cls = fcgb.FullCudaGraphBackend
    runner_cls = dcgr.DecodeCudaGraphRunner
    orig_capture_one = backend_cls.capture_one
    orig_capture = runner_cls.capture

    @functools.wraps(orig_capture_one)
    def patched_capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig_capture_one(
                self,
                shape_key,
                forward_fn,
                capture_inputs=capture_inputs,
                post_warmup_hook=post_warmup_hook,
            )

        TORCH_CHECK = (
            shape_key.stream_idx is None
            and shape_key.variant_label is None
            and shape_key.dsa_variant is None
        )
        if not TORCH_CHECK:
            raise RuntimeError(
                f"[Foundry] unsupported ShapeKey variant for persistence: {shape_key}"
            )

        if mode == CUDAGraphExtensionMode.LOAD:
            # No forward, no capture — graphs are restored after the loop by
            # the runner-level patch below. Register the key order so the
            # post-pass can validate coverage.
            _v3_load_keys.append(shape_key)
            return

        # SAVE: skip the 2 warmup forwards; capture directly with foundry.
        from foundry.integration.sglang.graph_ops import (
            capture_graph,
            create_device_graph,
            save_graph,
        )

        self._device_module.synchronize()
        self._tp_group.barrier()
        graph = create_device_graph()
        out = capture_graph(graph, self._pool, self._capture_stream, forward_fn)
        save_graph(graph, out, shape_key.size)
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    _v3_load_keys: list = []

    @functools.wraps(orig_capture)
    def patched_capture(self):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig_capture(self)

        if mode == CUDAGraphExtensionMode.LOAD:
            rt.log_alloc_offset("before_preallocate")
            rt.preallocate_for_load_mode()
            rt.log_alloc_offset("after_preallocate")

        _v3_load_keys.clear()
        result = orig_capture(self)

        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import (
                pack_fatbins,
                save_graph_manifest,
            )

            save_graph_manifest()
            pack_fatbins()
            rt.capture_final_alloc_offset()
            return result

        # LOAD: restore every archived graph and slot into the backend maps.
        from foundry.integration.sglang.graph_ops import load_all_graphs_v3

        backend = self.backend
        load_all_graphs_v3(backend, list(_v3_load_keys))
        rt.log_alloc_offset("after_load_all_graphs")
        return result

    backend_cls.capture_one = patched_capture_one
    runner_cls.capture = patched_capture


def _runner_parallel_ranks(model_runner):
    """Rank triple across engine eras: v3 forks keep ranks on a ParallelState
    object (model_runner.ps); older trees expose them directly."""
    ps = getattr(model_runner, "ps", None)
    if ps is not None:
        return ps.tp_rank, getattr(ps, "pp_rank", 0), getattr(ps, "dp_rank", None)
    return (
        model_runner.tp_rank,
        model_runner.pp_rank,
        _resolve_dp_rank(model_runner),
    )


def _patch_init_torch_distributed() -> None:
    from sglang.srt.model_executor import model_runner as mr

    cls = mr.ModelRunner
    orig = cls.init_torch_distributed

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self, *args, **kwargs)

        # Bind this rank's CUDA device BEFORE reserving the VMM region.
        # init_torch_distributed (orig) calls set_device(self.gpu_id)
        # internally, but foundry's set_allocation_region (inside
        # setup_graph_extension) reserves the region on the *current* device.
        # For DP rank > 0 the current device is still cuda:0 at this point, so
        # without setting it first the region lands on the wrong GPU and the
        # rank's later allocations fault with an async illegal memory access
        # (surfacing at the first Stream()/kernel). Mirrors model_runner's own
        # set_device(self.gpu_id). Single-GPU is unaffected (gpu_id == 0).
        if self.device == "cuda":
            import torch

            torch.get_device_module(self.device).set_device(self.gpu_id)

        tp_rank, pp_rank, dp_rank = _runner_parallel_ranks(self)
        rt.setup_graph_extension(
            self.server_args,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
        )
        rt.log_alloc_offset("after_setup_graph_ext")
        # NCCL buffers stay inside the VMM region (deterministic offsets);
        # the hook refuses legacy IPC export for region memory, steering
        # NCCL onto its SHM transport so no peer pointers end up inside
        # captured graphs. See cuIpcGetMemHandle in hook.cpp.
        result = orig(self, *args, **kwargs)
        rt.log_alloc_offset("after_init_torch_dist")
        rt.skip_to_scratch_boundary()
        rt.log_alloc_offset("after_scratch_skip")
        return result

    cls.init_torch_distributed = patched


def _patch_init_memory_pool() -> None:
    from sglang.srt.model_executor import model_runner_kv_cache_mixin as kv_mixin
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

    cls = kv_mixin.ModelRunnerKVCacheMixin
    orig = cls.init_memory_pool

    @functools.wraps(orig)
    def patched(self, pre_model_load_memory):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self, pre_model_load_memory)

        # Weight loading (the multi-threaded phase the free-side device sync
        # protects) is complete by the time the memory pool is resolved.
        rt.disable_sync_on_free()

        if mode == CUDAGraphExtensionMode.LOAD:
            import torch

            rt.log_alloc_offset("before_init_memory_pool")
            state = rt.load_warmup_state()
            if not state.memory_pool_config:
                raise RuntimeError("Foundry LOAD requires memory_pool_config")
            self.memory_pool_config = MemoryPoolConfig(**state.memory_pool_config)
            # Hybrid linear-attention models: restore SAVE-resolved
            # server_args values (max_mamba_cache_size etc.) that the
            # skipped init would otherwise have computed.
            rt.apply_server_args_overrides(self.server_args, state.server_args_overrides)
            # Mirror SAVE's ``_resolve_memory_pool_config`` ->
            # ``get_available_gpu_memory(empty_cache=True)`` side
            # effect. Without this, torch's caching allocator retains
            # segments that SAVE released — causing the
            # attention-backend init below to take a different
            # cuMemAlloc path and drift the VMM cursor away from
            # SAVE's recorded ``start_base_addr``.
            torch.cuda.empty_cache()
            self._apply_memory_pool_config(self.memory_pool_config)
            rt.log_alloc_offset("after_init_memory_pool")
            logger.info("[Foundry] SGLang reused saved memory pool config")
            return None

        rt.log_alloc_offset("before_init_memory_pool")
        result = orig(self, pre_model_load_memory)
        rt.log_alloc_offset("after_init_memory_pool")
        state = rt.create_warmup_state(
            asdict(self.memory_pool_config),
            rt.collect_server_args_overrides(self.server_args),
            rt.compute_archive_fingerprint(self.server_args),
        )
        rt.save_warmup_state(state)
        return result

    cls.init_memory_pool = patched


def _patch_load_model() -> None:
    from sglang.srt.model_executor import model_runner as mr

    cls = mr.ModelRunner
    orig = cls.load_model

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        return orig(self, *args, **kwargs)

    cls.load_model = patched


def _patch_kernel_warmup() -> None:
    from sglang.srt.model_executor import model_runner as mr

    cls = mr.ModelRunner
    orig = cls.kernel_warmup

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self, *args, **kwargs)
        # Phase 1 keeps pre-graph model-forward warmups out of SAVE/LOAD.
        logger.info("[Foundry] SGLang kernel_warmup skipped in %s mode", mode.value)
        return None

    cls.kernel_warmup = patched


def _patch_cuda_graph_capture() -> None:
    from sglang.srt.model_executor import cuda_graph_runner as cgr

    cls = cgr.CudaGraphRunner
    orig_capture = cls.capture
    orig_create_device_graph = cls._create_device_graph
    orig_capture_graph = cls._capture_graph
    orig_capture_one_batch_size = cls.capture_one_batch_size

    # When set, the capture machinery is being reused as a foundry-driven WARMUP
    # pass: run a real forward per bs (no graph capture, no save) to trigger all
    # of sglang's pre-capture lazy init — DeepEP buffer, DeepGEMM per-shape JIT,
    # etc. — that would otherwise fire inside the captured stream and abort with
    # "operation not permitted when stream is capturing". See `_run_warmup_pass`.
    warmup_active = [False]

    @functools.wraps(orig_create_device_graph)
    def patched_create_device_graph(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if warmup_active[0]:
            # Throwaway graph object; the warmup never captures into it.
            return orig_create_device_graph(self, *args, **kwargs)
        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import create_device_graph

            return create_device_graph()
        return orig_create_device_graph(self, *args, **kwargs)

    @functools.wraps(orig_capture_graph)
    def patched_capture_graph(self, graph, pool, stream, run_once_fn):
        mode = get_graph_extension_mode()
        if warmup_active[0]:
            # Warmup: run the forward eagerly (NO torch.cuda.graph), so DeepGEMM
            # JIT compile / NVSHMEM buffer creation happen outside stream capture.
            return run_once_fn()
        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import capture_graph

            return capture_graph(graph, pool, stream, run_once_fn)
        return orig_capture_graph(self, graph, pool, stream, run_once_fn)

    @functools.wraps(orig_capture_one_batch_size)
    def patched_capture_one_batch_size(self, bs, forward, stream_idx=None):
        mode = get_graph_extension_mode()
        if warmup_active[0]:
            # Warmup pass: run upstream unchanged (real warmup forwards + a
            # neutered "capture" that is just another real forward). No
            # suppression, no save_graph.
            return orig_capture_one_batch_size(self, bs, forward, stream_idx)
        if mode == CUDAGraphExtensionMode.SAVE:
            # Suppress the two pre-capture warmup forwards
            # (cuda_graph_runner.py: ``for _ in range(2): run_once()``).
            # Their non-deterministic activation allocations would pollute
            # the torch caching allocator with freed segments that LOAD
            # cannot reproduce — causing the per-bs init's cache-miss vs
            # cache-hit asymmetry that drifts the VMM cursor away from
            # each saved ``start_base_addr``. JIT / autotune still happens
            # inside the graph capture (3rd run_once invocation) and is
            # recorded as alloc events, mirroring vLLM doc 04 §2.
            counter = [0]
            real_forward = forward

            def warmup_skipping_forward(*args, **kwargs):
                counter[0] += 1
                if counter[0] <= 2:
                    return None
                return real_forward(*args, **kwargs)

            forward = warmup_skipping_forward
        graph, output = orig_capture_one_batch_size(self, bs, forward, stream_idx)
        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import save_graph

            # Mirror the inline key shape upstream uses for self.graphs[key]
            # in `_capture_one_stream`. ``_make_graph_key`` and
            # ``get_capture_lora_variant`` were removed in sglang
            # commit ce2506e1c (record_nolora_graph deprecation).
            key = bs if stream_idx is None else f"{stream_idx}_{bs}"
            save_graph(graph, output, key)
        return graph, output

    def _run_warmup_pass(self):
        """Foundry-driven pre-capture warmup for the DeepEP/EP path.

        Reuses the upstream capture loop with graph capture neutered (see
        ``warmup_active``) to run one real forward per ``capture_bs`` BEFORE the
        real capture/replay work — triggering every pre-capture lazy init sglang
        normally does in its (foundry-suppressed) warmup forwards: DeepEP buffer
        creation, DeepGEMM per-shape JIT compile, etc. Run on BOTH SAVE and LOAD
        so the VMM cursor trajectory stays symmetric. ``self.graphs`` /
        ``self.output_buffers`` are saved/cleared/restored so the throwaway
        warmup entries don't leak into the real pass.
        """
        warmup_active[0] = True
        saved_graphs, saved_buffers = self.graphs, self.output_buffers
        self.graphs, self.output_buffers = {}, {}
        t0 = time.perf_counter()
        try:
            orig_capture(self)
        finally:
            warmup_active[0] = False
            self.graphs, self.output_buffers = saved_graphs, saved_buffers
        logger.info(
            "[Foundry] SGLang EP warmup pass (lazy-init) completed in %.3fs",
            time.perf_counter() - t0,
        )

    @functools.wraps(orig_capture)
    def patched(self, *args, **kwargs):
        mode = get_graph_extension_mode()

        # DeepEP/EP only (gated so the validated dense / single-GPU / DP paths
        # are untouched):
        #   SAVE: sglang triggers pre-capture lazy init (DeepGEMM per-shape JIT,
        #     DeepEP buffer, ...) via warmup forwards that foundry suppresses;
        #     left lazy they fire inside the captured stream and abort
        #     ("operation not permitted when stream is capturing"). Run a real
        #     forward per bs up front so all that happens outside capture.
        #   LOAD: no capture happens — preallocate_for_load_mode reserves the
        #     whole region up to final_alloc_offset and replay places each alloc
        #     at its recorded absolute offset, so LOAD need NOT replay the warmup
        #     cursor trajectory. Running the warmup here is not just unnecessary
        #     but harmful: its orig_capture re-enters graph_capture() and leaves
        #     the context in a state that breaks the threaded finish_graph_loads
        #     ("invalid device context"). So warmup is SAVE-only.
        # bootstrap_deepep_buffer runs on both (cheap singleton) to guarantee the
        # NVSHMEM runtime is up before replay on LOAD / before capture on SAVE.
        if _pre_capture_warmup_needed(self):
            if mode == CUDAGraphExtensionMode.SAVE:
                _run_warmup_pass(self)
            elif mode == CUDAGraphExtensionMode.LOAD and (
                os.environ.get("FOUNDRY_LOAD_WARMUP", "1") == "1"
            ):
                if _ep_lazy_init_needed():
                    # EP validated with single-threaded graph builds only;
                    # threaded finish_graph_loads after a warmup pass hit
                    # "invalid device context" historically.
                    os.environ.setdefault("FOUNDRY_GRAPH_LOAD_THREADS", "1")
                # TP: the captured allreduce kernels reference NCCL P2P/IPC
                # transport buffers that SAVE's warmup pass established
                # lazily (first collective) BEFORE capture. Skipping warmup
                # on LOAD leaves that transport unestablished — the graph
                # addresses are mapped by preallocate, so replay silently
                # reads meaningless memory: prefill (eager) is correct,
                # decode emits garbage. Re-run the same warmup pass at the
                # same lifecycle point so pynccl/NCCL connects with the
                # same deterministic allocation trajectory as SAVE (warmup
                # activations precede the pre-pass wrappers in cursor
                # order, so a partial warmup would desync the cursor).
                # EP: the same symmetric warmup also fixes a DP-rank-parity
                # prefill corruption (one DP rank emitted nondeterministic
                # garbage on the FIRST decode token — i.e. the prefill
                # logits — while decode-graph replay was fine). Validated
                # 12/12 identical to SAVE with dlogprob=0 under
                # single-threaded graph builds (see below).
                _run_warmup_pass(self)
            # DeepEP buffer bootstrap stays EP-only (no-op / crash for dense TP).
            if _ep_lazy_init_needed() and mode != CUDAGraphExtensionMode.NONE:
                from foundry.integration.sglang.graph_ops import (
                    bootstrap_deepep_buffer,
                )

                bootstrap_deepep_buffer(self)

        if mode == CUDAGraphExtensionMode.LOAD:
            from sglang.srt.distributed.device_communicators.pynccl_allocator import (
                set_graph_pool_id,
            )

            from foundry.integration.sglang.graph_ops import (
                initialize_all_attention_metadata,
                load_all_graphs,
            )

            state = rt.get_state()
            if state is None:
                raise RuntimeError("Foundry SGLang state is not initialized")
            # Set up the graph memory pool once — sglang shares one pool
            # across all captured graphs, and runtime replay also requires
            # it to be set so pynccl knows which pool the graph belongs to.
            if cgr.get_global_graph_memory_pool() is None:
                cgr.set_global_graph_memory_pool(self.device_module.graph_pool_handle())
            set_graph_pool_id(cgr.get_global_graph_memory_pool())
            rt.log_alloc_offset("before_preallocate")
            rt.preallocate_for_load_mode()
            rt.log_alloc_offset("after_preallocate")
            # Pre-pass: allocate every per-bs FlashInfer wrapper in
            # ``reversed(capture_bs)`` order, matching the order SAVE used.
            # SAVE's patched ``init_forward_metadata_capture_cuda_graph``
            # is idempotent so the inner per-iter call inside the SAVE
            # capture loop does not re-allocate. Same upfront allocation
            # sequence on both sides → cursor sits at SAVE's
            # ``start_base_addr_0`` when graph load begins.
            # FlashInfer-only pre-pass (see SAVE branch). fa3 etc. allocate their
            # cuda-graph metadata once in init_cuda_graph_state, so skip it.
            if hasattr(self.attn_backend, "indices_updater_decode"):
                initialize_all_attention_metadata(self)
            rt.log_alloc_offset("after_pre_init")
            # Single ``start_graph_builds(all_paths)`` call so templates
            # and on-demand graphs link via ``shared_exec`` in the
            # manifest. ``finish_graph_loads`` replays alloc events
            # graph-by-graph in the same order SAVE captured them.
            load_all_graphs(self)
            rt.log_alloc_offset("after_load_all_graphs")
            self.graphs = {k: v[0] for k, v in state.loaded_graphs.items()}
            self.output_buffers = {k: v[1] for k, v in state.loaded_graphs.items()}
            # Non-FlashInfer backends (e.g. fa3) populate per-bs decode metadata
            # — looked up at replay as attn_backend.decode_cuda_graph_metadata[bs]
            # — inside the capture loop, which LOAD replaces. The FlashInfer
            # pre-pass above (gated) handled flashinfer; for the rest, populate it
            # now. Run AFTER load_all_graphs so the (already-correct) loaded-graph
            # VMM offsets are unaffected — fa3's metadata are lightweight views
            # over the fixed init_cuda_graph_state workspace, not graph memory.
            if not hasattr(self.attn_backend, "indices_updater_decode"):
                initialize_all_attention_metadata(self)
            # Initialize the DeepEP cuda-graph adapter's captured mode. Upstream
            # sets this in deepep_adapter.capture() during the capture loop,
            # which LOAD replaces — so without this, runtime replay() asserts
            # `_captured_deepep_mode is not None` on the first decode. capture()
            # self-gates on the DeepEP backend (no-op otherwise) and sets the
            # decode dispatch mode the captured graphs expect.
            self.deepep_adapter.capture(is_extend_in_batch=False)
            return None

        if mode == CUDAGraphExtensionMode.SAVE:
            # FlashInfer allocates a per-bs metadata wrapper (each with its own
            # _int_workspace_buffer) on every capture init, so foundry pre-allocates
            # them up front and installs a reuse shim that makes the inner init
            # reuse them — keeping the VMM cursor deterministic vs LOAD. Backends
            # with a single fixed cuda-graph metadata workspace allocated once in
            # init_cuda_graph_state (e.g. fa3 / FlashAttentionBackend) don't need
            # this and the plain capture is already SAVE/LOAD-deterministic. Detect
            # FlashInfer by its per-bs indices_updater_decode.
            attn_backend = self.attn_backend
            use_fi_prepass = hasattr(attn_backend, "indices_updater_decode")
            # Post-#26735 sglang removed the legacy per-bs capture API in favor
            # of ``init_forward_metadata_out_graph(fb, in_capture)`` which
            # delegates allocation to ``_prepare_cuda_graph_metadata``. Support
            # both eras: on the new ABC the reuse shim wraps _prepare_ instead.
            legacy_capture_api = hasattr(
                attn_backend, "init_forward_metadata_capture_cuda_graph"
            )
            real_init = (
                attn_backend.init_forward_metadata_capture_cuda_graph
                if legacy_capture_api
                else None
            )
            real_prepare = None

            def reuse_pre_pass_init(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            ):
                # The pre-pass already allocated a wrapper for this
                # bs and stored it in
                # ``decode_cuda_graph_metadata`` /
                # ``prefill_cuda_graph_metadata``. Reuse it directly
                # — no second torch.empty for ``_int_workspace_buffer``.
                # Re-run the planner with the same buffer slices the
                # capture forward uses, then point
                # ``forward_metadata`` at the same wrappers. Same
                # plan call on LOAD via the symmetric pre-pass, so
                # the captured graph kernels reference VMM addresses
                # that LOAD's wrappers actually occupy.
                from sglang.srt.layers.attention.flashinfer_backend import (
                    DecodeMetadata,
                    PrefillMetadata,
                )

                if forward_mode.is_decode_or_idle():
                    wrappers = attn_backend.decode_cuda_graph_metadata.get(bs)
                    if wrappers is None:
                        return real_init(
                            bs,
                            num_tokens,
                            req_pool_indices,
                            seq_lens,
                            encoder_lens,
                            forward_mode,
                            spec_info,
                        )
                    seq_lens_sum = seq_lens.sum().item()
                    attn_backend.indices_updater_decode.update(
                        req_pool_indices,
                        seq_lens,
                        seq_lens.cpu(),
                        seq_lens_sum,
                        decode_wrappers=wrappers,
                        encoder_lens=encoder_lens,
                        spec_info=spec_info,
                        fixed_split_size=None,
                        disable_split_kv=attn_backend.disable_cuda_graph_kv_split,
                    )
                    attn_backend.forward_metadata = DecodeMetadata(wrappers)
                    return
                if (
                    forward_mode.is_target_verify()
                    or forward_mode.is_draft_extend()
                    or forward_mode.is_dllm_extend()
                ):
                    wrappers = attn_backend.prefill_cuda_graph_metadata.get(bs)
                    if wrappers is None:
                        return real_init(
                            bs,
                            num_tokens,
                            req_pool_indices,
                            seq_lens,
                            encoder_lens,
                            forward_mode,
                            spec_info,
                        )
                    seq_lens_sum = seq_lens.sum().item()
                    use_ragged = forward_mode.is_dllm_extend()
                    prefix_lens = (
                        seq_lens - attn_backend.dllm_config.block_size
                        if forward_mode.is_dllm_extend()
                        else None
                    )
                    spec_info_arg = None if forward_mode.is_dllm_extend() else spec_info
                    attn_backend.indices_updater_prefill.update(
                        req_pool_indices,
                        seq_lens,
                        seq_lens.cpu(),
                        seq_lens_sum,
                        prefix_lens=prefix_lens,
                        prefill_wrappers=wrappers,
                        use_ragged=use_ragged,
                        encoder_lens=encoder_lens,
                        spec_info=spec_info_arg,
                    )
                    attn_backend.forward_metadata = PrefillMetadata(wrappers, use_ragged, False)
                    return
                # Unknown mode — fall back to real init.
                return real_init(
                    bs,
                    num_tokens,
                    req_pool_indices,
                    seq_lens,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                )

            if use_fi_prepass:
                from foundry.integration.sglang.graph_ops import (
                    initialize_all_attention_metadata,
                )

                rt.log_alloc_offset("save_before_pre_init")
                # Pre-pass: allocate every per-bs FlashInfer wrapper up front, in
                # the same ``reversed(capture_bs)`` order LOAD uses.
                initialize_all_attention_metadata(self)
                rt.log_alloc_offset("save_after_pre_init")
                # Drop the pre-pass's last forward_metadata ref so popping the dict
                # entry doesn't keep the wrapper alive at refcount 1.
                attn_backend.forward_metadata = None
                if legacy_capture_api:
                    attn_backend.init_forward_metadata_capture_cuda_graph = (
                        reuse_pre_pass_init
                    )
                else:
                    # New ABC (post-#26735): out_graph(in_capture=True) calls
                    # _prepare_cuda_graph_metadata (allocation) then runs the
                    # indices updater with decode_cuda_graph_metadata[bs].
                    # Reuse shim: skip the allocation when the pre-pass already
                    # populated the dict; mirror _prepare's forward_metadata
                    # assignment; fall through to real _prepare otherwise.
                    real_prepare = attn_backend._prepare_cuda_graph_metadata

                    def reuse_prepare(bs, num_tokens, forward_mode, spec_info):
                        from sglang.srt.layers.attention.flashinfer_backend import (
                            DecodeMetadata,
                            PrefillMetadata,
                        )

                        if forward_mode.is_decode_or_idle():
                            wrappers = attn_backend.decode_cuda_graph_metadata.get(bs)
                            if wrappers is not None:
                                attn_backend.forward_metadata = DecodeMetadata(wrappers)
                                return
                        elif (
                            forward_mode.is_target_verify()
                            or forward_mode.is_draft_extend()
                            or forward_mode.is_dllm_extend()
                        ):
                            wrappers = attn_backend.prefill_cuda_graph_metadata.get(bs)
                            if wrappers is not None:
                                attn_backend.forward_metadata = PrefillMetadata(
                                    wrappers, forward_mode.is_dllm_extend(), False
                                )
                                return
                        return real_prepare(bs, num_tokens, forward_mode, spec_info)

                    attn_backend._prepare_cuda_graph_metadata = reuse_prepare
            try:
                result = orig_capture(self, *args, **kwargs)
            finally:
                if use_fi_prepass:
                    if legacy_capture_api:
                        attn_backend.init_forward_metadata_capture_cuda_graph = real_init
                    elif real_prepare is not None:
                        attn_backend._prepare_cuda_graph_metadata = real_prepare

            from foundry.integration.sglang.graph_ops import (
                pack_fatbins,
                save_graph_manifest,
            )

            save_graph_manifest()
            pack_fatbins()
            rt.capture_final_alloc_offset()
            return result

        return orig_capture(self, *args, **kwargs)

    cls._create_device_graph = patched_create_device_graph
    cls._capture_graph = patched_capture_graph
    cls.capture_one_batch_size = patched_capture_one_batch_size
    cls.capture = patched


def _patch_spawn_sites() -> None:
    def _prepare_env(owner) -> None:
        """Resolve AUTO and export LD_PRELOAD/FOUNDRY_MODE for the children.

        server_args is fully resolved by spawn time, and the hook library
        reads FOUNDRY_MODE in its constructor — before any Python runs in the
        child — so the mode has to be concrete before the fork.
        """
        if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
            return
        rt.resolve_auto_mode(getattr(owner, "server_args", None) or _current_server_args())
        rt.setup_ld_preload_env()

    try:
        from sglang.srt.entrypoints import engine as engine_mod
    except Exception:
        engine_mod = None

    if engine_mod is not None:
        orig_launch = engine_mod.Engine._launch_scheduler_processes

        @functools.wraps(orig_launch)
        def patched_launch(self, *args, **kwargs):
            _prepare_env(self)
            return orig_launch(self, *args, **kwargs)

        engine_mod.Engine._launch_scheduler_processes = patched_launch

    try:
        from sglang.srt.managers import data_parallel_controller as dpc
    except Exception:
        dpc = None

    if dpc is not None:
        orig_start = dpc.DataParallelController.launch_tensor_parallel_group

        @functools.wraps(orig_start)
        def patched_start(self, *args, **kwargs):
            _prepare_env(self)
            return orig_start(self, *args, **kwargs)

        dpc.DataParallelController.launch_tensor_parallel_group = patched_start
