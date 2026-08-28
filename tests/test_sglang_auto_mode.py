"""Exercise the real _auto_decision branches with torch/sglang stubbed out.

Only the GPU-facing bits are faked; the decision code under test is the
shipped one.
"""
import json, sys, types, tempfile, os
from pathlib import Path

# --- stub torch (runtime.py imports it at module level) ---
torch = types.ModuleType("torch")
class _Props:
    name = "NVIDIA H200"; major = 9; minor = 0; total_memory = 1 << 37
torch.cuda = types.SimpleNamespace(
    get_device_properties=lambda i: _Props(), device_count=lambda: 2,
    empty_cache=lambda: None)
torch.version = types.SimpleNamespace(cuda="13.0")
torch.__version__ = "2.13.0"
torch.Tensor = type("Tensor", (), {})
torch.device = type("device", (), {})
torch.dtype = type("dtype", (), {})
sys.modules["torch"] = torch
# foundry.ops is a C extension; stub the symbols the package imports at load
ops = types.ModuleType("foundry.ops")
for _n in ("free_preallocated_region", "preallocate_region", "set_allocation_region",
           "get_current_alloc_offset", "set_current_alloc_offset", "stop_allocation_region",
           "resume_allocation_region", "set_skip_fatbin_processing",
           "load_cuda_modules_and_libraries", "set_sync_on_free"):
    setattr(ops, _n, lambda *a, **k: None)
ops.CUDAGraph = type("CUDAGraph", (), {})
sys.modules["foundry.ops"] = ops
sysver = types.ModuleType("sglang.version"); sysver.__version__ = "0.5.17"
sys.modules["sglang"] = types.ModuleType("sglang"); sys.modules["sglang.version"] = sysver

sys.path.insert(0, "/Users/namsangdae/research-foundry/repo/python")
from foundry.integration.sglang import config as cfgmod, runtime as rt
from foundry.integration.sglang.config import CUDAGraphExtensionMode as M

class SA:  # minimal server_args
    tp_size = 2; pp_size = 1; dp_size = 1; ep_size = 2
    enable_dp_attention = False
    model_path = "/models/Solar-Pro-4-W4AFP8"; quantization = None
    kv_cache_dtype = "fp8_e4m3"; attention_backend = "fa3"
    speculative_algorithm = None
    cuda_graph_config = "decode(bs=[1,2,4,8,12,16,24])"

def setup(root):
    cfgmod._config = cfgmod.CUDAGraphExtensionConfig(
        mode=M.AUTO, base_addr=0x600000000000, region_size="256GB",
        workspace_root=str(root), scratch_space_size="4096MB")

def bake(root, sa=SA, complete=True, fingerprint=True, world=2):
    """Write an archive the way SAVE would."""
    root.mkdir(parents=True, exist_ok=True)
    st = rt.create_warmup_state({"a": 1}, {},
                                rt.compute_archive_fingerprint(sa) if fingerprint else {})
    rt.save_warmup_state(st)
    for i in range(world):
        d = root / f"rank_{i}"; d.mkdir(exist_ok=True)
        if complete or i == 0:
            (d / "final_alloc_offset.json").write_text('{"final_alloc_offset": 146255380480}')

def decide(sa=SA):
    cfgmod._config.mode = M.AUTO
    return rt.resolve_auto_mode(sa), rt._auto_decision(sa)[1]

results = []
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "archive"; setup(root)

    # 1. 아카이브 없음 → SAVE
    results.append(("아카이브 없음", *decide()))

    # 2. 정상 아카이브 → LOAD
    bake(root); results.append(("완결·일치", *decide()))

    # 3. rank 하나가 캡처 미완료 → SAVE
    (root / "rank_1" / "final_alloc_offset.json").unlink()
    results.append(("불완전(rank1 미완료)", *decide()))
    bake(root)  # restore

    # 4. 조건 변경 (tp 2→4) → SAVE
    class SA4(SA): tp_size = 4
    results.append(("구성 변경 tp2→tp4", *decide(SA4)))

    # 5. GPU 아키텍처 변경 (H200 sm90 → B200 sm100) → SAVE
    _Props.name = "NVIDIA B200"; _Props.major = 10; _Props.minor = 0
    results.append(("아키텍처 sm90→sm100", *decide()))
    _Props.name = "NVIDIA H200"; _Props.major = 9; _Props.minor = 0

    # 6. 지문 없는 구버전 아카이브 → SAVE
    bake(root, fingerprint=False); results.append(("구버전(지문없음)", *decide()))

    # 7. 손상된 아카이브 → SAVE
    (root / "warmup_state.json").write_text("{ broken")
    results.append(("손상된 warmup_state", *decide()))

expected = ["save", "load", "save", "save", "save", "save", "save"]
print(f"{'케이스':<24} {'결정':<6} 이유")
print("-" * 96)
ok = True
for (name, mode, reason), exp in zip(results, expected):
    got = mode.value
    mark = "OK " if got == exp else "FAIL"
    ok &= got == exp
    print(f"{name:<24} {got:<6} {reason[:64]}   [{mark}]")
print("-" * 96)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
