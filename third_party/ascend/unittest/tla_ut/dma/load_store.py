"""GM/UB loads with immutable configuration and ordinary stores."""
import sys
from pathlib import Path

# Do not let this category's copy.py shadow Python's standard library.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (compile_offline, lower_offline, get_stages, check_kernel_binary, matrix)

CONFIGS = [dict(name="plain_" + dtype, dtype=dtype) for dtype in ("fp16", "fp32", "i32")]
CONFIGS += [
    dict(name="l2_cache_normal_fv", dtype="fp32", l2_cache_ctrl=tla.L2CachePolicy.NORMAL_FV, hivm_policy="EvictFirst"),
    dict(name="l2_cache_normal_lv", dtype="fp32", l2_cache_ctrl=tla.L2CachePolicy.NORMAL_LV, hivm_policy="EvictLast"),
    dict(name="l2_cache_raw_load", dtype="fp32", raw_l2_cache_mode=4),
]
DTYPES = {"fp16": (tl.float16, torch.float16), "fp32": (tl.float32, torch.float32), "i32": (tl.int32, torch.int32)}


@tla.jit
def transfer(X, Y, DTYPE: tl.constexpr, L2_CACHE_CTRL: tl.constexpr, RAW_L2_CACHE_MODE: tl.constexpr):
    with tla.scope.vector():
        src = tla.view(X, (2, 32))
        out = tla.view(Y, (2, 32))
        ub = tla.alloc((2, 32), DTYPE)
        # Construct in a JIT function, including constexpr fields.
        p: tl.constexpr = tla.LoadGM2UBParams(l2_cache_ctrl=L2_CACHE_CTRL, raw_l2_cache_mode=RAW_L2_CACHE_MODE)
        tla.load_gm_to_ub(src, ub, p)
        tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.store_ub_to_gm(ub, out)


def constants(case):
    return dict(DTYPE=DTYPES[case["dtype"]][0], L2_CACHE_CTRL=case.get("l2_cache_ctrl"),
                RAW_L2_CACHE_MODE=case.get("raw_l2_cache_mode"))


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    sig = {"X": "*" + case["dtype"], "Y": "*" + case["dtype"]}
    mod, ctx = compile_offline(transfer, sig, constants(case), output_dir=output_dir)
    text = str(lower_offline(mod, ctx, output_dir=output_dir))
    assert text.count("hivm.hir.load ") == 1 and text.count("hivm.hir.store ") == 1
    assert "#hivm.tcore_type<VECTOR>" in text
    assert 'hacc.mode = "tla"' in text
    # Pin existing public members and their meanings independently of enum values.
    mode = case.get("raw_l2_cache_mode")
    if mode is None:
        policy = case.get("hivm_policy", "EvictFirst")
        assert f"eviction_policy = <{policy}>" in text
    else:
        assert f"hivm.l2_cache_mode = {mode} : i32" in text
        assert "eviction_policy" not in text
    print("IR PASS:", case["name"])


def launch(case, compile_only=False):
    dtype = DTYPES[case["dtype"]][1]
    x_cpu = (torch.arange(64).reshape(2, 32) - 31).to(dtype)
    y = torch.full((2, 32), 2, dtype=dtype, device="npu")
    x = x_cpu.npu()
    if compile_only:
        return transfer.warmup(x, y, grid=(1, ), **constants(case))
    transfer[(1, )](x, y, **constants(case))
    torch.npu.synchronize()
    return y.cpu(), x_cpu


@matrix(CONFIGS)
def test_compile(case):
    check_kernel_binary(launch(case, True))


@matrix(CONFIGS)
def test_run(case):
    actual, expected = launch(case)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("ACCURACY PASS:", case["name"])


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)


if __name__ == "__main__":
    main()
