"""GM ND -> L1 NZ -> UB ND -> GM, including L2 cache configuration."""
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

CONFIGS = [dict(name="cache_" + str(cache), cache=cache) for cache in (None, 0, 1, 2, 3)]


@tla.jit
def roundtrip(X, Y, P: tl.constexpr):
    out = tla.alloc((16, 32), tl.float16)
    with tla.scope.cube():
        l1 = tla.alloc((2, 1, 16, 16), tl.float16, space="l1")
        tla.load_gm_to_l1(tla.view(X, (16, 32)), l1, P)
        tla.copy_l1_to_ub(l1, out, tla.CopyL12UBParams())
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE1, 0)
    with tla.scope.vector():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
        # L12UB writes the first AIV's UB; the second AIV must not overwrite GM.
        if tla.sub_block_id() == 0:
            tla.store_ub_to_gm(out, tla.view(Y, (16, 32)), tla.StoreUB2GMParams())


def params(case):
    return tla.LoadGM2L1Params(l2_cache_ctrl=case["cache"])


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    mod, ctx = compile_offline(roundtrip, {"X": "*fp16", "Y": "*fp16"}, {"P": params(case)}, output_dir=output_dir)
    text = str(lower_offline(mod, ctx, output_dir=output_dir))
    for op in ("nd2nz", "l12ub", "store"):
        assert text.count("hivm.hir." + op + " ") == 1
    assert "dst_continuous" in text
    assert text.count("hivm.hir.sync_block_set") == 1
    assert text.count("hivm.hir.sync_block_wait") == 1
    assert "hivm.hir.get_sub_block_idx" in text
    if case["cache"] is not None:
        assert "l2_cache_mode = " + str(case["cache"]) + " : i32" in text
    else:
        assert "l2_cache_mode" not in text
    print("IR PASS:", case["name"])


def launch(case, compile_only=False):
    expected = ((torch.arange(512).reshape(16, 32) * 7) % 43 - 21).half()
    x, y = expected.npu(), torch.full_like(expected, float("nan"), device="npu")
    if compile_only:
        return roundtrip.warmup(x, y, P=params(case), grid=(1, ))
    roundtrip[(1, )](x, y, P=params(case))
    torch.npu.synchronize()
    return y.cpu(), expected


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
