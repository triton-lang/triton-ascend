"""GM ND -> L1 NZ -> UB ND -> GM with default and explicit cache modes."""
import sys
from pathlib import Path

# Do not let this category's copy.py shadow Python's standard library.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (compile_offline, lower_offline, get_stages, check_kernel_binary, matrix)

CONFIGS = [
    dict(name="nd2nz_default", params=None),
    dict(name="nd2nz_default_params", params=tla.LoadGM2L1Params()),
    dict(name="nd2nz_explicit_mode", params=tla.LoadGM2L1Params(mode=tla.LoadGM2L1Mode.ENABLE_ND2NZ)),
    dict(name="cache_none", params=tla.LoadGM2L1Params(l2_cache_ctrl=None)),
    *(dict(name=f"cache_{mode}", params=tla.LoadGM2L1Params(l2_cache_ctrl=mode)) for mode in range(5)),
]


@tla.jit
def roundtrip(X, Y, P: tl.constexpr):
    out = tla.alloc((16, 32), tl.float16)
    with tla.scope.cube():
        l1 = tla.alloc((2, 1, 16, 16), tl.float16, space="l1")
        tla.load_gm_to_l1(tla.view(X, (16, 32)), l1, P)
        tla.copy_l1_to_ub(l1, out)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE1, 0)
    with tla.scope.vector():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
        # L12UB writes the first AIV's UB; the second AIV must not overwrite GM.
        if tla.sub_block_id() == 0:
            tla.store_ub_to_gm(out, tla.view(Y, (16, 32)))


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    mod, ctx = compile_offline(roundtrip, {"X": "*fp16", "Y": "*fp16"}, {"P": case["params"]}, output_dir=output_dir)
    text = str(lower_offline(mod, ctx, output_dir=output_dir))
    for op in ("nd2nz", "l12ub", "store"):
        assert text.count("hivm.hir." + op + " ") == 1
    assert "dst_continuous" in text
    nd2nz = next(line for line in text.splitlines() if "hivm.hir.nd2nz " in line)
    assert "enable_nd2nz" not in nd2nz and " mode =" not in nd2nz
    assert text.count("hivm.hir.sync_block_set") == 1
    assert text.count("hivm.hir.sync_block_wait") == 1
    assert "hivm.hir.get_sub_block_idx" in text
    cache = case["params"].l2_cache_ctrl if case["params"] is not None else None
    if cache is None:
        assert "l2_cache_mode" not in text
    else:
        assert f"l2_cache_mode = {cache} : i32" in text
    print("IR PASS:", case["name"])


@pytest.mark.parametrize("kwargs,message", [
    ({"mode": "enable_nd2nz"}, "mode must be a compile-time LoadGM2L1Mode"),
    ({"mode": True}, "mode must be a compile-time LoadGM2L1Mode"),
    ({"mode": None}, "mode must be a compile-time LoadGM2L1Mode"),
    ({"l2_cache_ctrl": True}, "l2_cache_ctrl must be a compile-time integer or None"),
    ({"l2_cache_ctrl": 1.0}, "l2_cache_ctrl must be a compile-time integer or None"),
    ({"l2_cache_ctrl": "4"}, "l2_cache_ctrl must be a compile-time integer or None"),
    ({"l2_cache_ctrl": -1}, "l2_cache_ctrl must be between 0 and 4"),
    ({"l2_cache_ctrl": 5}, "l2_cache_ctrl must be between 0 and 4"),
])
def test_invalid_params(kwargs, message):
    params = tla.LoadGM2L1Params(**kwargs)
    with pytest.raises(Exception, match=message):
        compile_offline(roundtrip, {"X": "*fp16", "Y": "*fp16"}, {"P": params})


def launch(case, compile_only=False):
    expected = ((torch.arange(512).reshape(16, 32) * 7) % 43 - 21).half()
    x, y = expected.npu(), torch.full_like(expected, float("nan"), device="npu")
    if compile_only:
        return roundtrip.warmup(x, y, P=case["params"], grid=(1, ))
    roundtrip[(1, )](x, y, P=case["params"])
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
