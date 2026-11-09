"""UB -> L1 -> UB; direct frontend contracts for L12L0/L12BT/FixBUF."""
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

CONFIGS = [dict(name="ub_l1_" + str(n), n=n) for n in (16, 32, 64)]


@tla.jit
def transfer(X, Y, N: tl.constexpr):
    l1 = tla.alloc((N // 16, 1, 16, 16), tl.float16, space="l1")
    out = tla.alloc((16, N), tl.float16)
    with tla.scope.vector():
        if tla.sub_block_id() == 0:
            src = tla.alloc((N // 16, 1, 16, 16), tl.float16)
            tla.load_gm_to_ub(tla.view(X, (N // 16, 1, 16, 16)), src)
            tla.copy_ub_to_l1(src, l1, tla.CopyUB2L1Params())
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)
    with tla.scope.cube():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
        tla.copy_l1_to_ub(l1, out, tla.CopyL12UBParams())
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE1, 1)
    with tla.scope.vector():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1)
        if tla.sub_block_id() == 0:
            tla.store_ub_to_gm(out, tla.view(Y, (16, N)))


@tla.jit
def route(src, dst, P: tl.constexpr, ROUTE: tl.constexpr):
    if ROUTE == "l0":
        with tla.scope.cube():
            tla.copy_l1_to_l0(src, dst, P)
    elif ROUTE == "bt":
        with tla.scope.cube():
            tla.copy_l1_to_bt(src, dst, P)
    elif ROUTE == "fb":
        tla.copy_l1_to_fb(src, dst, P)
    elif ROUTE == "l1":
        tla.load_gm_to_l1(src, dst, P)
    elif ROUTE == "ub_l1":
        tla.copy_ub_to_l1(src, dst, P)
    else:
        tla.copy(src, dst, P)


def route_ir(src_space, dst_space, p, name, dtype=tl.float16):
    sig = {"src": tla.buffer_type((16, 64), dtype, src_space), "dst": tla.buffer_type((16, 64), dtype, dst_space)}
    mod, ctx = compile_offline(route, sig, {"P": p, "ROUTE": name})
    return str(lower_offline(mod, ctx))


def check_param_contracts():
    # Typed function arguments isolate frontend mapping without fake L0/BT allocations.
    for dst in ("l0a", "l0b"):
        for transpose in (False, True):
            p = tla.CopyL1ToL0Params(is_transpose=transpose, k_part_idx=0, k_part=64, k_part_ceil=64, k_part_loop=1,
                                     k_part_actual=64, m=16)
            text = route_ir("l1", dst, p, "l0")
            assert "hivm.hir.l12l0 " in text
            for field, value in (("k_part_idx", 0), ("k_part", 64), ("k_part_ceil", 64), ("k_part_loop", 1),
                                 ("k_part_actual", 64), ("m", 16)):
                assert f"{field} = {value}" in text
            if transpose:
                assert "is_transposed = true" in text
            assert "memref.alloc" not in text and "pointer_cast" not in text
    for dtype in (tl.float16, tl.float32):
        text = route_ir("l1", "bt", tla.CopyL12BTParams(n=48), "bt", dtype)
        assert "hivm.hir.l12bt " in text and "48 : index" in text
        assert "memref.alloc" not in text and "pointer_cast" not in text
    text = route_ir("ub", "l1", tla.CopyUB2L1Params(collapse_reassociation=((0, 1), )), "ub_l1")
    assert "collapse_reassociation = [[0, 1]]" in text
    errors = [("l1", "fb", tla.CopyL12FBParams(), "fb", "no L1->FixBUF"),
              ("l1", "l0a", tla.CopyL1ToL0Params(actual_batch=1), "l0", "actual_batch"),
              ("gm", "l1", tla.LoadGM2L1Params(enable_nz2nd=True), "l1", "reverse-layout"),
              ("gm", "l1", tla.LoadGM2L1Params(enable_nz2dn=True), "l1", "reverse-layout"),
              ("ub", "l1", tla.LoadGM2UBParams(), "ub_l1", "CopyUB2L1Params"),
              ("gm", "ub", tla.CopyUB2L1Params(), "ub_l1", "DMA interface expects")]
    for src, dst, p, name, message in errors:
        with pytest.raises(Exception, match=message):
            route_ir(src, dst, p, name)


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    mod, ctx = compile_offline(transfer, {"X": "*fp16", "Y": "*fp16"}, {"N": case["n"]}, output_dir=output_dir)
    text = str(lower_offline(mod, ctx, output_dir=output_dir))
    for op in ("load", "copy", "l12ub", "store"):
        assert text.count("hivm.hir." + op + " ") == 1
    assert text.count("hivm.hir.sync_block_set") == 2
    assert text.count("hivm.hir.sync_block_wait") == 2
    assert "hivm.hir.get_sub_block_idx" in text
    if case["n"] == 16:
        check_param_contracts()
    print("IR PASS:", case["name"])


def launch(case, compile_only=False):
    n = case["n"]
    expected = ((torch.arange(16 * n).reshape(16, n) * 7) % 43 - 21).half()
    packed = expected.reshape(1, 16, n // 16, 16).permute(2, 0, 1, 3).contiguous()
    x, y = packed.npu(), torch.full_like(expected, float("nan"), device="npu")
    if compile_only:
        return transfer.warmup(x, y, N=n, grid=(1, ))
    transfer[(1, )](x, y, N=n)
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
