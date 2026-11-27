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
from utils.tla_test_utils import (Options, compile_offline, lower_offline, get_stages, check_kernel_binary, matrix)

BASIC_CONFIGS = [dict(name="ub_l1_" + str(n), n=n) for n in (16, 32, 64)]


@tla.jit
def transfer(X, Y, N: tl.constexpr):
    l1 = tla.alloc((N // 16, 1, 16, 16), tl.float16, space="l1")
    out = tla.alloc((16, N), tl.float16)
    with tla.scope.vector():
        if tla.sub_block_id() == 0:
            src = tla.alloc((N // 16, 1, 16, 16), tl.float16)
            tla.load_gm_to_ub(tla.view(X, (N // 16, 1, 16, 16)), src)
            tla.copy_ub_to_l1(src, l1)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)
    with tla.scope.cube():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
        tla.copy_l1_to_ub(l1, out)
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
        tla.copy_ub_to_l1(src, dst)
    else:
        tl.static_assert(False, "unknown explicit DMA route")


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
    text = route_ir("ub", "l1", None, "ub_l1")
    assert "hivm.hir.copy " in text
    errors = [("l1", "fb", tla.CopyL12FBParams(), "fb", "no L1->FixBUF"),
              ("l1", "l0a", tla.CopyL1ToL0Params(actual_batch=1), "l0", "actual_batch"),
              ("gm", "l1", tla.LoadGM2UBParams(), "l1", "expected LoadGM2L1Params"),
              ("gm", "ub", None, "ub_l1", "DMA interface expects")]
    for src, dst, p, name, message in errors:
        with pytest.raises(Exception, match=message):
            route_ir(src, dst, p, name)


class A5Options(Options):
    arch = "Ascend950DT_9582"


LAYOUT_DTYPES = {
    "fp16": (tl.float16, torch.float16), "bf16": (tl.bfloat16, torch.bfloat16), "fp32": (tl.float32, torch.float32)
}
LAYOUT_CONFIGS = [
    dict(name=f"{route}_{dtype}", route=route, dtype=dtype)
    for route in ("gm", "nz", "strided")
    for dtype in LAYOUT_DTYPES
] + [
    dict(name="l1_n_pitch_fp32", route="nz", dtype="fp32", source_m_step=1),
    dict(name="l1_nm_pitch_fp32", route="nz", dtype="fp32", source_m_step=2),
]


@tla.jit
def transfer_layout(X, Y, DTYPE: tl.constexpr, C: tl.constexpr, ROUTE: tl.constexpr, SOURCE_M_STEP: tl.constexpr):
    # Use two M blocks and three N blocks to distinguish tile traversal order.
    M: tl.constexpr = 32
    N: tl.constexpr = 3 * C
    NZ: tl.constexpr = (3, 2, 16, C)
    STORAGE_M_TILES: tl.constexpr = 2 * SOURCE_M_STEP + 1 if SOURCE_M_STEP else 2
    BACKING_NZ: tl.constexpr = (4, STORAGE_M_TILES, 16, C)
    backing = tla.alloc(BACKING_NZ, DTYPE, space="l1")
    if SOURCE_M_STEP:
        l1 = tla.subview(backing, (1, 0, 0, 0), NZ, (1, SOURCE_M_STEP, 1, 1))
    else:
        l1 = tla.subview(backing, (1, 0, 0, 0), NZ)
    out_backing = tla.alloc((M, N + C), DTYPE)
    out = tla.subview(out_backing, (0, C), (M, N))

    if ROUTE != "gm":
        with tla.scope.vector():
            if tla.sub_block_id() == 0:
                if SOURCE_M_STEP:
                    # Fill the entire backing allocation, including sentinel blocks,
                    # with a contiguous copy. Only L1->UB sees the pitched view.
                    src = tla.alloc(BACKING_NZ, DTYPE)
                    tla.load_gm_to_ub(tla.view(X, BACKING_NZ), src)
                    tla.copy_ub_to_l1(src, backing)
                elif ROUTE == "nz":
                    src = tla.alloc(NZ, DTYPE)
                    tla.load_gm_to_ub(tla.view(X, NZ), src)
                    tla.copy_ub_to_l1(src, l1)
                else:
                    rows: tl.constexpr = 3 * 2 * 16
                    src_backing = tla.alloc((rows, 2 * C), DTYPE)
                    tla.load_gm_to_ub(tla.view(X, (rows, 2 * C)), src_backing)
                    src = tla.subview(src_backing, (0, C), (rows, C))
                    tla.copy_ub_to_l1(src, tla.reshape(l1, (rows, C)))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)

    with tla.scope.cube():
        if ROUTE == "gm":
            src = tla.view(X, (M, N), strides=(N + C, 1), offsets=(0, C))
            tla.load_gm_to_l1(src, l1)
        else:
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
        tla.copy_l1_to_ub(l1, out)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE1, 1)

    with tla.scope.vector():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1)
        if tla.sub_block_id() == 0:
            tla.store_ub_to_gm(out, tla.view(Y, (M, N)))


def _layout_constants(case):
    dtype = case["dtype"]
    return {
        "DTYPE": LAYOUT_DTYPES[dtype][0], "C": 8 if dtype == "fp32" else 16, "ROUTE": case["route"], "SOURCE_M_STEP":
        case.get("source_m_step", 0)
    }


def _check_layout_ir(case):
    ptr = "*" + case["dtype"]
    module, ctx = compile_offline(transfer_layout, {"X": ptr, "Y": ptr}, _layout_constants(case), options=A5Options())
    text = str(lower_offline(module, ctx))
    assert "hivm.hir.l12ub " in text
    assert "hivm.hir.sync_block_set" in text and "hivm.hir.sync_block_wait" in text
    expected_op = "nd2nz" if case["route"] == "gm" else "copy"
    assert f"hivm.hir.{expected_op} " in text
    step = case.get("source_m_step", 0)
    if step:
        l12ub = next(line for line in text.splitlines() if "hivm.hir.l12ub " in line)
        assert f"strided<[{(2 * step + 1) * 128}, {step * 128}, 8, 1]" in l12ub


def _launch_layout(case, compile_only=False):
    args = _layout_constants(case)
    c = args["C"]
    dtype = LAYOUT_DTYPES[case["dtype"]][1]
    # Exact binary fractions, distinguish rows, columns and regions left unwritten.
    nd = (((torch.arange(32 * 3 * c).reshape(32, 3 * c) * 17) % 1009) / 16).to(dtype)
    if case["route"] == "gm":
        data = torch.full((32, 4 * c), -99, dtype=dtype)
        data[:, c:] = nd
    else:
        nz = nd.reshape(2, 16, 3, c).permute(2, 0, 1, 3).contiguous()
        if step := args["SOURCE_M_STEP"]:
            data = torch.full((4, 2 * step + 1, 16, c), -99, dtype=dtype)
            data[1:, :2 * step:step] = nz
        elif case["route"] == "nz":
            data = nz
        else:
            data = torch.full((3 * 2 * 16, 2 * c), -99, dtype=dtype)
            data[:, c:] = nz.reshape(-1, c)
    x = data.npu()
    y = torch.full_like(nd, float("nan"), device="npu")
    if compile_only:
        return transfer_layout.warmup(x, y, grid=(1, ), **args)
    transfer_layout[(1, )](x, y, **args)
    torch.npu.synchronize()
    return y.cpu(), nd


@tla.jit
def copy_fp32_line():
    with tla.scope.vector():
        src = tla.alloc((128, ), tl.float32)
        dst = tla.alloc((128, ), tl.float32, space="l1")
        tla.copy_ub_to_l1(src, dst)


def test_fp32_contiguous_line():
    mod, ctx = compile_offline(copy_fp32_line, {}, {}, options=A5Options())
    assert "hivm.hir.copy " in str(lower_offline(mod, ctx))


@tla.jit
def copy_swapped_nz(src, dst):
    with tla.scope.cube():
        tla.copy_l1_to_ub(src, dst)


def test_fp32_swapped_nz_strides():
    src = tla.buffer_type((2, 2, 16, 8), tl.float32, "l1", (128, 256, 8, 1))
    dst = tla.buffer_type((32, 16), tl.float32, "ub")
    mod, ctx = compile_offline(copy_swapped_nz, {"src": src, "dst": dst}, {}, options=A5Options())
    assert "hivm.hir.l12ub " in str(lower_offline(mod, ctx))


CONFIGS = BASIC_CONFIGS + LAYOUT_CONFIGS


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    if "route" in case:
        _check_layout_ir(case)
        return
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
    if "route" in case:
        return _launch_layout(case, compile_only)
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
