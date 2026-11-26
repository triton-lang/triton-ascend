"""UB -> L1 -> UB; direct frontend contracts for L12L0/L12BT/FixBUF."""
import re
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
from triton.language.core import builtin
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
        tl.static_assert(False, "unknown explicit DMA route")


def route_ir(src_space, dst_space, p, name, dtype=tl.float16, src_shape=(16, 64), dst_shape=(16, 64)):
    sig = {"src": tla.buffer_type(src_shape, dtype, src_space), "dst": tla.buffer_type(dst_shape, dtype, dst_space)}
    mod, ctx = compile_offline(route, sig, {"P": p, "ROUTE": name})
    return str(lower_offline(mod, ctx))


def check_param_contracts():
    # Typed function arguments isolate frontend mapping without fake L0/BT allocations.
    for dst in ("l0a", "l0b"):
        for transpose in (False, True):
            p = tla.CopyL1ToL0Params(is_transpose=transpose)
            src_shape = (1, 4, 16, 16) if (dst == "l0a") == transpose else (4, 1, 16, 16)
            text = route_ir("l1", dst, p, "l0", src_shape=src_shape, dst_shape=(4, 1, 16, 16))
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
              ("gm", "l1", tla.LoadGM2L1Params(enable_nz2nd=True), "l1", "reverse-layout"),
              ("gm", "l1", tla.LoadGM2L1Params(enable_nz2dn=True), "l1", "reverse-layout"),
              ("ub", "l1", tla.LoadGM2UBParams(), "ub_l1", "CopyUB2L1Params"),
              ("gm", "ub", tla.CopyUB2L1Params(), "ub_l1", "DMA interface expects")]
    for src, dst, p, name, message in errors:
        with pytest.raises(Exception, match=message):
            route_ir(src, dst, p, name)


@tla.jit
def logical_route(DTYPE: tl.constexpr, TO_B: tl.constexpr, TRANSPOSE: tl.constexpr):
    with tla.scope.cube():
        if TO_B:
            rows: tl.constexpr = 64
            cols: tl.constexpr = 32
        else:
            rows: tl.constexpr = 32
            cols: tl.constexpr = 64
        if TRANSPOSE:
            sr: tl.constexpr = cols
            sc: tl.constexpr = rows
        else:
            sr: tl.constexpr = rows
            sc: tl.constexpr = cols
        src = tla.alloc((sr, sc), DTYPE, space="l1")
        dst = tla.alloc((rows, cols), DTYPE, space="l0b" if TO_B else "l0a")
        tla.copy_l1_to_l0(src, dst, tla.CopyL1ToL0Params(is_transpose=TRANSPOSE))


@tla.jit
def transpose_route(src, dst, transpose):
    with tla.scope.cube():
        tla.copy_l1_to_l0(src, dst, tla.CopyL1ToL0Params(is_transpose=transpose))


@tla.jit
def batch_route(src, dst, count):
    with tla.scope.cube():
        tla.copy_l1_to_l0(src, dst, tla.CopyL1ToL0Params(actual_batch=count))


@tla.jit
def logical_slice_route(src, dst, MATRIX_OFFSET: tl.constexpr, ROW_STEP: tl.constexpr):
    with tla.scope.cube():
        s = tla.subview(src, (MATRIX_OFFSET, 16), (32, 64), (ROW_STEP, 1))
        tla.copy_l1_to_l0(s, dst)


@tla.jit
def alloc_subview_route():
    with tla.scope.cube():
        src_parent = tla.alloc((64, 96), tl.float16, space="l1")
        src = tla.subview(src_parent, (16, 16), (32, 64))
        dst = tla.alloc((32, 64), tl.float16, space="l0a")
        tla.copy_l1_to_l0(src, dst)


@builtin
def hinted_buffer(_semantic=None):
    buf = tla.alloc((32, 64), tl.float16, space="l1", _semantic=_semantic)
    # Test-only producer hint: nZ axes are already normalized.
    buf.handle.set_attr("hivm.fractal_layout", _semantic.builder.parse_attr('"nZ"'))
    return buf


@tla.jit
def hinted_route():
    with tla.scope.cube():
        src = hinted_buffer()
        dst = tla.alloc((32, 64), tl.float16, space="l0a")
        tla.copy_l1_to_l0(src, dst, tla.CopyL1ToL0Params(is_transpose=True))


def check_l12l0_contracts():
    shape = (4, 2, 16, 16)  # M=32, K=64
    for dtype in (tl.float16, tl.bfloat16, tl.float32):
        c0 = 8 if dtype == tl.float32 else 16
        physical = (64 // c0, 2, 16, c0)
        for p in (None, tla.CopyL1ToL0Params()):
            text = route_ir("l1", "l0a", p, "l0", dtype, physical, physical)
            copy = next(line for line in text.splitlines() if "hivm.hir.l12l0 " in line)
            assert "k_part_idx = 0" in copy and "k_part_loop = 1" in copy
            assert "k_part = 64" in copy and "m = 32" in copy
            assert "actual_batch" not in copy
        # These malformed fractals still yield K and M/N divisible by 16.
        # Check each endpoint independently; transpose cannot legalize them.
        malformed = ((64 // c0, 2, 8, c0), (64 // c0, 2, 32, c0), (64 // c0, 2, 16, c0 // 2), (64 // c0, 2, 16, c0 * 2))
        for bad_shape in malformed:
            for bad_dst in (False, True):
                for transpose in (False, True):
                    with pytest.raises(Exception, match="physical views require inner dimensions"):
                        route_ir("l1", "l0a", tla.CopyL1ToL0Params(transpose), "l0", dtype,
                                 bad_shape if not bad_dst else physical, bad_shape if bad_dst else physical)
    for to_b in (False, True):
        for transpose in (False, True):
            dst_shape = (64, 32) if to_b else (32, 64)
            src_shape = dst_shape[::-1] if transpose else dst_shape
            text = route_ir("l1", "l0b" if to_b else "l0a", tla.CopyL1ToL0Params(transpose), "l0", src_shape=src_shape,
                            dst_shape=dst_shape)
            copy = next(line for line in text.splitlines() if "hivm.hir.l12l0 " in line)
            assert "x".join(map(str, src_shape)) + "xf16" in copy
            assert "x".join(map(str, dst_shape)) + "xf16" in copy
            assert "k_part = 64" in copy and "m = 32" in copy
    sig = dict(src=tla.buffer_type(shape, tl.float16, "l1"), dst=tla.buffer_type(shape, tl.float16, "l0a"))
    with pytest.raises(Exception, match="compile-time bool"):
        compile_offline(transpose_route, {**sig, "transpose": "i1"}, {})
    with pytest.raises(Exception, match="matching"):
        compile_offline(route, {**sig, "dst": tla.buffer_type(shape, tl.bfloat16, "l0a")}, {"P": None, "ROUTE": "l0"})
    with pytest.raises(Exception, match="matching"):
        route_ir("l1", "l0a", None, "l0", tl.int32, shape, shape)
    with pytest.raises(Exception, match="DMA interface expects"):
        route_ir("ub", "l0a", None, "l0", src_shape=shape, dst_shape=shape)
    with pytest.raises(Exception, match="compile-time bool"):
        route_ir("l1", "l0a", tla.CopyL1ToL0Params(is_transpose=1), "l0")
    for strides in ((1024, 256, 16, 1), (1024, 512, 16, 1)):
        sig = dict(src=tla.buffer_type(shape, tl.float16, "l1", strides=strides),
                   dst=tla.buffer_type(shape, tl.float16, "l0a"))
        mod, ctx = compile_offline(route, sig, {"P": None, "ROUTE": "l0"})
        text = str(lower_offline(mod, ctx))
        # Target stride legality is checked by NPUIR, after this boundary.
        assert f"strided<[{', '.join(map(str, strides))}]>" in text
    for removed in ("k_part_idx", "k_part", "k_part_ceil", "k_part_loop", "k_part_actual", "m"):
        with pytest.raises(TypeError):
            tla.CopyL1ToL0Params(**{removed: 1})
    with pytest.raises(Exception, match="multiples of 16"):
        route_ir("l1", "l0a", None, "l0", src_shape=(3, 16), dst_shape=(3, 16))
    for dtype in (tl.float16, tl.bfloat16, tl.float32):
        for to_b in (False, True):
            for transpose in (False, True):
                constants = dict(DTYPE=dtype, TO_B=to_b, TRANSPOSE=transpose)
                mod, ctx = compile_offline(logical_route, {}, constants)
                text = str(lower_offline(mod, ctx))
                copy = next(line for line in text.splitlines() if "hivm.hir.l12l0 " in line)
                elem = {tl.float16: "f16", tl.bfloat16: "bf16", tl.float32: "f32"}[dtype]
                rows, cols = (64, 32) if to_b else (32, 64)
                sr, sc = (cols, rows) if transpose else (rows, cols)
                assert f"{sr}x{sc}x{elem}" in copy and f"{rows}x{cols}x{elem}" in copy
                assert "k_part = 64" in copy
                assert "m = 32" in copy and "scf.for" not in text
    # The frontend preserves logical parent pitches, slice origins and steps.
    # These checks stop before NPUIR layout legalization and target validation.
    sig = dict(src=tla.buffer_type((64, 96), tl.float16, "l1"), dst=tla.buffer_type((32, 64), tl.float16, "l0a"))
    for origin, step in ((16, 1), (1, 1), (0, 2)):
        mod, ctx = compile_offline(logical_slice_route, sig, {"MATRIX_OFFSET": origin, "ROW_STEP": step})
        text = str(lower_offline(mod, ctx))
        copy = next(line for line in text.splitlines() if "hivm.hir.l12l0 " in line)
        assert "32x64xf16" in copy
        assert f"strided<[{96 * step}, 1], offset: {origin * 96 + 16}>" in copy
        assert "scf.for" not in text
    mod, ctx = compile_offline(alloc_subview_route, {}, {})
    text = str(lower_offline(mod, ctx))
    copy = next(line for line in text.splitlines() if "hivm.hir.l12l0 " in line)
    assert "32x64xf16" in copy and "strided<[96, 1], offset: 1552>" in copy
    mod, ctx = compile_offline(hinted_route, {}, {})
    text = str(lower_offline(mod, ctx))
    copy = next(line for line in text.splitlines() if "hivm.hir.l12l0 " in line)
    assert copy.count("32x64xf16") == 2
    assert 'hivm.fractal_layout = "nZ"' in text
    assert "is_transposed = true" in copy


def check_l12l0_batch_contracts():
    for dtype in (tl.float16, tl.bfloat16, tl.float32):
        c0 = 32 // dtype.itemsize
        for to_b in (False, True):
            dst_space = "l0b" if to_b else "l0a"
            rows, cols = (64, 32) if to_b else (32, 64)
            for transpose in (False, True):
                sr, sc = (cols, rows) if transpose else (rows, cols)
                src_physical = (sc // c0, sr // 16, 16, c0)
                dst_physical = (64 // c0, 2, 16, c0)
                # Different capacities and mixed logical/physical endpoints.
                for src, dst, n in (((sr, sc), dst_physical, 1), (src_physical, (rows, cols), 2)):
                    params = tla.CopyL1ToL0Params(is_transpose=transpose, actual_batch=n)
                    text = route_ir("l1", dst_space, params, "l0", dtype, (4, *src), (2, *dst))
                    copies = [line for line in text.splitlines() if "hivm.hir.l12l0 " in line]
                    assert len(copies) == 1 and "scf.for" not in text
                    count = re.search(r"actual_batch = (%\w+)", copies[0]).group(1)
                    assert f"{count} = arith.constant {n} : index" in text
                    assert "k_part = 64" in copies[0] and "m = 32" in copies[0]
                    assert ("is_transposed = true" in copies[0]) == transpose

    shape = (4, 32, 64)
    physical = (2, 4, 2, 16, 16)
    sig = dict(src=tla.buffer_type(shape, tl.float16, "l1"), dst=tla.buffer_type(physical, tl.float16, "l0a"))
    for count_type in ("i32", "i64", "u8"):
        mod, ctx = compile_offline(batch_route, {**sig, "count": count_type}, {})
        if count_type == "u8":
            assert re.search(r"arith.extui .* : i8 to i64", str(mod))
        text = str(lower_offline(mod, ctx))
        assert text.count("hivm.hir.l12l0 ") == 1 and "scf.for" not in text
        assert re.search(r"actual_batch = %\w+", text) and "arith.index_cast" in text

    for count_type in ("i1", "fp32", "*i32", tl.block_type(tl.int32, [4])):
        with pytest.raises(Exception, match="actual_batch must be an integer scalar"):
            compile_offline(batch_route, {**sig, "count": count_type}, {})
    for n in (True, False, 1.5, 0, -1, 3):
        with pytest.raises(Exception, match="actual_batch"):
            route_ir("l1", "l0a", tla.CopyL1ToL0Params(actual_batch=n), "l0", src_shape=shape, dst_shape=physical)
    with pytest.raises(Exception, match="fit both batch views"):
        route_ir("l1", "l0a", tla.CopyL1ToL0Params(actual_batch=2), "l0", src_shape=(1, 32, 64), dst_shape=physical)
    for src, dst, n in ((shape, physical, None), (shape, physical[1:], None), (shape[1:], physical, None),
                        (shape[1:], physical[1:], 1), (shape, physical[1:], 1), (shape[1:], physical, 1),
                        ((None, 32, 64), physical, 2)):
        with pytest.raises(Exception, match="requires static rank"):
            route_ir("l1", "l0a", tla.CopyL1ToL0Params(actual_batch=n), "l0", src_shape=src, dst_shape=dst)
    bad_physical = (2, 4, 2, 8, 16)
    for src, dst in ((bad_physical, physical), (shape, bad_physical)):
        with pytest.raises(Exception, match="physical views require inner dimensions"):
            route_ir("l1", "l0a", tla.CopyL1ToL0Params(actual_batch=1), "l0", src_shape=src, dst_shape=dst)


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
        check_l12l0_contracts()
        check_l12l0_batch_contracts()
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
