"""Stride transfers: emitted IR, device compilation and independent references.

Run on the development machine with TLA_TEST_STAGE=ir/compile/run, using
pytest or this file directly. The run stage checks load padding separately
from masked stores and compares complete GM allocations, including guards.
"""
import itertools
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from triton.compiler.errors import CompilationError

from utils.tla_test_utils import compile_offline, lower_offline, get_stages, check_kernel_binary, matrix


DTYPES = {
    "fp16": (tl.float16, torch.float16),
    "bf16": (tl.bfloat16, torch.bfloat16),
    "fp32": (tl.float32, torch.float32),
    "i32": (tl.int32, torch.int32),
}
LAYOUTS = [
    dict(shape=(16, ), strides=(3, ), out_strides=(5, ), numels=(11, )),
    dict(shape=(3, 16), strides=(53, 2), out_strides=(61, 3), numels=(2, 13)),
    dict(shape=(2, 3, 8), strides=(151, 37, 3), out_strides=(173, 41, 2), numels=(1, 2, 5)),
]
CONFIGS = [
    dict(layout, name=f"{dtype}_{rank}d", dtype=dtype, pitched=False)
    for dtype in DTYPES for rank, layout in enumerate(LAYOUTS, 1)
]
CONFIGS += [
    dict(LAYOUTS[1], name="pitched_ub", dtype="fp32", pitched=True),
    dict(LAYOUTS[1], name="empty", dtype="fp32", pitched=True, numels=(0, 13)),
    dict(LAYOUTS[1], name="negative_count", dtype="fp32", pitched=False, numels=(-1, 13)),
    dict(LAYOUTS[2], name="clipped_counts", dtype="fp32", pitched=False, numels=(7, 9, 20)),
    dict(name="singleton_zero_stride", dtype="fp32", pitched=False, shape=(1, 16),
         strides=(0, 2), out_strides=(0, 3), numels=(1, 11)),
]


@tla.jit
def transfer(X, Y, Dump, row, col, offset, stride0, numel0, other,
             SHAPE: tl.constexpr, STRIDES: tl.constexpr, OUT_STRIDES: tl.constexpr,
             NUMELS: tl.constexpr, DTYPE: tl.constexpr, PITCHED: tl.constexpr):
    with tla.scope.vector():
        # Multi-dimensional GM subviews force rank-one descriptor adaptation
        # while keeping nonzero, runtime view offsets.
        src = tla.subview(tla.view(X, (16, 64)), (row, col), (8, 48))
        dst = tla.subview(tla.view(Y, (16, 64)), (1, 7), (8, 48))
        if PITCHED:
            backing = tla.alloc((5, 24), DTYPE)
            tile = tla.subview(backing, (1, 4), SHAPE)
        else:
            tile = tla.alloc(SHAPE, DTYPE)
        if len(SHAPE) == 1:
            tla.stride_load(src, tile, offset, other, (stride0, ), (numel0, ))
            tla.stride_store(tile, dst, 9, OUT_STRIDES, (numel0, ))
            tla.stride_store(tile, tla.view(Dump, (256, )), 7, (1, ), SHAPE)
        elif len(SHAPE) == 2:
            tla.stride_load(src, tile, offset, other, (stride0, STRIDES[1]), (numel0, NUMELS[1]))
            tla.stride_store(tile, dst, 9, OUT_STRIDES, (numel0, NUMELS[1]))
            tla.stride_store(tile, tla.view(Dump, (256, )), 7, (SHAPE[1], 1), SHAPE)
        else:
            tla.stride_load(src, tile, offset, other, (stride0, STRIDES[1], STRIDES[2]),
                            (numel0, NUMELS[1], NUMELS[2]))
            tla.stride_store(tile, dst, 9, OUT_STRIDES, (numel0, NUMELS[1], NUMELS[2]))
            tla.stride_store(tile, tla.view(Dump, (256, )), 7, (SHAPE[1] * SHAPE[2], SHAPE[2], 1), SHAPE)


def constants(case):
    return dict(SHAPE=case["shape"], STRIDES=case["strides"], OUT_STRIDES=case["out_strides"],
                NUMELS=case["numels"], DTYPE=DTYPES[case["dtype"]][0], PITCHED=case["pitched"])


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    dtype = case["dtype"]
    signature = dict(X="*" + dtype, Y="*" + dtype, Dump="*" + dtype, row="i32", col="i32",
                     offset="i64", stride0="i32", numel0="i32", other="fp32")
    module, ctx = compile_offline(transfer, signature, constants(case), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    loads = [line for line in text.splitlines() if "hivm.hir.stride_load " in line]
    stores = [line for line in text.splitlines() if "hivm.hir.stride_store " in line]
    assert len(loads) == 1 and len(stores) == 2
    assert "#hivm.vf_mode<SIMT>" in loads[0]
    assert "offset(" in loads[0] and ": i64)" in loads[0]
    assert "arith.extsi" in text, "mixed i32/i64 indices must be widened"
    # GM is rank one on both ops, UB retains its original rank.
    assert re.search(r"ins\([^:]+: memref<\d+x(?:f\d+|bf16|i\d+)[,>]", loads[0])
    assert all(re.search(r"outs\([^:]+: memref<\d+x(?:f\d+|bf16|i\d+)[,>]", line) for line in stores)
    tile_shape = "x".join(str(size) for size in case["shape"]) + "x"
    assert re.search(r"outs\([^:]+: memref<" + tile_shape, loads[0])
    assert all(re.search(r"ins\([^:]+: memref<" + tile_shape, line) for line in stores)
    assert "memref.subview" in text or "memref.reinterpret_cast" in text


def launch(case, compile_only=False):
    torch_dtype = DTYPES[case["dtype"]][1]
    x = ((torch.arange(1024) % 101) - 50).to(torch_dtype)
    y = torch.full((1024, ), -97, dtype=torch_dtype)
    dump = torch.full((256, ), -97, dtype=torch_dtype)
    dx, dy, ddump = x.npu(), y.npu(), dump.npu()
    args = (dx, dy, ddump, 2, 3, 5, case["strides"][0], case["numels"][0], -7.0)
    if compile_only:
        return transfer.warmup(*args, grid=(1, ), **constants(case))
    transfer[(1, )](*args, **constants(case))
    torch.npu.synchronize()

    tile = torch.full(case["shape"], -7, dtype=torch_dtype)
    for index in itertools.product(*(range(size) for size in case["shape"])):
        if all(i < n for i, n in zip(index, case["numels"])):
            read_at = 2 * 64 + 3 + 5 + sum(i * s for i, s in zip(index, case["strides"]))
            write_at = 64 + 7 + 9 + sum(i * s for i, s in zip(index, case["out_strides"]))
            tile[index] = x[read_at]
            y[write_at] = x[read_at]
    dump[7:7 + math.prod(case["shape"])] = tile.flatten()
    return dy.cpu(), y, ddump.cpu(), dump


@matrix(CONFIGS)
def test_compile(case):
    check_kernel_binary(launch(case, compile_only=True))


@matrix(CONFIGS)
def test_run(case):
    actual, expected, actual_dump, expected_dump = launch(case)
    torch.testing.assert_close(actual_dump, expected_dump, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@tla.jit
def invalid_transfer(src, dst, OFFSET: tl.constexpr, STRIDES: tl.constexpr, NUMELS: tl.constexpr,
                     SCOPE: tl.constexpr):
    if SCOPE == "cube":
        with tla.scope.cube():
            tla.stride_load(src, dst, OFFSET, 0, STRIDES, NUMELS)
    elif SCOPE == "simd":
        with tla.scope.simd():
            tla.stride_load(src, dst, OFFSET, 0, STRIDES, NUMELS)
    else:
        tla.stride_load(src, dst, OFFSET, 0, STRIDES, NUMELS)


@pytest.mark.parametrize("changes,match", [
    (dict(STRIDES=(1, )), "one per UB tile dimension"),
    (dict(NUMELS=(16, )), "one per UB tile dimension"),
    (dict(OFFSET=1.5), "scalar integer"),
    (dict(OFFSET=True), "scalar integer"),
    (dict(OFFSET=2**31), "signed i32"),
    (dict(STRIDES=(-16, 1)), "must be positive"),
    (dict(STRIDES=(0, 1)), "singleton"),
    (dict(SCOPE="cube"), "AIV core"),
    (dict(SCOPE="simd"), "outside SIMD/SIMT"),
])
def test_invalid_arguments(changes, match):
    signature = dict(src=tla.buffer_type((128, ), tl.float32, "gm"),
                     dst=tla.buffer_type((4, 16), tl.float32, "ub"))
    params = dict(OFFSET=0, STRIDES=(16, 1), NUMELS=(4, 16), SCOPE="vector")
    params.update(changes)
    with pytest.raises(CompilationError, match=match):
        compile_offline(invalid_transfer, signature, params)


@pytest.mark.parametrize("src_type,dst_type,match", [
    ("*fp32", tla.buffer_type((4, 16), tl.float32, "ub"), "two tla.buffer"),
    (tla.buffer_type((128, ), tl.float32, "ub"), tla.buffer_type((4, 16), tl.float32, "gm"), "GM->UB"),
    (tla.buffer_type((128, ), tl.float32, "gm"), tla.buffer_type((4, 16), tl.float16, "ub"), "matching"),
    (tla.buffer_type((128, ), tl.float64, "gm"), tla.buffer_type((4, 16), tl.float64, "ub"), "element type"),
    (tla.buffer_type((128, ), tl.float32, "gm"), tla.buffer_type((1, 1, 4, 16), tl.float32, "ub"), "rank"),
])
def test_invalid_buffers(src_type, dst_type, match):
    with pytest.raises(CompilationError, match=match):
        compile_offline(invalid_transfer, dict(src=src_type, dst=dst_type),
                        dict(OFFSET=0, STRIDES=(16, 1), NUMELS=(4, 16), SCOPE="vector"))


@tla.jit
def scalar_arguments(src, dst, offset, stride, numel, other, STORE: tl.constexpr):
    if STORE:
        tla.stride_store(dst, src, offset, (stride, ), (numel, ))
    else:
        tla.stride_load(src, dst, offset, other, (stride, ), (numel, ))


@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("index_type,ir_type", [("i16", "i32"), ("u32", "i64"), ("i64", "i64")])
def test_runtime_index_promotion(store, index_type, ir_type):
    signature = dict(src=tla.buffer_type((128, ), tl.float32, "gm"),
                     dst=tla.buffer_type((16, ), tl.float32, "ub"), offset="i32", stride=index_type,
                     numel="i32", other="fp32")
    module, _ = compile_offline(scalar_arguments, signature, dict(STORE=store))
    line = next(line for line in str(module).splitlines() if "hivm.hir.stride_" in line)
    # The op verifier additionally requires all stride/numel operands to have
    # this same type. Unsigned i32 must zero-extend when promoted to i64.
    assert re.search(r"offset\([^:]+: " + ir_type + r"\)", line)
    if index_type == "u32":
        assert "arith.extui" in str(module)


@pytest.mark.parametrize("argument,arg_type,match", [
    ("offset", "fp32", "scalar integer"),
    ("stride", tl.block_type(tl.int32, [4]), "scalar integer"),
    ("numel", "*i32", "scalar integer"),
    ("other", tl.block_type(tl.float32, [4]), "scalar number"),
    ("other", "*fp32", "scalar number"),
])
def test_nonscalar_or_noninteger_arguments(argument, arg_type, match):
    signature = dict(src=tla.buffer_type((128, ), tl.float32, "gm"),
                     dst=tla.buffer_type((16, ), tl.float32, "ub"), offset="i32", stride="i32",
                     numel="i32", other="fp32")
    signature[argument] = arg_type
    with pytest.raises(CompilationError, match=match):
        compile_offline(scalar_arguments, signature, dict(STORE=False))


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
