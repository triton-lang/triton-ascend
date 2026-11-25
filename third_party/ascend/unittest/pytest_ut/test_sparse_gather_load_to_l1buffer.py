# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""TTIR contract for ``al.sparse_gather_load_to_L1buffer``.

These are host-only: they stop at ast_to_ttir and never reach a device. What the
frontend has to get right is the shape of the op it emits, because everything
downstream keys off it -- the destination has to arrive as an output rather than
an input, it has to be declared as living in L1 and already in fractal layout,
and the library call name has to follow the operand dtypes. Getting any of those
wrong is not a compile error, it is a wrong kernel.

Run with: python -m pytest -q third_party/ascend/unittest/pytest_ut/test_sparse_gather_load_to_l1buffer.py
"""
import re

import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource

REGION_VALID_SHIFT = tl.constexpr(24)
REGION_ID_MASK = tl.constexpr((1 << 24) - 1)

BUILTIN = "__builtin_sparse_gather_load_to_L1buffer"
# base, index and the eight scalars; the destination follows them as the output.
NUM_INPUTS = 10


@pytest.fixture(scope="module", autouse=True)
def assign_npu():
    """Override the parent conftest's device fixture: these tests are host-only."""


@triton.jit
def _gather(dst, k_ptr, packed_ptr, BLOCK_R: tl.constexpr, REGION: tl.constexpr, DIM: tl.constexpr):
    al.sparse_gather_load_to_L1buffer(dst, k_ptr, packed_ptr, block_r=BLOCK_R, valid_region_count=BLOCK_R,
                                      region_size=REGION, dim_size=DIM, stride_token=DIM,
                                      region_valid_shift=REGION_VALID_SHIFT, region_id_mask=REGION_ID_MASK)


@triton.jit
def gather_fp16(k_ptr, packed_ptr, REGION: tl.constexpr, DIM: tl.constexpr, BLOCK_R: tl.constexpr):
    dst = bl.alloc(tl.float16, [BLOCK_R * REGION, DIM], al.ascend_address_space.L1)
    _gather(dst, k_ptr, packed_ptr, BLOCK_R, REGION, DIM)


@triton.jit
def gather_bf16(k_ptr, packed_ptr, REGION: tl.constexpr, DIM: tl.constexpr, BLOCK_R: tl.constexpr):
    dst = bl.alloc(tl.bfloat16, [BLOCK_R * REGION, DIM], al.ascend_address_space.L1)
    _gather(dst, k_ptr, packed_ptr, BLOCK_R, REGION, DIM)


@triton.jit
def dst_in_ub(k_ptr, packed_ptr, REGION: tl.constexpr, DIM: tl.constexpr, BLOCK_R: tl.constexpr):
    dst = bl.alloc(tl.float16, [BLOCK_R * REGION, DIM], al.ascend_address_space.UB)
    _gather(dst, k_ptr, packed_ptr, BLOCK_R, REGION, DIM)


@triton.jit
def dst_too_short(k_ptr, packed_ptr, REGION: tl.constexpr, DIM: tl.constexpr, BLOCK_R: tl.constexpr):
    dst = bl.alloc(tl.float16, [BLOCK_R * REGION // 2, DIM], al.ascend_address_space.L1)
    _gather(dst, k_ptr, packed_ptr, BLOCK_R, REGION, DIM)


@triton.jit
def dst_dtype_differs_from_base(k_ptr, packed_ptr, REGION: tl.constexpr, DIM: tl.constexpr, BLOCK_R: tl.constexpr):
    dst = bl.alloc(tl.float32, [BLOCK_R * REGION, DIM], al.ascend_address_space.L1)
    _gather(dst, k_ptr, packed_ptr, BLOCK_R, REGION, DIM)


def build_ttir(kernel, base_dtype="*fp16", dim=16, region=16, block_r=2):
    src = ASTSource(kernel, {"k_ptr": base_dtype, "packed_ptr": "*i32"},
                    {"REGION": region, "DIM": dim, "BLOCK_R": block_r})
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    return str(ast_to_ttir(kernel, src, context, NPUOptions(), {}, {}))


def error_text(exc):
    # A guard trips inside the inlined `_gather`, so the message the builtin
    # raised sits on a chained CompilationError rather than on the one the
    # caller sees.
    parts = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts.append(str(exc))
        exc = exc.__cause__ or exc.__context__
    return "\n".join(parts)


def custom_op_line(ttir):
    for line in ttir.splitlines():
        if BUILTIN in line:
            return line.strip()
    pytest.fail(f"no {BUILTIN} op in:\n{ttir}")


def test_destination_is_an_output_not_an_input():
    # The template writes the destination in place. It has to reach the op as a
    # DPS init, since an operand listed under ins() would be read instead.
    line = custom_op_line(build_ttir(gather_fp16))
    ins, outs = line.split(" outs(")
    assert "%dst" in outs
    assert "%dst" not in ins.split(" ins(")[1]


def test_op_yields_no_result_for_a_buffer_destination():
    # A buffer out is written in place, so the op must not also return it.
    # A result here becomes an sret pointer on the library call, which shifts
    # every real argument by one position.
    line = custom_op_line(build_ttir(gather_fp16))
    assert " -> " not in line


def test_output_is_declared_l1_and_fractal():
    # hivm-insert-load-store-for-mix-cv reads the address space off arg_attrs
    # rather than guessing from the pipe (which would say GM), and the Fractal
    # layout lets a consumer's ND -> Fractal convert_layout fold away instead of
    # synthesising a vector transpose that drags the tile back out of L1.
    line = custom_op_line(build_ttir(gather_fp16))
    arg_attrs = re.search(r"arg_attrs = \[(.*?)\], ", line).group(1)
    entries = re.findall(r"\{[^{}]*\}", arg_attrs)
    assert len(entries) == NUM_INPUTS + 1
    # Only the destination carries attributes; it sits after the inputs.
    assert all(entry == "{}" for entry in entries[:NUM_INPUTS])
    assert "#hivm.address_space<cbuf>" in entries[NUM_INPUTS]
    assert "#hivm.data_layout<Fractal>" in entries[NUM_INPUTS]


def test_runs_on_the_cube_core_over_mte2():
    # GM -> L1 is a cube-core MTE2 path on A5; no other target has the instruction.
    line = custom_op_line(build_ttir(gather_fp16))
    assert "#hivm.tcore_type<CUBE>" in line
    assert "#hivm.pipe<PIPE_MTE2>" in line


@pytest.mark.parametrize("kernel, base_dtype, symbol", [
    (gather_fp16, "*fp16", "sparse_gather_load_to_L1buffer_half_int32_t"),
    (gather_bf16, "*bf16", "sparse_gather_load_to_L1buffer_bfloat16_t_int32_t"),
])
def test_symbol_follows_the_operand_dtypes(kernel, base_dtype, symbol):
    # Must stay in step with REGISTER_SPARSE_GATHER_LOAD_TO_L1BUFFER in the
    # template; a name that resolves to nothing fails only at link time.
    line = custom_op_line(build_ttir(kernel, base_dtype=base_dtype))
    assert f'symbol = "{symbol}"' in line


def test_stride_token_is_widened_to_i64():
    # stride_token is an element count that can exceed what the other scalar
    # arguments carry, so the builtin overrides its type.
    line = custom_op_line(build_ttir(gather_fp16))
    operand_types = line.split(" : ")[1].split(")")[0]
    assert operand_types.split(", ")[6] == "i64"


def test_destination_is_allocated_in_l1():
    ttir = build_ttir(gather_fp16)
    assert re.search(r"memref\.alloc\(\).*memref<32x16xf16, #hivm\.address_space<cbuf>>", ttir)


@pytest.mark.parametrize("kernel, dim, message", [
    (dst_in_ub, 16, "out must be a buffer allocated in L1"),
    (dst_too_short, 16, "needs block_r * region_size * dim_size"),
    (dst_dtype_differs_from_base, 16, "out should have same dtype as base"),
    (gather_fp16, 8, "must be 32B aligned"),
])
def test_rejects_a_destination_the_template_would_corrupt(kernel, dim, message):
    # The template fills block_r * region_size token rows without bounds
    # checking and addresses a row as one 32B block, so a destination that
    # disagrees with the scalar arguments is silent corruption on the device.
    with pytest.raises(Exception) as excinfo:
        build_ttir(kernel, dim=dim)
    assert message in error_text(excinfo.value)
