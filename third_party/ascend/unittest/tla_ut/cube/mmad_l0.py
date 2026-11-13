"""A5 standalone MMAD L0: IR / compilation / accuracy cases.

A5 FP16/BF16 -> FP32, one output tile, L1-resident inputs and two explicit L0 slots.
Source counterpart: Catlass::Gemm::L1Mmad -> TileMmadTla -> AscendCBisheng::Mmad.
This demonstrates the L0 ping-pong mechanism, not all CATLASS GM/L1 scheduling.
Suggested shape: M=32, N=64, K=128, L0_K=32; launch grid=(1,).
Requires the matching npuir MMAD L0 schema, A5 planner and CATLASS wrapper.
BF16 cases require the backend BF16 MMAD L0 extension. Accuracy references use
CPU FP32 arithmetic on the same quantized FP16/BF16 inputs sent to the device.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import triton.language as tl
import triton.language.extra.tla as tla
from triton.compiler.errors import CompilationError

from utils.tla_test_utils import Options, compile_offline, lower_offline, check_kernel_binary, get_stages, matrix


class A5Options(Options):
    arch = "Ascend950PR_9589"


DTYPES = ("float16", "bfloat16")
CONFIGS = [
    dict(name=f"{name}_plain" if dtype == "float16" else f"{name}_plain_bf16", dtype=dtype, M=m, N=n, K=k, L0_K=l0_k)
    for dtype in DTYPES
    for name, m, n, k, l0_k in (
        ("single", 32, 64, 32, 32),
        ("odd", 32, 64, 96, 32),
        ("pingpong", 32, 64, 128, 32),
        ("distinct_mkn", 32, 64, 144, 48),
        ("minimum", 16, 16, 16, 16),
    )
]

INVALID_DTYPES = [
    (tl.float16, tl.bfloat16, tl.float32, "a and b must have same dtype"),
    (tl.bfloat16, tl.float16, tl.float32, "a and b must have same dtype"),
    (tl.float32, tl.float32, tl.float32, "a must have dtype"),
    (tl.bfloat16, tl.bfloat16, tl.bfloat16, "acc must have dtype"),
]


def signature(case):
    element = "fp16" if case["dtype"] == "float16" else "bf16"
    return dict(A="*" + element, B="*" + element, C="*fp32")


def constants(case):
    return {name: case[name] for name in ("M", "N", "K", "L0_K")}


@tla.jit
def catlass_mmad_l0_expected(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    L0_K: tl.constexpr,
):
    tl.static_assert(M > 0 and N > 0 and K > 0 and L0_K > 0)
    tl.static_assert(M % 16 == 0 and N % 16 == 0 and L0_K % 16 == 0)
    tl.static_assert(K % L0_K == 0, "this example has no partial K tile")
    tl.static_assert(2 * 2 * M * L0_K <= 64 * 1024)
    tl.static_assert(2 * 2 * N * L0_K <= 64 * 1024)
    tl.static_assert(4 * M * N <= 256 * 1024)
    tl.static_assert(2 * K * (M + N) <= 512 * 1024)

    with tla.scope.cube():
        a_l1 = tla.alloc((K // 16, M // 16, 16, 16), A.dtype.element_ty, space="l1")
        b_l1 = tla.alloc((N // 16, K // 16, 16, 16), B.dtype.element_ty, space="l1")
        # Real separate allocations make slot identity visible. No proposed
        # buffer-indexing API or implicit versions=2 scheduling is assumed.
        a0 = tla.alloc((L0_K // 16, M // 16, 16, 16), A.dtype.element_ty, space="l0a")
        a1 = tla.alloc((L0_K // 16, M // 16, 16, 16), A.dtype.element_ty, space="l0a")
        b0 = tla.alloc((L0_K // 16, N // 16, 16, 16), B.dtype.element_ty, space="l0b")
        b1 = tla.alloc((L0_K // 16, N // 16, 16, 16), B.dtype.element_ty, space="l0b")
        acc = tla.alloc((N // 16, M // 16, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(A, (M, K)), a_l1)
        tla.load_gm_to_l1(tla.view(B, (K, N)), b_l1)
        tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)

        # Initially both slots are available. Each iteration consumes and
        # returns exactly one M->MTE1 event for its own slot.
        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 0)
        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 1)
        for ki in tl.static_range(0, K // L0_K):
            if ki % 2 == 0:
                a_tile = a0
                b_tile = b0
            else:
                a_tile = a1
                b_tile = b1
            tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, ki % 2)
            tla.copy_l1_to_l0(
                a_l1,
                a_tile,
                tla.CopyL1ToL0Params(
                    is_transpose=False,
                    k_part_idx=ki,
                    k_part=L0_K,
                    k_part_ceil=L0_K,
                    k_part_loop=K // L0_K,
                    k_part_actual=L0_K,
                    m=M,
                ),
            )
            tla.copy_l1_to_l0(
                b_l1,
                b_tile,
                tla.CopyL1ToL0Params(
                    is_transpose=False,
                    k_part_idx=ki,
                    k_part=L0_K,
                    k_part_ceil=L0_K,
                    k_part_loop=K // L0_K,
                    k_part_actual=L0_K,
                    m=N,
                ),
            )
            tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
            tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
            tla.cube.mmad_l0(
                a_tile,
                b_tile,
                acc,
                m=M,
                n=N,
                k=L0_K,
                clear_accum=(ki == 0),
                disable_gemv=True,
                unit_flag_ctrl=tla.UnitFlagMode.DISABLED,
            )
            # TileMmadTla owns the small-tile PIPE_M barrier. Do not emit a
            # second barrier here; explicit cross-pipe slot events remain.
            tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, ki % 2)

        # Consume both final free-slot tokens, including an unused slot when
        # K/L0_K == 1. Do not leak event state to later work on this core.
        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 0)
        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 1)
        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
        tla.store_l0c_to_gm(
            acc,
            tla.view(C, (M, N)),
            tla.FixpipeL0C2GMParams(
                dma_mode=tla.FixpipeDMAMode.NZ2ND,
                unit_flag_ctrl=tla.UnitFlagMode.DISABLED,
            ),
        )
        tla.sync.barrier(tla.sync.PIPE.ALL)


def mmad_ops(text):
    return re.findall(r"hivm\.hir\.mmadL0[^\n]*", text)


def input_operands(op):
    return re.search(r"ins\((.*?) :", op).group(1).split(", ")


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    module, ctx = compile_offline(catlass_mmad_l0_expected, signature(case), constants(case), output_dir=output_dir,
                                  options=A5Options())
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    ops = mmad_ops(text)
    iterations = case["K"] // case["L0_K"]
    assert len(ops) == iterations
    values = {name: int(value) for name, value in re.findall(r"(%[\w]+) = arith\.constant (\d+) : index", text)}
    booleans = {name: value == "true" for name, value in re.findall(r"(%[\w]+) = arith\.constant (true|false)", text)}
    input_dtype = "f16" if case["dtype"] == "float16" else "bf16"
    pairs = []
    for index, op in enumerate(ops):
        inputs = input_operands(op)
        assert len(inputs) == 8
        assert tuple(values[value] for value in inputs[2:5]) == (case["M"], case["L0_K"], case["N"])
        assert booleans[inputs[5]] is False  # k_direction_align
        assert booleans[inputs[6]] is False  # is_with_bias
        assert booleans[inputs[7]] == (index == 0)
        assert " bias(" not in op
        assert "unit_flag<DISABLED>" in op
        physical_tiles = (
            ((case["L0_K"] // 16, case["M"] // 16, 16, 16), input_dtype, "ca"),
            ((case["L0_K"] // 16, case["N"] // 16, 16, 16), input_dtype, "cb"),
            ((case["N"] // 16, case["M"] // 16, 16, 16), "f32", "cc"),
        )
        for shape, dtype, space in physical_tiles:
            dimensions = "x".join(str(extent) for extent in shape)
            assert f"memref<{dimensions}x{dtype}, #hivm.address_space<{space}>>" in op
        pairs.append(tuple(inputs[:2]))
    accumulators = [re.search(r"outs\((%[\w]+) :", op).group(1) for op in ops]
    assert len(set(accumulators)) == 1, "all K partitions must update the same accumulator"
    assert len(set(pairs)) == min(2, iterations), "explicit L0 slots must stay distinct"
    assert all(pair == pairs[index % 2] for index, pair in enumerate(pairs)), "wrong ping-pong slot"
    assert text.count("hivm.hir.l12l0 ") == 2 * iterations
    assert "hivm.hir.l12bt " not in text
    assert text.count("hivm.hir.fixpipe ") == 1
    assert "hivm.hir.mmadL1" not in text
    assert "hivm.hir.pipe_barrier[<PIPE_M>]" not in text, "small-tile barrier belongs to CATLASS"
    for event in (0, 1):
        for action in ("set", "wait"):
            assert f"hivm.hir.{action}_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID{event}>]" in text
    assert "!tt.ptr" not in text and "tensor<" not in text
    assert "hivm.multi_buffer" not in text, "slots are explicit allocations"
    print("IR PASS: MMAD L0", case["name"])


@tla.jit
def l0_operand_contract(A, B, C, Clear, UNIT: tl.constexpr):
    with tla.scope.cube():
        tla.cube.mmad_l0(A, B, C, m=32, n=64, k=48, clear_accum=Clear, unit_flag_ctrl=UNIT)


@pytest.mark.parametrize("offset", [0, 256, None])
@pytest.mark.parametrize("unit", [None, tla.UnitFlagMode.DISABLED])
@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
def test_ir_operands(offset, unit, dtype):
    element_type = getattr(tl, dtype)
    signature = dict(A=tla.buffer_type((3, 2, 16, 16), element_type, "l0a", offset=offset), B=tla.buffer_type(
        (3, 4, 16, 16), element_type, "l0b", offset=offset), C=tla.buffer_type((4, 2, 16, 16), tl.float32, "l0c",
                                                                               offset=offset), Clear="i1")
    module, _ = compile_offline(l0_operand_contract, signature, dict(UNIT=unit), options=A5Options())
    op, = mmad_ops(str(module))
    assert ("unit_flag_mode" in op) == (unit is not None)
    assert "bias(" not in op
    input_dtype = "f16" if dtype == "float16" else "bf16"
    assert f"memref<3x2x16x16x{input_dtype}," in op
    assert f"memref<3x4x16x16x{input_dtype}," in op
    assert "memref<4x2x16x16xf32," in op
    # The clear operand must remain the runtime i1 argument, not a
    # Python truth value or an arith.constant created by the binding.
    clear = input_operands(op)[7]
    assert re.search(re.escape(clear) + r": i1", str(module))
    if offset is not None and offset != 0:
        assert op.count("offset: 256>") == 3
    elif offset is None:
        assert op.count("offset: ?>") == 3


@pytest.mark.parametrize("a_dtype,b_dtype,acc_dtype,message", INVALID_DTYPES,
                         ids=("fp16_bf16", "bf16_fp16", "fp32_inputs", "bf16_accumulator"))
def test_ir_invalid_dtypes(a_dtype, b_dtype, acc_dtype, message):
    signature = dict(A=tla.buffer_type((3, 2, 16, 16), a_dtype, "l0a"), B=tla.buffer_type(
        (3, 4, 16, 16), b_dtype, "l0b"), C=tla.buffer_type((4, 2, 16, 16), acc_dtype, "l0c"), Clear="i1")
    with pytest.raises((CompilationError, ValueError), match=re.escape("tla.cube.mmad_l0 " + message)):
        compile_offline(l0_operand_contract, signature, dict(UNIT=None), options=A5Options())


@tla.jit
def l0_compact_subviews(A, B, C):
    with tla.scope.cube():
        a = tla.subview(A, (3, 0, 0, 0), (3, 2, 16, 16))
        b = tla.subview(B, (3, 0, 0, 0), (3, 4, 16, 16))
        acc = tla.subview(C, (4, 0, 0, 0), (4, 2, 16, 16))
        tla.cube.mmad_l0(a, b, acc, m=32, n=64, k=48, clear_accum=True)


@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
def test_ir_subviews(dtype):
    element_type = getattr(tl, dtype)
    signature = dict(A=tla.buffer_type((6, 2, 16, 16), element_type, "l0a"), B=tla.buffer_type(
        (6, 4, 16, 16), element_type, "l0b"), C=tla.buffer_type((8, 2, 16, 16), tl.float32, "l0c"))
    module, ctx = compile_offline(l0_compact_subviews, signature, {}, options=A5Options())
    ast = str(module)
    assert ast.count("memref.subview") == 3
    op, = mmad_ops(ast)
    views = re.findall(r"(%[\w]+) = memref.subview", ast)
    assert input_operands(op)[:2] == views[:2]
    assert re.search(r"outs\(" + re.escape(views[2]) + r" :", op)
    assert "bias(" not in op
    text = str(lower_offline(module, ctx))
    assert len(mmad_ops(text)) == 1 and "!tt.ptr" not in text
    op, = mmad_ops(text)
    input_dtype = "f16" if dtype == "float16" else "bf16"
    assert f"memref<3x2x16x16x{input_dtype}," in op
    assert f"memref<3x4x16x16x{input_dtype}," in op
    assert "memref<4x2x16x16xf32," in op


def _launch(case, seed=7, compile_only=False):
    import torch
    import torch_npu  # noqa: F401

    generator = torch.Generator().manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    a = torch.randn((case["M"], case["K"]), generator=generator).to(dtype)
    b = torch.randn((case["K"], case["N"]), generator=generator).to(dtype)
    x, y = a.npu(), b.npu()
    out = torch.full((case["M"], case["N"]), float("nan"), dtype=torch.float32, device="npu")
    # The kernel supplies cross-pipe synchronization explicitly.
    options = dict(unit_flag=False, sync_solver=False)
    if compile_only:
        kernel = catlass_mmad_l0_expected.warmup(x, y, out, grid=(1, ), **constants(case), **options)
        return kernel, None, None
    kernel = catlass_mmad_l0_expected[(1, )](x, y, out, **constants(case), **options)
    torch.npu.synchronize()
    # Compare FP32 accumulation of the actual FP16/BF16 inputs, without
    # rounding the reference result back to the input dtype.
    expected = a.float() @ b.float()
    return kernel, out.cpu(), expected


@matrix(CONFIGS)
def test_compile(case):
    kernel, _, _ = _launch(case, compile_only=True)
    check_kernel_binary(kernel)


@matrix(CONFIGS)
def test_run(case):
    import torch

    for seed in (7, 19):
        _, actual, expected = _launch(case, seed)
        assert actual.isfinite().all(), "MMAD L0 left unwritten or non-finite output"
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        print("ACCURACY PASS: MMAD L0", case["name"], "seed", seed, "max_abs_error",
              (actual - expected).abs().max().item())


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)
    if do_ir:
        for dtype in DTYPES:
            test_ir_subviews(dtype)
            for offset in (0, 256, None):
                for unit in (None, tla.UnitFlagMode.DISABLED):
                    test_ir_operands(offset, unit, dtype)
        for args in INVALID_DTYPES:
            test_ir_invalid_dtypes(*args)


if __name__ == "__main__":
    main()
