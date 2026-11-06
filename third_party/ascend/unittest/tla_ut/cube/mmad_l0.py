"""A5 standalone MMAD L0: IR / compilation / accuracy cases.

A5 FP16 -> FP32, one output tile, L1-resident inputs and two explicit L0 slots.
Source counterpart: Catlass::Gemm::L1Mmad -> TileMmadTla -> AscendCBisheng::Mmad.
This demonstrates the L0 ping-pong mechanism, not all CATLASS GM/L1 scheduling.
Suggested shape: M=32, N=64, K=128, L0_K=32; launch grid=(1,).
BT is only read at ki=0, so one explicit BT allocation is sufficient here.
Requires the matching npuir MMAD L0 schema, A5 planner and CATLASS wrapper.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import Options, compile_offline, lower_offline, check_kernel_binary, get_stages, matrix


class A5Options(Options):
    arch = "Ascend950PR_9589"


SIGNATURE = dict(A="*fp16", B="*fp16", Bias="*fp32", C="*fp32")
CONFIGS = [
    dict(name=f"{name}_{'bias' if bias else 'plain'}", M=m, N=n, K=k, L0_K=l0_k, WITH_BIAS=bias)
    for name, m, n, k, l0_k in (
        ("single", 32, 64, 32, 32),
        ("odd", 32, 64, 96, 32),
        ("pingpong", 32, 64, 128, 32),
        ("distinct_mkn", 32, 64, 144, 48),
        ("minimum", 16, 16, 16, 16),
    )
    for bias in (False, True)
]


def constants(case):
    return {name: case[name] for name in ("M", "N", "K", "L0_K", "WITH_BIAS")}


@tla.jit
def catlass_mmad_l0_expected(
    A, B, Bias, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    L0_K: tl.constexpr, WITH_BIAS: tl.constexpr,
):
    tl.static_assert(M > 0 and N > 0 and K > 0 and L0_K > 0)
    tl.static_assert(M % 16 == 0 and N % 16 == 0 and L0_K % 16 == 0)
    tl.static_assert(K % L0_K == 0, "this example has no partial K tile")
    tl.static_assert(2 * 2 * M * L0_K <= 64 * 1024)
    tl.static_assert(2 * 2 * N * L0_K <= 64 * 1024)
    tl.static_assert(4 * M * N <= 256 * 1024)
    tl.static_assert(2 * K * (M + N) + 4 * N <= 512 * 1024)
    if WITH_BIAS:
        tl.static_assert(4 * N <= 4 * 1024)
        bias_l1 = tla.alloc((N,), tl.float32, space="l1")
        with tla.scope.vector():
            if tla.sub_block_id() == 0:
                bias_ub = tla.alloc((N,), tl.float32, space="ub")
                tla.load_gm_to_ub(tla.view(Bias, (N,)), bias_ub)
                tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE3, 0)
                tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE3, 0)
                tla.copy_ub_to_l1(bias_ub, bias_l1, tla.CopyUB2L1Params())
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)

    with tla.scope.cube():
        a_l1 = tla.alloc((K // 16, M // 16, 16, 16), tl.float16, space="l1")
        b_l1 = tla.alloc((N // 16, K // 16, 16, 16), tl.float16, space="l1")
        # Real separate allocations make slot identity visible. No proposed
        # buffer-indexing API or implicit versions=2 scheduling is assumed.
        a0 = tla.alloc((L0_K // 16, M // 16, 16, 16), tl.float16, space="l0a")
        a1 = tla.alloc((L0_K // 16, M // 16, 16, 16), tl.float16, space="l0a")
        b0 = tla.alloc((L0_K // 16, N // 16, 16, 16), tl.float16, space="l0b")
        b1 = tla.alloc((L0_K // 16, N // 16, 16, 16), tl.float16, space="l0b")
        acc = tla.alloc((N // 16, M // 16, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(A, (M, K)), a_l1)
        tla.load_gm_to_l1(tla.view(B, (K, N)), b_l1)
        tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        if WITH_BIAS:
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
            bias_bt = tla.alloc((N,), tl.float32, space="bt")
            tla.copy_l1_to_bt(bias_l1, bias_bt, tla.CopyL12BTParams(n=N))

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
                a_l1, a_tile,
                tla.CopyL1ToL0Params(
                    is_transpose=False, k_part_idx=ki, k_part=L0_K,
                    k_part_ceil=L0_K, k_part_loop=K // L0_K,
                    k_part_actual=L0_K, m=M,
                ),
            )
            tla.copy_l1_to_l0(
                b_l1, b_tile,
                tla.CopyL1ToL0Params(
                    is_transpose=False, k_part_idx=ki, k_part=L0_K,
                    k_part_ceil=L0_K, k_part_loop=K // L0_K,
                    k_part_actual=L0_K, m=N,
                ),
            )
            tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
            tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
            if WITH_BIAS and ki == 0:
                tla.cube.mmad_l0(
                    a_tile, b_tile, acc, m=M, n=N, k=L0_K,
                    bias=bias_bt, clear_accum=False, disable_gemv=True,
                    unit_flag_ctrl=tla.UnitFlagMode.DISABLED,
                )
            else:
                tla.cube.mmad_l0(
                    a_tile, b_tile, acc, m=M, n=N, k=L0_K,
                    clear_accum=(ki == 0), disable_gemv=True,
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
            acc, tla.view(C, (M, N)),
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
    module, ctx = compile_offline(catlass_mmad_l0_expected, SIGNATURE, constants(case), output_dir=output_dir,
                                  options=A5Options())
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    ops = mmad_ops(text)
    iterations = case["K"] // case["L0_K"]
    assert len(ops) == iterations
    values = {name: int(value) for name, value in re.findall(r"(%[\w]+) = arith\.constant (\d+) : index", text)}
    booleans = {name: value == "true" for name, value in re.findall(r"(%[\w]+) = arith\.constant (true|false)", text)}
    pairs = []
    for index, op in enumerate(ops):
        inputs = input_operands(op)
        assert len(inputs) == 8
        assert tuple(values[value] for value in inputs[2:5]) == (case["M"], case["L0_K"], case["N"])
        assert booleans[inputs[5]] is False  # k_direction_align
        assert booleans[inputs[6]] == (case["WITH_BIAS"] and index == 0)
        assert booleans[inputs[7]] == (not case["WITH_BIAS"] and index == 0)
        assert (" bias(" in op) == (case["WITH_BIAS"] and index == 0)
        assert "unit_flag<DISABLED>" in op
        for shape, dtype, space in (((case["L0_K"] // 16, case["M"] // 16, 16, 16), "f16", "ca"),
                                    ((case["L0_K"] // 16, case["N"] // 16, 16, 16), "f16", "cb"),
                                    ((case["N"] // 16, case["M"] // 16, 16, 16), "f32", "cc")):
            dimensions = "x".join(str(extent) for extent in shape)
            assert f"memref<{dimensions}x{dtype}, #hivm.address_space<{space}>>" in op
        pairs.append(tuple(inputs[:2]))
    accumulators = [re.search(r"outs\((%[\w]+) :", op).group(1) for op in ops]
    assert len(set(accumulators)) == 1, "all K partitions must update the same accumulator"
    assert len(set(pairs)) == min(2, iterations), "explicit L0 slots must stay distinct"
    assert all(pair == pairs[index % 2] for index, pair in enumerate(pairs)), "wrong ping-pong slot"
    assert text.count("hivm.hir.l12l0 ") == 2 * iterations
    assert text.count("hivm.hir.l12bt ") == int(case["WITH_BIAS"])
    assert text.count("hivm.hir.fixpipe ") == 1
    assert "hivm.hir.mmadL1" not in text
    assert "hivm.hir.pipe_barrier[<PIPE_M>]" not in text, "small-tile barrier belongs to CATLASS"
    for event in (0, 1):
        for action in ("set", "wait"):
            assert f"hivm.hir.{action}_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID{event}>]" in text
    if case["WITH_BIAS"]:
        bias = re.search(r"bias\((%[\w]+) :", ops[0]).group(1)
        copy = next(line for line in text.splitlines() if "hivm.hir.l12bt " in line)
        assert re.search(r"\bouts\(" + re.escape(bias) + r"\s*:", copy), "MMAD must read the copied BT buffer"
    assert "!tt.ptr" not in text and "tensor<" not in text
    assert "hivm.multi_buffer" not in text, "slots are explicit allocations"
    print("IR PASS: MMAD L0", case["name"])


@tla.jit
def l0_operand_contract(A, B, C, Bias, Clear, WITH_BIAS: tl.constexpr, UNIT: tl.constexpr):
    with tla.scope.cube():
        if WITH_BIAS:
            tla.cube.mmad_l0(A, B, C, m=32, n=64, k=48, bias=Bias, clear_accum=False, unit_flag_ctrl=UNIT)
        else:
            tla.cube.mmad_l0(A, B, C, m=32, n=64, k=48, clear_accum=Clear, unit_flag_ctrl=UNIT)


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("offset", [0, 256, None])
@pytest.mark.parametrize("unit", [None, tla.UnitFlagMode.DISABLED])
def test_ir_operands(with_bias, offset, unit):
    signature = dict(A=tla.buffer_type((3, 2, 16, 16), tl.float16, "l0a", offset=offset),
                     B=tla.buffer_type((3, 4, 16, 16), tl.float16, "l0b", offset=offset),
                     C=tla.buffer_type((4, 2, 16, 16), tl.float32, "l0c", offset=offset),
                     Bias=tla.buffer_type((64,), tl.float32, "bt", offset=16), Clear="i1")
    module, _ = compile_offline(l0_operand_contract, signature, dict(WITH_BIAS=with_bias, UNIT=unit),
                                options=A5Options())
    op, = mmad_ops(str(module))
    assert ("unit_flag_mode" in op) == (unit is not None)
    assert ("bias(" in op) == with_bias
    if with_bias:
        assert "offset: 16>" in op and "address_space<biasbuf>" in op
    else:
        # The clear operand must remain the runtime i1 argument, not a
        # Python truth value or an arith.constant created by the binding.
        clear = input_operands(op)[7]
        assert re.search(re.escape(clear) + r": i1", str(module))
    if offset is not None and offset != 0:
        assert op.count("offset: 256>") == 3
    elif offset is None:
        assert op.count("offset: ?>") == 3


@tla.jit
def l0_compact_subviews(A, B, C, Bias):
    with tla.scope.cube():
        a = tla.subview(A, (3, 0, 0, 0), (3, 2, 16, 16))
        b = tla.subview(B, (3, 0, 0, 0), (3, 4, 16, 16))
        acc = tla.subview(C, (4, 0, 0, 0), (4, 2, 16, 16))
        bias = tla.subview(Bias, (16,), (64,))
        tla.cube.mmad_l0(a, b, acc, m=32, n=64, k=48, bias=bias)


def test_ir_subviews():
    signature = dict(A=tla.buffer_type((6, 2, 16, 16), tl.float16, "l0a"),
                     B=tla.buffer_type((6, 4, 16, 16), tl.float16, "l0b"),
                     C=tla.buffer_type((8, 2, 16, 16), tl.float32, "l0c"),
                     Bias=tla.buffer_type((128,), tl.float32, "bt"))
    module, ctx = compile_offline(l0_compact_subviews, signature, {}, options=A5Options())
    ast = str(module)
    assert ast.count("memref.subview") == 4
    op, = mmad_ops(ast)
    views = re.findall(r"(%[\w]+) = memref.subview", ast)
    assert input_operands(op)[:2] == views[:2]
    assert re.search(r"outs\(" + re.escape(views[2]) + r" :", op)
    assert re.search(r"bias\(" + re.escape(views[3]) + r" :", op)
    text = str(lower_offline(module, ctx))
    assert len(mmad_ops(text)) == 1 and "!tt.ptr" not in text


def _launch(case, seed=7, compile_only=False):
    import torch
    import torch_npu  # noqa: F401

    generator = torch.Generator().manual_seed(seed)
    a = torch.randn((case["M"], case["K"]), generator=generator).to(torch.float16)
    b = torch.randn((case["K"], case["N"]), generator=generator).to(torch.float16)
    bias = torch.randn((case["N"],), generator=generator, dtype=torch.float32)
    x, y, bt = a.npu(), b.npu(), bias.npu()
    out = torch.full((case["M"], case["N"]), float("nan"), dtype=torch.float32, device="npu")
    # The kernel owns cross-pipe and cross-core synchronization explicitly.
    options = dict(unit_flag=False, sync_solver=False)
    if compile_only:
        kernel = catlass_mmad_l0_expected.warmup(x, y, bt, out, grid=(1,), **constants(case), **options)
        return kernel, None, None
    kernel = catlass_mmad_l0_expected[(1,)](x, y, bt, out, **constants(case), **options)
    torch.npu.synchronize()
    expected = a.float() @ b.float()
    if case["WITH_BIAS"]:
        expected += bias
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
        test_ir_subviews()
        for with_bias in (False, True):
            for offset in (0, 256, None):
                for unit in (None, tla.UnitFlagMode.DISABLED):
                    test_ir_operands(with_bias, offset, unit)


if __name__ == "__main__":
    main()
