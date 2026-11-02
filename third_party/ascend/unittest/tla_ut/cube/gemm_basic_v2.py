"""Single-tile fp16 GEMM using explicit TLA DMA, with three-stage validation.

C[16, 64] = A[16, 32] @ B[32, 64], accumulated and stored as fp32.
On NPU, running this file checks IR, compilation and numerical accuracy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import get_stages, compile_offline, lower_offline, check_kernel_binary, matrix

SHAPE = dict(M=16, N=64, K=32)
SIGNATURE = {"A": "*fp16", "B": "*fp16", "C": "*fp32"}
CONFIGS = [
    dict(name="default", unit=None, accumulate=False),
    dict(name="unit_disabled", unit=tla.UnitFlagMode.DISABLED, accumulate=False),
    dict(name="unit_update", unit=tla.UnitFlagMode.ENABLED_WITH_UPDATE, accumulate=False),
    dict(name="unit_accumulate", unit=tla.UnitFlagMode.ENABLED_WITH_UPDATE, accumulate=True),
]


def constants(case):
    return dict(SHAPE, UNIT=case["unit"], GROUP=None if case["unit"] is None else 0, ACCUMULATE=case["accumulate"])


@tla.jit
def gemm_basic_v2(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, UNIT: tl.constexpr, GROUP: tl.constexpr,
                  ACCUMULATE: tl.constexpr):
    tl.static_assert(M % 16 == 0 and N % 16 == 0 and K % 16 == 0)
    with tla.scope.cube():
        a_l1 = tla.alloc((K // 16, M // 16, 16, 16), tl.float16, space="l1")
        b_l1 = tla.alloc((N // 16, K // 16, 16, 16), tl.float16, space="l1")
        acc = tla.alloc((N // 16, M // 16, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(A, (M, K)), a_l1)
        tla.load_gm_to_l1(tla.view(B, (K, N)), b_l1)
        if ACCUMULATE:
            tla.cube.mmad(a_l1, b_l1, acc, clear_accum=True, unit_flag_ctrl=tla.UnitFlagMode.ENABLED_WITHOUT_UPDATE,
                          unit_flag_group_id=GROUP)
        tla.cube.mmad(a_l1, b_l1, acc, clear_accum=not ACCUMULATE, unit_flag_ctrl=UNIT, unit_flag_group_id=GROUP)
        tla.store_l0c_to_gm(
            acc, tla.view(C, (M, N)),
            tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND, unit_flag_ctrl=UNIT, unit_flag_group_id=GROUP))


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    module, ctx = compile_offline(gemm_basic_v2, SIGNATURE, constants(case), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    for op, count in (("nd2nz", 2), ("mmadL1", 2 if case["accumulate"] else 1), ("fixpipe", 1)):
        assert text.count("hivm.hir." + op + " ") == count, op
    if case["unit"] is None:
        assert "unit_flag_mode" not in text and "unit_flag_group_id" not in text
    else:
        count = 3 if case["accumulate"] else 2
        assert text.count("unit_flag_group_id = 0") == count
        assert text.count("unit_flag<" + case["unit"].name + ">") == 2
        if case["accumulate"]:
            assert text.count("unit_flag<ENABLED_WITHOUT_UPDATE>") == 1
    assert "dma_mode = #hivm.dma_mode<nz2nd>" in text
    assert text.count("memref.alloc()") == 3
    assert "#hivm.address_space<cbuf>" in text and "#hivm.address_space<cc>" in text
    assert "#hivm.tcore_type<CUBE>" in text
    assert 'mix_mode = "mix"' in text
    assert "hivm.vector_function" not in text
    assert "!tt.ptr" not in text and "tt.store" not in text
    assert "tensor<" not in text and "bufferization." not in text
    print("IR PASS: gemm_basic_v2", case["name"])


def _launch(case, seed=7, compile_only=False):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn((SHAPE["M"], SHAPE["K"]), generator=generator, dtype=torch.float16)
    b = torch.randn((SHAPE["K"], SHAPE["N"]), generator=generator, dtype=torch.float16)
    x, y = a.npu(), b.npu()
    out = torch.full((SHAPE["M"], SHAPE["N"]), float("nan"), dtype=torch.float32, device="npu")
    # Automatic UnitFlag inference rewrites explicit modes. Disable only that
    # optimization when exercising manual flags; ordinary pipe sync stays on.
    options = {} if case["unit"] is None else {"unit_flag": False}
    if compile_only:
        kernel = gemm_basic_v2.warmup(x, y, out, grid=(1, ), **constants(case), **options)
        return kernel, None, None
    kernel = gemm_basic_v2[(1, )](x, y, out, **constants(case), **options)
    torch.npu.synchronize()
    expected = a.float() @ b.float()
    return kernel, out.cpu(), expected * (2 if case["accumulate"] else 1)


@matrix(CONFIGS)
def test_compile(case):
    kernel, _, _ = _launch(case, compile_only=True)
    assert kernel.metadata.mix_mode == "mix", "Cube kernel must use the Cube-capable launcher"
    check_kernel_binary(kernel)


@matrix(CONFIGS)
def test_run(case):
    for seed in (7, 19):
        _, actual, expected = _launch(case, seed)
        assert torch.isfinite(actual).all(), "GEMM left unwritten or non-finite output"
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=3e-2)
        print("ACCURACY PASS: gemm_basic_v2", case["name"], SHAPE, "seed", seed, "max_abs_error",
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


if __name__ == "__main__":
    main()
