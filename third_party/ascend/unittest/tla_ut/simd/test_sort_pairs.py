"""Exercise both UB score/index pair sorting paths on Ascend 950."""

import pytest

import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import check_kernel_binary, compile_offline, lower_offline


@tla.jit
def _sort(scores, indices, dst, scratch, width: tl.constexpr, use_simd: tl.constexpr):
    if use_simd:
        stages: tl.constexpr = (width.bit_length() - 5) // 2
        if stages % 2 == 0:
            tla.simd.vbs(scores, indices, dst)
        else:
            tla.simd.vbs(scores, indices, scratch)
        for stage in tl.static_range(stages):
            if (stages - stage) % 2 == 0:
                tla.simd.vmrgsort4(dst, scratch, 32 * 4**stage)
            else:
                tla.simd.vmrgsort4(scratch, dst, 32 * 4**stage)
    else:
        tla.sort_pairs(scores, indices, dst, scratch)


@tla.jit
def sort_pairs_kernel(Scores, Indices, Out, WIDTH: tl.constexpr, USE_SIMD: tl.constexpr):
    with tla.scope.vector():
        scores = tla.alloc((WIDTH, ), tl.float32, space="ub")
        indices = tla.alloc((WIDTH, ), tl.uint32, space="ub")
        pairs = tla.alloc((2 * WIDTH, ), tl.uint32, space="ub")
        scratch = tla.alloc((2 * WIDTH, ), tl.uint32, space="ub")
        tla.load_gm_to_ub(tla.view(Scores, (WIDTH, )), scores)
        tla.load_gm_to_ub(tla.view(Indices, (WIDTH, )), indices)
        _sort(scores, indices, pairs, scratch, WIDTH, USE_SIMD)
        tla.store_ub_to_gm(pairs, tla.view(Out, (2 * WIDTH, )))


@pytest.mark.parametrize("use_simd", [False, True], ids=["template", "simd"])
def test_ir(use_simd, tmp_path):
    signature = {"Scores": "*fp32", "Indices": "*u32", "Out": "*u32"}
    module, ctx = compile_offline(sort_pairs_kernel, signature, {"WIDTH": 256, "USE_SIMD": use_simd},
                                  output_dir=tmp_path)
    ir = str(lower_offline(module, ctx, output_dir=tmp_path))
    if use_simd:
        assert ir.count("ave.hir.vbs") == 1
        assert ir.count("ave.hir.vmrgsort4") == 2
    else:
        assert ir.count("hivm.hir.vsort_pairs") == 1


@pytest.mark.parametrize("use_simd", [False, True], ids=["template", "simd"])
def test_compile(use_simd):
    import torch

    scores = torch.empty((256, ), device="npu", dtype=torch.float32)
    indices = torch.empty((256, ), device="npu", dtype=torch.int32)
    pairs = torch.empty((512, ), device="npu", dtype=torch.int32)
    result = sort_pairs_kernel.warmup(scores, triton.reinterpret(indices, tl.uint32),
                                      triton.reinterpret(pairs, tl.uint32), 256, use_simd, grid=(1, ))
    check_kernel_binary(result)


@pytest.mark.parametrize("use_simd", [False, True], ids=["template", "simd"])
def test_run(use_simd):
    import torch

    gen = torch.Generator().manual_seed(23)
    scores = torch.randint(-7, 8, (256, ), generator=gen).float()
    indices = torch.randperm(256, dtype=torch.int32, generator=gen)
    pairs = torch.empty((512, ), device="npu", dtype=torch.int32)
    sort_pairs_kernel[(1, )](scores.npu(), triton.reinterpret(indices.npu(), tl.uint32),
                             triton.reinterpret(pairs, tl.uint32), 256, use_simd)
    words = pairs.cpu().reshape(-1, 2)
    sorted_scores = words[:, 0].contiguous().view(torch.float32)
    sorted_indices = words[:, 1]
    torch.testing.assert_close(sorted_scores, scores.sort(descending=True).values)
    torch.testing.assert_close(sorted_indices.sort().values, indices.sort().values)
    torch.testing.assert_close(sorted_scores, scores[indices.argsort()[sorted_indices.long()]])
