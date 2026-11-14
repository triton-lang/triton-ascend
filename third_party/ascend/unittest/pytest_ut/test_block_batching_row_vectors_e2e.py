# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import triton
import triton.language as tl
from block_batching_test_utils import batching_target  # noqa: F401


@triton.jit(do_not_specialize=['N', 'SX', 'SY', 'SO'])
def _row_relative_vector(X, Y, O, P, N, SX, SY, SO, SCALE, SINGLETON: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    if SINGLETON:
        col = col[None, :]
    mask = col < N
    x = tl.load(X + row * SX + col, mask, other=0).to(tl.float32)
    y = tl.load(Y + row * SY + col, mask, other=0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    tl.store(O + row * SO + col, x * sig * y * SCALE, mask)
    tl.store(P + row * SO + col, sig * (1.0 + x * (1.0 - sig)) * y * SCALE, mask)


@pytest.mark.parametrize('factor', [2, 3, 8])
@pytest.mark.parametrize('axis', [-1, 1])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_row_vector_and_singleton_expression_have_same_decision(factor, axis, dtype, monkeypatch):
    monkeypatch.setenv('TA_BLOCK_BATCHING_CORE_FLOOR', '0')
    hashes = {False: set(), True: set()}
    for n, padding in [(131, 13), (128, 21), (131, 19)]:
        rows, sx, sy, so = 5, n + padding, n + padding + 7, n + padding + 11
        gen = torch.Generator().manual_seed(810)
        xc = torch.randn((rows, sx), generator=gen).to(dtype)
        yc = torch.randn((rows, sy), generator=gen).to(dtype)
        x, y = xc.npu(), yc.npu()
        expected_x, expected_y = xc[:, :n].float(), yc[:, :n].float()
        sig = torch.sigmoid(expected_x)
        expected = [expected_x * sig * expected_y * .5, sig * (1 + expected_x * (1 - sig)) * expected_y * .5]
        pair_outputs = []
        for singleton in (False, True):
            outputs = []
            for f in (1, factor):
                allocations = [torch.full((rows * so + 32, ), -123., device='npu', dtype=dtype) for _ in range(2)]
                views = [v[16:-16].reshape(rows, so) for v in allocations]
                kernel = _row_relative_vector[(rows, triton.cdiv(n, 32))](x, y, *views, n, sx, sy, so, .5, singleton,
                                                                          32, block_batching_factor=f,
                                                                          block_batching_axis=axis, num_warps=4)
                torch.npu.synchronize()
                for out, backing, ref in zip(views, allocations, expected):
                    torch.testing.assert_close(out[:, :n].float().cpu(), ref,
                                               rtol=.02 if dtype == torch.bfloat16 else 2e-4,
                                               atol=.02 if dtype == torch.bfloat16 else 2e-5)
                    assert (out[:, n:].cpu() == -123).all()
                    assert (backing[:16].cpu() == -123).all() and (backing[-16:].cpu() == -123).all()
                if f != 1:
                    assert kernel.last_launch['applied_factor'] == factor, kernel.block_batching
                    assert kernel.last_launch['selected_axis'] == 1
                    assert all(p['rewritten'] and p['action'] == 'PointwiseRegion'
                               for p in kernel.block_batching['layout']['accesses'])
                    assert 'triton_indirect' not in kernel.candidate.asm['ttadapter']
                    assert 'linalg.transpose' not in kernel.candidate.asm['ttadapter']
                    hashes[singleton].add(kernel.candidate.hash)
                outputs.append([v.cpu() for v in views])
            for baseline, candidate in zip(*outputs):
                torch.testing.assert_close(baseline, candidate, rtol=0, atol=0)
            pair_outputs.append(outputs[1])
        for vector, singleton in zip(*pair_outputs):
            torch.testing.assert_close(vector, singleton, rtol=0, atol=0)
        torch.testing.assert_close(x.cpu(), xc, rtol=0, atol=0)
        torch.testing.assert_close(y.cpu(), yc, rtol=0, atol=0)
    assert all(len(v) == (2 if axis == -1 else 1) for v in hashes.values())
