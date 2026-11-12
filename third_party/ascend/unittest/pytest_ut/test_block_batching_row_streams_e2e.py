"""Real row-stream kernels, including tails, zero trips and untouched padding."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N', 'SX', 'SO'])
def row_stream(X, O, N, SX, SO, B: tl.constexpr, MODE: tl.constexpr):
    row = tl.program_id(0)
    for start in range(0, N, B):
        column = start + tl.arange(0, B)
        value = tl.load(X + row * SX + column, column < N, other=0).to(tl.float32)
        if MODE == 0:
            result = -value
        elif MODE == 1:
            result = value * .5 + .25
        else:
            result = tl.exp(value * .1)
        tl.store(O + row * SO + column, result, column < N)


@pytest.mark.parametrize('factor,n,mode,dtype', [
    (2, 0, 0, torch.float32),
    (2, 1, 0, torch.float32),
    (2, 127, 0, torch.float32),
    (2, 128, 1, torch.float32),
    (2, 129, 1, torch.float16),
    (3, 513, 2, torch.float32),
    (6, 255, 0, torch.float32),
    (2, 8193, 2, torch.float32),
])
def test_unbridged_uniform_loop_uses_identical_baseline(factor, n, mode, dtype):
    rows = factor * 64 + 1
    sx, so = n + 11, n + 23
    source = torch.randn((rows, sx), generator=torch.Generator().manual_seed(881)).to(dtype)
    if mode == 0 and n > 8:
        source[:, :8] = torch.tensor([-0., 0., -float('inf'), float('inf'), float('nan'), 1e-5, -1e5, 1.], dtype=dtype)
    x = source.npu()
    allocations = [torch.full((rows * so + 32, ), -123., dtype=dtype, device='npu') for _ in range(2)]
    outputs = [a[16:-16].reshape(rows, so) for a in allocations]
    kernels = [
        row_stream[(rows, )](x, out, n, sx, so, 128, mode, block_batching_factor=f, block_batching_axis=0, num_warps=4)
        for f, out in zip((1, factor), outputs)
    ]
    torch.npu.synchronize()
    value = source[:, :n].float()
    expected = (-value if mode == 0 else value * .5 + .25 if mode == 1 else torch.exp(value * .1)).to(dtype)
    for out, allocation in zip(outputs, allocations):
        actual = out.cpu()
        torch.testing.assert_close(actual[:, :n], expected, rtol=2e-4 if dtype == torch.float32 else 2e-3,
                                   atol=2e-5 if dtype == torch.float32 else 2e-3, equal_nan=True)
        assert (actual[:, n:] == -123.).all()
        assert (allocation[:16].cpu() == -123.).all()
        assert (allocation[-16:].cpu() == -123.).all()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0, equal_nan=True)
    zeros = torch.isfinite(expected) & (expected == 0)
    assert torch.equal(torch.signbit(outputs[0].cpu()[:, :n][zeros]), torch.signbit(outputs[1].cpu()[:, :n][zeros]))
    torch.testing.assert_close(x.cpu(), source, rtol=0, atol=0, equal_nan=True)
    bundle = kernels[1]
    assert bundle.last_launch['applied_factor'] == 1
    assert bundle.last_launch['reason'] == 'LOOP_POINTWISE_MEMORY_VIEW_REQUIRED'
    assert bundle.candidate is None
    assert kernels[0].kernel == bundle.baseline.kernel
