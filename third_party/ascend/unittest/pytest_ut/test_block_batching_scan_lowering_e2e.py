"""Independent scan rows, reverse order and partial batch/column tails."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def scan_rows(X, Y, N: tl.constexpr, B: tl.constexpr, REVERSE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    value = tl.load(X + row * N + col, col < N, other=0)
    result = tl.cumsum(value, 0, reverse=REVERSE)
    tl.store(Y + row * N + col, result, col < N)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("rows,columns", [(128, 512), (129, 255)])
def test_scan_rows_match_reference_and_preserve_tail(dtype, reverse, rows, columns):
    torch.manual_seed(881)
    source = torch.randint(-4, 5, (rows, columns), dtype=torch.int32).to(dtype)
    x = source.npu()
    outputs = [torch.full((rows + 1, columns), -12345, dtype=dtype, device="npu") for _ in range(2)]
    kernels = [
        scan_rows[(rows, )](x, out, columns, triton.next_power_of_2(columns), reverse, block_batching_factor=factor,
                            num_warps=4, num_stages=1) for factor, out in zip((1, 2), outputs)
    ]
    torch.npu.synchronize()
    expected = source.flip(1) if reverse else source
    expected = expected.cumsum(1, dtype=dtype)
    if reverse:
        expected = expected.flip(1)
    for out in outputs:
        torch.testing.assert_close(out[:rows].cpu(), expected, rtol=0, atol=0)
        assert torch.all(out[rows].cpu() == -12345)
    assert kernels[1].last_launch["applied_factor"] == 2
    assert kernels[1].metadata.block_batching["lane_scans_lowered"] == 1
    assert kernels[0].kernel == kernels[1].baseline.kernel
