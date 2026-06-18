import pytest
import torch
import torch_npu
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def max_contiguous_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    # Declare that the first BLOCK_SIZE elements are contiguous
    x = tl.max_contiguous(x, [BLOCK_SIZE])
    tl.store(out_ptr + offsets, x * 2)


def test_max_contiguous():
    BLOCK_SIZE = 128
    x = torch.randn(BLOCK_SIZE, device="npu", dtype=torch.float32)
    out = torch.empty(BLOCK_SIZE, device="npu", dtype=torch.float32)

    max_contiguous_kernel[(1, )](x, out, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()

    # max_contiguous is only a compiler hint and does not change the result
    assert_close(out, x * 2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_max_contiguous()
    print("test_max_contiguous PASSED!")
