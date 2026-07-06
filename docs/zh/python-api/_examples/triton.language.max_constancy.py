import pytest
import torch
import torch_npu
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def max_constancy_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    # Declare the constancy pattern: every 4 consecutive values are equal
    x = tl.max_constancy(x, [4])
    tl.store(out_ptr + offsets, x * 2)


def test_max_constancy():
    BLOCK_SIZE = 128
    x = torch.randn(BLOCK_SIZE, device="npu", dtype=torch.float32)
    out = torch.empty(BLOCK_SIZE, device="npu", dtype=torch.float32)

    max_constancy_kernel[(1, )](x, out, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()

    # max_constancy is only a compiler hint and does not change the result
    assert_close(out, x * 2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_max_constancy()
    print("test_max_constancy PASSED!")
