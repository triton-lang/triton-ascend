import pytest
import torch
import torch_npu
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def static_range_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    # static_range unrolls the loop at compile time (BLOCK_SIZE must be constexpr)
    for i in tl.static_range(BLOCK_SIZE):
        x = tl.load(x_ptr + i)
        tl.store(out_ptr + i, x * x)


def test_static_range():
    BLOCK_SIZE = 128
    x = torch.randn(BLOCK_SIZE, device="npu", dtype=torch.float32)
    out = torch.empty(BLOCK_SIZE, device="npu", dtype=torch.float32)

    static_range_kernel[(1, )](x, out, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()

    assert_close(out, x * x, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_static_range()
    print("test_static_range PASSED!")
