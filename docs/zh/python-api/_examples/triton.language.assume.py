import pytest
import torch
import torch_npu
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def assume_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    # Assume BLOCK_SIZE is a power of 2 so the compiler can optimize the division
    tl.assume((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0)
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    out = x // BLOCK_SIZE + y % BLOCK_SIZE
    tl.store(out_ptr + offsets, out)


def test_assume():
    BLOCK_SIZE = 128
    x = torch.randint(0, 1000, (BLOCK_SIZE, ), device="npu", dtype=torch.int32)
    y = torch.randint(0, 1000, (BLOCK_SIZE, ), device="npu", dtype=torch.int32)
    out = torch.empty(BLOCK_SIZE, device="npu", dtype=torch.int32)

    assume_kernel[(1, )](x, y, out, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()

    ref = x // BLOCK_SIZE + y % BLOCK_SIZE
    assert_close(out, ref)


if __name__ == "__main__":
    test_assume()
    print("test_assume PASSED!")
