import pytest
import torch
import torch_npu
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def debug_barrier_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Stage 1: load data
    x = tl.load(x_ptr + offsets)
    # Barrier: make sure all threads finished loading
    tl.debug_barrier()
    # Stage 2: compute
    y = x * 2
    # Barrier: make sure all threads finished computing
    tl.debug_barrier()
    # Stage 3: store the result
    tl.store(out_ptr + offsets, y)


def test_debug_barrier():
    BLOCK_SIZE = 128
    x = torch.randn(BLOCK_SIZE, device="npu", dtype=torch.float32)
    out = torch.empty(BLOCK_SIZE, device="npu", dtype=torch.float32)

    debug_barrier_kernel[(1, )](x, out, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()

    assert_close(out, x * 2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_debug_barrier()
    print("test_debug_barrier PASSED!")
