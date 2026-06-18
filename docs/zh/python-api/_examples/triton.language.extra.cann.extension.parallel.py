import torch
import torch_npu
from torch.testing import assert_close

import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension


@triton.jit
def parallel_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    # Load the full [M, N] block
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    block = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    # Split M across 2 vector sub-blocks; iterations are independent (no cross-iteration dependency)
    SUB_M: tl.constexpr = M // 2
    for s in extension.parallel(0, 2, bind_sub_block=True):
        sub = extension.extract_slice(block, (s * SUB_M, 0), (SUB_M, N), (1, 1))
        sub = sub * 2.0  # a compute instruction on the sub-block
        offs_sub_m = s * SUB_M + tl.arange(0, SUB_M)
        out_ptrs = out_ptr + offs_sub_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, sub)


def test_parallel():
    M = 128
    N = 128
    x = torch.randn((M, N), device="npu", dtype=torch.float32)
    out = torch.empty((M, N), device="npu", dtype=torch.float32)
    parallel_kernel[(1, )](x, out, M=M, N=N)
    torch.npu.synchronize()
    assert_close(out, x * 2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_parallel()
    print("test_parallel PASSED!")
