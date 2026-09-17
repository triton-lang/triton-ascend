import triton
import triton.language as tl
import torch
import math


@triton.jit
def kernel_randn(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    offsets = block_offset + tl.arange(0, XBLOCK)  # Block-level offset tensor
    mask = offsets < N
    rand_vals = tl.randn(5, 10 + offsets, n_rounds)  # Generate a block of random numbers at once
    tl.store(x_ptr + offsets, rand_vals, mask=mask)


def test_randn():
    shape = (1, 3)

    y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    numel = y_calf.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)
    kernel_randn[ncore, 1, 1](y_calf, 10, numel, xblock)


if __name__ == "__main__":
    test_randn()
