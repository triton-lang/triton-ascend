import triton
import triton.language as tl
import torch
import math


@triton.jit
def kernel_randint4x(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, 4)
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(0, block_size, step=4):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint4x(5, 10 + global_offset, n_rounds)  # Generate random numbers for each index
        mask = (global_offset + indices) < N
        tl.store(x_ptr + global_offset + indices, rand_vals, mask)  # Store the random numbers


def test_randint4x():
    shape = (1, 3)

    y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()
    numel = y_cali.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)
    kernel_randint4x[ncore, 1, 1](y_cali, 10, numel, xblock)


if __name__ == "__main__":
    test_randint4x()
