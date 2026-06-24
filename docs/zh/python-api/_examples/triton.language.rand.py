import triton
import triton.language as tl
import torch
import math


@triton.jit
def kernel_rand(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.rand(5, 10 + global_offset, n_rounds)  # Generate a random number for each index
        tl.store(x_ptr + global_offset, rand_vals)  # Store the random number


def test_rand():
    shape = (1, 3)

    y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    numel = y_calf.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)
    kernel_rand[ncore, 1, 1](y_calf, 10, numel, xblock)


if __name__ == "__main__":
    test_rand()
