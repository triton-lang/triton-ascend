import triton
import triton.language as tl
import torch
import math


@triton.jit
def kernel_randint4x(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(0, block_size, step=4):
        global_offset = block_offset + inner_idx
        # randint4x returns a tuple of 4 tensors, unpack them
        r0, r1, r2, r3 = tl.randint4x(5, 10 + global_offset, n_rounds)
        mask0 = (global_offset + 0) < N
        mask1 = (global_offset + 1) < N
        mask2 = (global_offset + 2) < N
        mask3 = (global_offset + 3) < N
        tl.store(x_ptr + global_offset + 0, r0, mask=mask0)
        tl.store(x_ptr + global_offset + 1, r1, mask=mask1)
        tl.store(x_ptr + global_offset + 2, r2, mask=mask2)
        tl.store(x_ptr + global_offset + 3, r3, mask=mask3)


def test_randint4x():
    shape = (32, )
    y = torch.zeros(shape, dtype=torch.int32, device='npu')
    numel = y.numel()
    ncore = 1 if numel < 32 else 2
    xblock = math.ceil(numel / ncore)
    kernel_randint4x[ncore, 1, 1](y, 10, numel, xblock)
    # Verify that random values were written (non-zero in at least some positions)
    assert y.min().item() != y.max().item(), "Expected random values to vary"


if __name__ == "__main__":
    test_randint4x()
