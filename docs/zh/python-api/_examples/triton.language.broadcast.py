import torch
import triton
import triton.language as tl


@triton.jit
def broadcast_kernel(output_ptr, BLOCK_SIZE: tl.constexpr):
    # Broadcast the scalar to the same shape as the vector using broadcast_to.
    scalar = tl.full([1], 5.0, dtype=tl.float32)
    broadcasted_scalar = tl.broadcast_to(scalar, (BLOCK_SIZE, ))
    offsets = tl.arange(0, BLOCK_SIZE)
    result = tl.arange(0, BLOCK_SIZE) + broadcasted_scalar
    tl.store(output_ptr + offsets, result)


def test_broadcast():
    BLOCK = 128
    out = torch.empty(BLOCK, dtype=torch.float32, device="npu")
    broadcast_kernel[(1, )](out, BLOCK_SIZE=BLOCK)
    ref = torch.arange(BLOCK, dtype=torch.float32) + 5.0
    torch.testing.assert_close(out.cpu(), ref)


if __name__ == "__main__":
    test_broadcast()
