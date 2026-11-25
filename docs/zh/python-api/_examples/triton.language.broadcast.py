import torch
import triton
import triton.language as tl


@triton.jit
def broadcast_kernel(output_ptr, BLOCK_SIZE: tl.constexpr):
    # Broadcast the [1] block and the [BLOCK_SIZE] block to a common shape.
    fill_value = tl.full([1], 5.0, dtype=tl.float32)
    offsets = tl.arange(0, BLOCK_SIZE)
    broadcasted_fill, broadcasted_offsets = tl.broadcast(fill_value, offsets)
    result = broadcasted_offsets + broadcasted_fill
    tl.store(output_ptr + offsets, result)


def test_broadcast():
    BLOCK = 128
    out = torch.empty(BLOCK, dtype=torch.float32, device="npu")
    broadcast_kernel[(1, )](out, BLOCK_SIZE=BLOCK)
    ref = torch.arange(BLOCK, dtype=torch.float32) + 5.0
    torch.testing.assert_close(out.cpu(), ref)


if __name__ == "__main__":
    test_broadcast()
    print("test_broadcast passed!")
