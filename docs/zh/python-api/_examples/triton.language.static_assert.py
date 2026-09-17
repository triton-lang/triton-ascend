import torch
import torch_npu

import triton
import triton.language as tl


@triton.jit
def static_assert_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    # static_assert is checked at compile time; cond must be a constexpr
    tl.static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be a power of 2")
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, x)


def test_static_assert():
    BLOCK_SIZE = 128  # power of 2, so the static_assert passes
    x = torch.randn(BLOCK_SIZE, device="npu", dtype=torch.float32)
    out = torch.empty(BLOCK_SIZE, device="npu", dtype=torch.float32)
    static_assert_kernel[(1, )](x, out, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()


if __name__ == "__main__":
    test_static_assert()
    print("test_static_assert PASSED!")
