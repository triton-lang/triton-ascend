import torch
import torch_npu

import triton
import triton.language as tl


@triton.jit
def device_assert_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    # device_assert checks a condition at runtime on device.
    # Requires both TRITON_DEBUG=1 and TRITON_DEVICE_PRINT=1 to take effect.
    tl.device_assert(x == x, "x contains NaN")


def test_device_assert():
    BLOCK_SIZE = 8
    x = torch.randn(BLOCK_SIZE, device="npu", dtype=torch.float32)
    device_assert_kernel[(1, )](x, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()


if __name__ == "__main__":
    test_device_assert()
    print("test_device_assert PASSED!")
