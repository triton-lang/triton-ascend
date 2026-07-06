import os
import torch
import torch_npu

import triton
import triton.language as tl

os.environ["TRITON_DEVICE_PRINT"] = "1"


@triton.jit
def device_print_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    # device_print runs at runtime; the first argument must be a string prefix.
    # Set TRITON_DEVICE_PRINT=1 to see the output.
    tl.device_print("x =", x)


def test_device_print():
    BLOCK_SIZE = 8
    x = torch.randn(BLOCK_SIZE, device="npu", dtype=torch.float32)
    device_print_kernel[(1, )](x, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()


if __name__ == "__main__":
    test_device_print()
    print("test_device_print PASSED!")
