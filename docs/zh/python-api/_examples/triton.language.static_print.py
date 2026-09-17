import torch
import torch_npu

import triton
import triton.language as tl


@triton.jit
def static_print_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    tl.static_print("BLOCK_SIZE =", BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, x)


def test_static_print():
    BLOCK_SIZE = 128
    x = torch.randn(BLOCK_SIZE, device="npu", dtype=torch.float32)
    out = torch.empty(BLOCK_SIZE, device="npu", dtype=torch.float32)
    static_print_kernel[(1, )](x, out, BLOCK_SIZE=BLOCK_SIZE)
    torch.npu.synchronize()


if __name__ == "__main__":
    test_static_print()
    print("test_static_print PASSED!")
