import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch


@triton.jit
def triton_kernel(input, output, n_elements, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        mask = x0 < n_elements
        tmp0 = tl.load(input + (x0), mask=mask)
        tmp1 = libdevice.nearbyint(tmp0)
        tl.store(output + (x0), tmp1, mask=mask)


if __name__ == "__main__":
    dtype, shape, ncore, xblock, xblock_sub = ['float32', (128, 4096), 512, 1024, 1024]
    input = torch.randn(shape, dtype=eval('torch.' + dtype)).npu()
    output = torch.zeros_like(input)
    triton_kernel[ncore, 1, 1](input, output, xblock, xblock_sub)
