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
        tmp1 = libdevice.tan(tmp0)
        tl.store(output + (x0), tmp1, mask=mask)


def test_tan():
    param_list = [(2, 256, 4), 2, 2048, 1024]
    shape, ncore, xblock, xblock_sub = param_list
    x0 = torch.randn(size=shape, dtype=torch.float32).npu() * 0.5

    torch_res = torch.tan(x0)
    triton_res = torch.empty_like(x0)
    triton_kernel[ncore, 1, 1](x0, triton_res, x0.numel(), xblock, xblock_sub)

    torch.testing.assert_close(torch_res, triton_res, rtol=1e-03, atol=1e-03, equal_nan=True)


if __name__ == "__main__":
    test_tan()
