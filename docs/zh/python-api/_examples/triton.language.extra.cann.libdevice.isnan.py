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
        tmp1 = libdevice.isnan(tmp0)
        tl.store(output + (x0), tmp1, mask=mask)


def test_isnan():
    param_list = [(2, 256, 4), 2, 2048, 1024]
    shape, ncore, xblock, xblock_sub = param_list
    x0 = torch.randn(size=shape, dtype=torch.float32)
    x0.view(-1)[0] = float("nan")
    x0.view(-1)[1] = float("inf")
    x0 = x0.npu()

    torch_res = torch.isnan(x0).to(torch.int32)
    triton_res = torch.empty(shape, dtype=torch.int32, device='npu')
    triton_kernel[ncore, 1, 1](x0, triton_res, x0.numel(), xblock, xblock_sub)

    torch.testing.assert_close(torch_res.cpu(), triton_res.cpu(), rtol=0, atol=0)


if __name__ == "__main__":
    test_isnan()
