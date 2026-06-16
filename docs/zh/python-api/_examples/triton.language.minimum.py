import triton
import triton.language as tl
import torch
import pytest


def torch_minimum(x0, x1):
    res = torch.minimum(x0, x1)
    return res


@triton.jit
def triton_minimum(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp1 = tl.load(in_ptr1 + x_index, xmask)
        tmp2 = tl.minimum(tmp0, tmp1)
        tl.store(out_ptr0 + x_index, tmp2, xmask)


def test_minimum(param_list):
    param_list = ['float32', (2, 4096, 8), 2, 32768, 1024]
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = torch.randn(size=shape, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.randn(size=shape, dtype=eval('torch.' + dtype)).npu()

    torch_res = torch_minimum(x0, x1)
    triton_res = torch.empty_like(x0)
    triton_minimum[ncore, 1, 1](x0, x1, triton_res, x0.numel(), xblock, xblock_sub)

    torch.testing.assert_close(torch_res, triton_res, rtol=1e-04, atol=1e-04, equal_nan=True)


if __name__ == '__main__':
    test_minimum()
