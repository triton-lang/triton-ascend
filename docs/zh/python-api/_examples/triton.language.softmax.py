import triton
import triton.language as tl
import torch
import math


def torch_softmax_d0(x1):
    res = torch.softmax(x1, axis=0).to(x1.dtype)
    return res


@triton.jit
def triton_softmax_2d(in_ptr, out_ptr, ynumel: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    a = tl.load(in_ptr + idx)
    ret = tl.softmax(a)

    tl.store(out_ptr + idx, ret)


def test_softmax_2d():
    shape = (16, 32)
    xnumel, ynumel = shape
    XB, YB = xnumel, ynumel
    x0 = torch.randn(size=shape, dtype=torch.float32).npu()

    torch_res = torch_softmax_d0(x0)
    triton_res = torch.empty_like(x0)
    triton_softmax_2d[1, 1, 1](x0, triton_res, ynumel, XB, YB)

    torch.testing.assert_close(torch_res, triton_res, rtol=1e-04, atol=1e-04, equal_nan=True)


if __name__ == '__main__':
    test_softmax_2d()
