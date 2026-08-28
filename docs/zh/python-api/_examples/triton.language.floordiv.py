import triton
import triton.language as tl
import torch


def torch_floordiv(x0, x1):
    res = x0 // x1
    return res


@triton.jit
def triton_kernel(out_ptr0, in_ptr0, in_ptr1, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(in_ptr0 + idx)
    y = tl.load(in_ptr1 + idx)
    ret = x // y
    tl.store(out_ptr0 + idx, ret)


def test_floordiv():
    param_list = ['int32', (2, 256, 2), 2]
    dtype, shape, ncore = param_list
    x0 = torch.randint(1, 10, size=shape, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.randint(1, 10, size=shape, dtype=eval('torch.' + dtype)).npu()

    torch_res = torch_floordiv(x0, x1)
    triton_res = torch.empty_like(x0)
    triton_kernel[ncore, 1, 1](triton_res, x0, x1, N=x0.numel())

    torch.testing.assert_close(torch_res, triton_res, rtol=0, atol=0, equal_nan=True)


if __name__ == '__main__':
    test_floordiv()
