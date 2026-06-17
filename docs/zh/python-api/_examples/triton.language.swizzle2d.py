import pytest
import triton
import triton.language as tl
import torch
import torch_npu


@triton.jit
def fn_npu_(output_ptr, x_ptr):
    i = tl.arange(0, 4)
    j = tl.arange(0, 4)
    xx, yy = tl.swizzle2d(i, j, size_i=4, size_j=4, size_g=2)

    tl.store(output_ptr + tl.arange(0, 4), xx)
    tl.store(x_ptr + tl.arange(0, 4), yy)


def test_swizzle2d():
    shape = (2, 256, 16)
    dtype = 'int32'
    ncore = 1
    x = torch.randint(1, (4, ), dtype=eval('torch.' + dtype)).npu()
    a = torch.tensor([[0, 2, 1, 3], [4, 6, 5, 7], [8, 10, 9, 11], [12, 14, 13, 15]], dtype=eval('torch.' + dtype)).npu()
    output = torch.randint(1, (4, ), dtype=eval('torch.' + dtype)).npu()
    fn_npu_[ncore, 1, 1](output, x)
    triton_ret = output[:, None] * 4 + x[None, :]
    torch.testing.assert_close(triton_ret, a)


if __name__ == "main":
    test_swizzle2d()
