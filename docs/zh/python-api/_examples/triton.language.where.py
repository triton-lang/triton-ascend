import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, XNUMEL: tl.constexpr,
            YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    tmp2 = X < Y
    ret = tl.where(tmp2, X, 1)
    tl.store(output_ptr + idx, ret)


def test_where():
    shape = (8, 16)
    dtype = 'float32'
    x = torch.randn(shape, dtype=eval(f'torch.{dtype}'), device="npu")
    y = torch.randn(shape, dtype=eval(f'torch.{dtype}'), device="npu")

    ans = torch.where(x < y, x, 1)
    output = torch.zeros_like(ans)
    fn_npu_[1, 1, shape[1]](output, x, y, 1, shape[0], 1, 1, shape[0], shape[1])
    torch.testing.assert_close(ans, output)


if __name__ == "main":
    test_where()
