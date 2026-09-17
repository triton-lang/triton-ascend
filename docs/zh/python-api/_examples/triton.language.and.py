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
    ret = X & Y

    tl.store(output_ptr + idx, ret)


def test_and():
    M, N = 8, 16
    dtype = 'int32'
    x = torch.randint(low=0, high=256, size=(M, N), dtype=eval(f'torch.{dtype}')).npu()
    y = torch.randint(low=0, high=256, size=(M, N), dtype=eval(f'torch.{dtype}')).npu()
    ans = x & y
    output = torch.zeros_like(ans)
    fn_npu_[1, 1, N](output, x, y, 1, M, 1, 1, M, N)
    torch.testing.assert_close(ans, output)


if __name__ == "main":
    test_and()
