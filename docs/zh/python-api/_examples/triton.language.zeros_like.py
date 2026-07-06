import torch
import triton
import triton.language as tl


@triton.jit
def zeros_like_2d_kernel(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr):
    """
    Load the input tensor X, generate a zero tensor with the same shape, and store it to output_ptr.
    """
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)

    idx = xidx[:, None] * YB + yidx[None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.zeros_like(X)

    tl.store(output_ptr + idx, ret)


def test_zeros_like_2d():
    XB = 4
    YB = 8

    x = torch.randn((XB, YB), dtype=torch.float32).npu()
    output = torch.zeros_like(x)

    zeros_like_2d_kernel[(1, )](output, x, XB=XB, YB=YB)

    expected = torch.zeros_like(x.cpu())

    assert torch.allclose(output.cpu(), expected.cpu())


if __name__ == "__main__":
    test_zeros_like_2d()
