import torch
import triton
import triton.language as tl


@triton.jit
def zeros_2d_kernel(output_ptr, XB: tl.constexpr, YB: tl.constexpr):
    """
    Generate a 2D zero tensor with shape (XB, YB) and store it to output_ptr in row-major order.
    """
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)

    ret = tl.zeros((XB, YB), dtype=tl.float32)

    oidx = xidx[:, None] * YB + yidx[None, :]

    tl.store(output_ptr + oidx, ret)


def test_zeros_2d():
    XB = 4
    YB = 8

    output = torch.zeros((XB, YB), dtype=torch.float32).npu()

    zeros_2d_kernel[(1, )](output, XB=XB, YB=YB)

    expected = torch.zeros((XB, YB), dtype=torch.float32)

    assert torch.allclose(output.cpu(), expected.cpu())


if __name__ == "__main__":
    test_zeros_2d()
