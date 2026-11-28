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

    out = torch.ones((XB, YB), dtype=torch.float32).npu()

    zeros_2d_kernel[(1, )](out, XB=XB, YB=YB)

    ref = torch.zeros((XB, YB), dtype=torch.float32)

    torch.testing.assert_close(out.cpu(), ref.cpu())


if __name__ == "__main__":
    test_zeros_2d()
    print("test_zeros_2d passed!")
