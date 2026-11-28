import torch
import triton
import triton.language as tl


@triton.jit
def full_2d_kernel(output_ptr, XB: tl.constexpr, YB: tl.constexpr, VALUE: tl.constexpr):
    """
    Generate a 2D tensor of shape (XB, YB) with all elements set to VALUE,
    and store it in row-major order at output_ptr.
    """
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)

    ret = tl.full((XB, YB), value=VALUE, dtype=tl.float32)

    oidx = xidx[:, None] * YB + yidx[None, :]

    tl.store(output_ptr + oidx, ret)


def test_full_2d():
    XB = 4
    YB = 8
    VALUE = 100.0

    out = torch.zeros((XB, YB), dtype=torch.float32).npu()

    full_2d_kernel[(1, )](out, XB=XB, YB=YB, VALUE=VALUE)

    ref = torch.full((XB, YB), VALUE, dtype=torch.float32)

    torch.testing.assert_close(out.cpu(), ref.cpu())


if __name__ == "__main__":
    test_full_2d()
    print("test_full_2d passed!")
