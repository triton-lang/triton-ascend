import torch
import triton
import triton.language as tl


@triton.jit
def cat_1d_kernel(output_ptr, x_ptr, y_ptr, XB: tl.constexpr):
    """
    Concatenate two 1D tensors x and y of length XB into a continuous tensor of length XB*2.
    """
    idx = tl.arange(0, XB)
    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = tl.cat(X, Y, can_reorder=True)

    oidx = tl.arange(0, XB * 2)
    tl.store(output_ptr + oidx, ret)


def test_cat_1d():
    XB = 8

    x = torch.randn(XB, dtype=torch.float32).npu()
    y = torch.randn(XB, dtype=torch.float32).npu()
    output = torch.zeros(XB * 2, dtype=torch.float32).npu()

    cat_1d_kernel[(1, )](output, x, y, XB=XB)

    expected = torch.cat([x.cpu(), y.cpu()], dim=0)

    assert torch.allclose(output.cpu(), expected.cpu())


if __name__ == "__main__":
    test_cat_1d()
