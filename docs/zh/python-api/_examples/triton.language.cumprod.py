import torch
import triton
import triton.language as tl


@triton.jit
def cumprod_2d_kernel(
    out_ptr0,
    in_ptr0,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    numel_x: tl.constexpr,
    numel_r: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    """
    Cumulatively product the input tensor (shape numel_x × numel_r) along the specified dimension,
    writing the result to out_ptr0.
    """
    tl.static_assert(numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel")
    tl.static_assert(numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel")
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


def test_cumprod_2d():
    X = 4
    R = 8
    dim = 0
    reverse = False

    x = torch.randn((X, R), dtype=torch.float32).npu()
    output = torch.zeros_like(x)

    cumprod_2d_kernel[(1, )](output, x, dim=dim, reverse=reverse, numel_x=X, numel_r=R, XBLOCK=X, RBLOCK=R)

    if reverse:
        expected = torch.flip(torch.cumprod(torch.flip(x.cpu(), [dim]), dim=dim), [dim])
    else:
        expected = torch.cumprod(x.cpu(), dim=dim)

    assert torch.allclose(output.cpu(), expected.cpu())


if __name__ == "__main__":
    test_cumprod_2d()
