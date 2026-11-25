import torch
import triton
import triton.language as tl


@triton.jit
def cumsum_2d_kernel(
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
    Cumulatively sum the input tensor (shape numel_x × numel_r) along the specified dimension,
    writing the result to out_ptr0.
    """
    tl.static_assert(numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel")
    tl.static_assert(numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel")
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


def test_cumsum_2d():
    X = 4
    R = 8
    dim = 0
    reverse = False

    x = torch.randn((X, R), dtype=torch.float32).npu()
    out = torch.zeros_like(x)

    cumsum_2d_kernel[(1, )](out, x, dim=dim, reverse=reverse, numel_x=X, numel_r=R, XBLOCK=X, RBLOCK=R)

    if reverse:
        ref = torch.flip(torch.cumsum(torch.flip(x.cpu(), [dim]), dim=dim), [dim])
    else:
        ref = torch.cumsum(x.cpu(), dim=dim)

    torch.testing.assert_close(out.cpu(), ref.cpu())


if __name__ == "__main__":
    test_cumsum_2d()
    print("test_cumsum_2d passed!")
