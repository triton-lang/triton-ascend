import torch
import triton
import triton.language as tl


def combine_fn_test(a, b):
    return a + b


@triton.jit
def associative_scan_2d_kernel(
    out_ptr0,
    in_ptr0,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    numel_x: tl.constexpr,
    numel_r: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    tl.static_assert(numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel")
    tl.static_assert(numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel")
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=combine_fn_test)
    tl.store(out_ptr0 + idx, ret)


def test_associative_scan_2d():
    X = 4
    R = 8
    dim = 0
    reverse = False

    x = torch.randn((X, R), dtype=torch.float32).npu()
    output = torch.zeros_like(x)

    associative_scan_2d_kernel[(1, )](output, x, dim=dim, reverse=reverse, numel_x=X, numel_r=R, XBLOCK=X, RBLOCK=R)

    expected = torch.cumsum(x.cpu(), dim=dim)

    assert torch.allclose(output.cpu(), expected.cpu())


if __name__ == "__main__":
    test_associative_scan_2d()
