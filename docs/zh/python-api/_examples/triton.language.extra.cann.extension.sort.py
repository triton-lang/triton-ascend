import triton
import torch
import triton.language as tl
import triton.language.extra.cann.extension as extension


@triton.jit
def sort_kernel_2d(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    offx = tl.arange(0, M)
    offy = tl.arange(0, N) * M
    off2d = offx[None, :] + offy[:, None]
    x = tl.load(X + off2d)
    x = extension.sort(x, descending=descending, dim=1)
    tl.store(Z + off2d, x)


def test_sort_2d():
    shape = (8, 16)
    descending = True
    x = torch.randn(size=shape, dtype=torch.float32).npu()
    out = torch.empty_like(x)

    ref = torch.sort(x, descending=descending)[0]
    N = x.shape[0]
    M = x.shape[1]
    sort_kernel_2d[(1, )](x, out, N, M, descending)
    torch.testing.assert_close(out, ref)


if __name__ == '__main__':
    test_sort_2d()
    print("test_sort_2d passed!")
