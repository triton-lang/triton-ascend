import triton
import torch
import triton.language as tl


@triton.jit
def sort_kernel_1d(X, Z, M: tl.constexpr, descending: tl.constexpr):
    off = tl.arange(0, M)
    x = tl.load(X + off)
    x = tl.sort(x, descending=descending, dim=0)
    tl.store(Z + off, x)


def test_sort_1d():
    shape = (2, )
    descending = True
    x = torch.randint(size=shape, low=0, high=2000, dtype=torch.int16).npu()
    out = torch.empty_like(x)

    ref = torch.sort(x, descending=descending)[0]
    M = x.shape[0]
    sort_kernel_1d[(1, )](x, out, M, descending)
    torch.testing.assert_close(out, ref)


if __name__ == '__main__':
    test_sort_1d()
    print("test_sort_1d passed!")
