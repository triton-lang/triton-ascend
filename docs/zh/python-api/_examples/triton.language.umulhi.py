import triton
import triton.language as tl
import torch


def torch_umulhi(a, b):
    a_64 = a.to(torch.int64)
    b_64 = b.to(torch.int64)
    product_64 = a_64 * b_64
    # get the high part
    result_high_32 = product_64 >> 32
    return result_high_32.to(a.dtype)


@triton.jit
def umulhi_kernel(X, Y, Z, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    z = tl.umulhi(x, y)
    tl.store(Z + tl.arange(0, N), z)


def test_umulhi():
    N = 16
    x = torch.randint(low=0, high=2000, size=(N, ), dtype=torch.int32).npu()
    y = torch.randint(low=0, high=2000, size=(N, ), dtype=torch.int32).npu()

    triton_res = torch.empty_like(x)
    torch_res = torch_umulhi(x, y)

    umulhi_kernel[(1, )](x, y, triton_res, N=N)
    assert torch.equal(torch_res, triton_res)


if __name__ == '__main__':
    test_umulhi()
