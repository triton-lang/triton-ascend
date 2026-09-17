import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def tt_sum_2d(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, dim: tl.constexpr):
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = mblk_idx[:, None] * N + nblk_idx[None, :]
    x = tl.load(in_ptr + idx)
    ret = tl.sum(x, dim)
    if dim == 0:
        tl.store(out_ptr + tl.arange(0, N), ret)
    else:
        tl.store(out_ptr + tl.arange(0, M), ret)


def test_sum():
    M, N, dim = 4, 8, 1
    x = torch.randn(M, N, dtype=torch.float32).npu()
    out = torch.empty(M, dtype=torch.float32).npu()
    tt_sum_2d[1, 1, 1](x, out, M, N, dim)
    ref = torch.sum(x, dim=dim)
    assert torch.allclose(out.cpu(), ref.cpu()), "sum result mismatch"
    print("sum result:", out)


if __name__ == "__main__":
    test_sum()
