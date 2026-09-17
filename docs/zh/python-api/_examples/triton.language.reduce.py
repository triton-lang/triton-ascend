import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _reduce_combine(a, b):
    return a + b


@triton.jit
def tt_reduce_2d(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, dim: tl.constexpr):
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = mblk_idx[:, None] * N + nblk_idx[None, :]
    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)
    if dim == 0:
        tl.store(out_ptr + tl.arange(0, N), ret)
    else:
        tl.store(out_ptr + tl.arange(0, M), ret)


def test_reduce():
    M, N, dim = 4, 8, 1
    x = torch.randn(M, N, dtype=torch.float32).npu()
    out = torch.empty(M, dtype=torch.float32).npu()
    tt_reduce_2d[1, 1, 1](x, out, M, N, dim)
    ref = torch.sum(x, dim=dim)
    assert torch.allclose(out.cpu(), ref.cpu()), "reduce result mismatch"
    print("reduce result:", out)


if __name__ == "__main__":
    test_reduce()
