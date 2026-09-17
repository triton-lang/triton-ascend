import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def triton_argmax_2d(in_ptr0, out_ptr0, dim: tl.constexpr, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr,
                     NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0, MNUMEL)
    nblk_idx = tl.arange(0, NNUMEL)
    mmask = mblk_idx < M
    nmask = nblk_idx < N
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * N + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=-float('inf'))
    tmp4 = tl.argmax(x, dim)
    if dim == 0:
        tl.store(out_ptr0 + tl.arange(0, N), tmp4, None)
    else:
        tl.store(out_ptr0 + tl.arange(0, M), tmp4, None)


def test_argmax():
    M, N, dim = 4, 8, 1
    x = torch.randn(M, N, dtype=torch.float32).npu()
    out = torch.empty(M, dtype=torch.int32).npu()
    triton_argmax_2d[1, 1, 1](x, out, dim, M, N, M, N)
    ref = torch.argmax(x, dim=dim).to(torch.int32)
    assert torch.equal(out.cpu(), ref.cpu()), "argmax result mismatch"
    print("argmax result:", out)


if __name__ == "__main__":
    test_argmax()
