import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def triton_xorsum_2d(in_ptr0, out_ptr0, dim: tl.constexpr, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr,
                     NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0, MNUMEL)
    nblk_idx = tl.arange(0, NNUMEL)
    mmask = mblk_idx < M
    nmask = nblk_idx < N
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * N + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=0)
    tmp4 = tl.xor_sum(x, axis=dim)
    if dim == 0:
        tl.store(out_ptr0 + tl.arange(0, N), tmp4, None)
    else:
        tl.store(out_ptr0 + tl.arange(0, M), tmp4, None)


def test_xor_sum():
    M, N, dim = 4, 8, 1
    x = torch.randint(0, 128, (M, N), dtype=torch.int32).npu()
    out = torch.empty(M, dtype=torch.int32).npu()
    triton_xorsum_2d[1, 1, 1](x, out, dim, M, N, M, N)
    ref = torch.zeros(M, dtype=torch.int32)
    for i in range(M):
        val = 0
        for j in range(N):
            val ^= x[i, j].item()
        ref[i] = val
    assert torch.equal(out.cpu(), ref), "xor_sum result mismatch"
    print("xor_sum result:", out)


if __name__ == "__main__":
    test_xor_sum()
