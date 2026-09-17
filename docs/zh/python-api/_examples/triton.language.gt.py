import torch

import triton
import triton.language as tl


@triton.jit
def triton_gt_3d(in_ptr0, in_ptr1, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0 > x1
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)


def test_gt():
    shape = (2, 4, 8)
    torch.manual_seed(0)
    x_cpu = torch.randn(shape, dtype=torch.float32)
    y_cpu = torch.randn(shape, dtype=torch.float32)

    out = torch.empty(shape, dtype=torch.bool, device="npu")
    triton_gt_3d[(1, 1, 1)](x_cpu.npu(), y_cpu.npu(), out, L=shape[0], M=shape[1], N=shape[2], debug=True)

    assert torch.equal(out.cpu(), torch.gt(x_cpu, y_cpu))


if __name__ == "__main__":
    test_gt()
