import torch

import triton
import triton.language as tl


@triton.jit
def triton_logical_or_3d(in_ptr0, in_ptr1, out_ptr0, XB, YB, ZB, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L) + tl.program_id(0) * XB
    mblk_idx = tl.arange(0, M) + tl.program_id(1) * YB
    nblk_idx = tl.arange(0, N) + tl.program_id(2) * ZB
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0.logical_or(x1)
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)


def test_logical_or():
    shape = (2, 4, 8)
    block = shape
    torch.manual_seed(0)
    x_cpu = torch.randint(-1, 2, shape, dtype=torch.int32)
    y_cpu = torch.randint(-1, 2, shape, dtype=torch.int32)

    out = torch.empty(shape, dtype=torch.bool, device="npu")
    triton_logical_or_3d[(1, 1, 1)](
        x_cpu.npu(),
        y_cpu.npu(),
        out,
        block[0],
        block[1],
        block[2],
        L=shape[0],
        M=shape[1],
        N=shape[2],
        debug=True,
    )

    assert torch.equal(out.cpu(), torch.logical_or(x_cpu, y_cpu))


if __name__ == "__main__":
    test_logical_or()
