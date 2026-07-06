import torch

import triton
import triton.language as tl


@triton.jit
def abs_kernel(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, XNUMEL: tl.constexpr,
               YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs
    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]
    x = tl.load(x_ptr + idx)
    ret = tl.abs(x)
    tl.store(output_ptr + idx, ret)


def test_abs():
    torch.manual_seed(0)
    x_cpu = torch.randn((2, 4, 8), dtype=torch.float32)
    shape = x_cpu.shape
    out = torch.empty_like(x_cpu, device="npu")
    abs_kernel[(1, 1, 1)](
        out,
        x_cpu.npu(),
        XB=shape[0],
        YB=shape[1],
        ZB=shape[2],
        XNUMEL=shape[0],
        YNUMEL=shape[1],
        ZNUMEL=shape[2],
        debug=True,
    )
    torch.testing.assert_close(out.cpu(), torch.abs(x_cpu), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_abs()
