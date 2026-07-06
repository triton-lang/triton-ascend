import torch

import triton
import triton.language as tl


@triton.jit
def cdiv_kernel(output_ptr, x_ptr, y_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, XNUMEL: tl.constexpr,
                YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs
    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]
    x = tl.load(x_ptr + idx)
    y = tl.load(y_ptr + idx)
    ret = tl.cdiv(x, y)
    tl.store(output_ptr + idx, ret)


def test_cdiv():
    torch.manual_seed(0)
    x_cpu = torch.randint(1, 33, (2, 4, 8), dtype=torch.int32)
    y_cpu = torch.randint(1, 9, (2, 4, 8), dtype=torch.int32)
    expected = torch.div(x_cpu + y_cpu - 1, y_cpu, rounding_mode="floor")
    shape = x_cpu.shape
    out = torch.empty(shape, dtype=torch.int32, device="npu")
    cdiv_kernel[(1, 1, 1)](
        out,
        x_cpu.npu(),
        y_cpu.npu(),
        XB=shape[0],
        YB=shape[1],
        ZB=shape[2],
        XNUMEL=shape[0],
        YNUMEL=shape[1],
        ZNUMEL=shape[2],
        debug=True,
    )
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_cdiv()
