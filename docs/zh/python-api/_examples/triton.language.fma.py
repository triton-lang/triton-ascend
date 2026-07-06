import torch

import triton
import triton.language as tl


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, XNUMEL: tl.constexpr,
            YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)
    Z = tl.load(z_ptr + idx)

    ret = tl.fma(X, Y, Z)

    tl.store(output_ptr + idx, ret)


def test_fma():
    shape = (2, 4, 8)
    torch.manual_seed(0)
    x_cpu = torch.randn(shape, dtype=torch.float32)
    y_cpu = torch.randn(shape, dtype=torch.float32)
    z_cpu = torch.randn(shape, dtype=torch.float32)
    expected = x_cpu * y_cpu + z_cpu

    out = torch.empty(shape, dtype=torch.float32, device="npu")
    fn_npu_[(1, 1, 1)](
        out,
        x_cpu.npu(),
        y_cpu.npu(),
        z_cpu.npu(),
        XB=shape[0],
        YB=shape[1],
        ZB=shape[2],
        XNUMEL=shape[0],
        YNUMEL=shape[1],
        ZNUMEL=shape[2],
        debug=True,
    )

    torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_fma()
