import torch

import triton
import triton.language as tl


@triton.jit
def tt_clamp_2d(in_ptr, out_ptr, min_ptr, max_ptr, xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    min_ = tl.load(min_ptr + idx)
    max_ = tl.load(max_ptr + idx)
    ret = tl.clamp(x, min_, max_)

    tl.store(out_ptr + idx, ret)


def test_clamp():
    shape = (8, 16)
    x_block, y_block = 4, 8
    dtype = torch.float32

    torch.manual_seed(0)
    x_cpu = torch.randn(shape, dtype=dtype)
    min_cpu = torch.randn(shape, dtype=dtype) - 0.5
    max_cpu = min_cpu + torch.rand(shape, dtype=dtype) + 0.5
    expected = torch.minimum(torch.maximum(x_cpu, min_cpu), max_cpu)

    x = x_cpu.npu()
    min_tensor = min_cpu.npu()
    max_tensor = max_cpu.npu()
    out = torch.empty_like(x)

    grid = (triton.cdiv(shape[0], x_block), triton.cdiv(shape[1], y_block))
    tt_clamp_2d[grid](
        x,
        out,
        min_tensor,
        max_tensor,
        xnumel=shape[0],
        ynumel=shape[1],
        znumel=1,
        XB=x_block,
        YB=y_block,
        ZB=1,
        debug=True,
    )

    torch.testing.assert_close(out.cpu(), expected, rtol=1e-3, atol=1e-3, equal_nan=True)


if __name__ == "__main__":
    test_clamp()
