@triton.jit
def kernel(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, XNUMEL: tl.constexpr,
           YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XB) + tl.program_id(0) * XB
    yidx = tl.arange(0, YB) + tl.program_id(1) * YB
    zidx = tl.arange(0, ZB) + tl.program_id(2) * ZB
    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]
    X = tl.load(x_ptr + idx)
    ret = tl.flip(X, 2)
    oidx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)
