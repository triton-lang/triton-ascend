@triton.jit
def fn_f32(output_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    ret = tl.full((XB, YB, ZB), value=100, dtype=tl.float32)
    oidx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)
