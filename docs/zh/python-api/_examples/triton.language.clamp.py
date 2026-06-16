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
