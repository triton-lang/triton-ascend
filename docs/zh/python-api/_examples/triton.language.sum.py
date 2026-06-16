@triton.jit
def tt_sum_2d(in_ptr, out_ptr, xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr, XB: tl.constexpr,
              YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.sum(x, dim)

    if dim == 0:
        oidx = yidx
    else:
        oidx = xidx
    tl.store(out_ptr + oidx, ret)
