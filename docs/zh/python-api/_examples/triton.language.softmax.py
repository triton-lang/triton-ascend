@triton.jit
def tt_softmax_3d(in_ptr, out_ptr, xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr, XB: tl.constexpr,
                  YB: tl.constexpr, ZB: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    a = tl.load(in_ptr + idx)
    ret = tl.softmax(a, dim=2)

    tl.store(out_ptr + idx, ret)
