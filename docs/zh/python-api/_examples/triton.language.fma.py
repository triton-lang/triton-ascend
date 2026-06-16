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
