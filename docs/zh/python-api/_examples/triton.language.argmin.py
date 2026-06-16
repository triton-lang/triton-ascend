@triton.jit
def triton_argmin_1d(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None)
    tmp4 = tl.argmin(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)
