@triton.jit
def kernel(out_ptr0, in_ptr1, in_ptr2, in_ptr3, stride_in_r, XS: tl.constexpr, RS: tl.constexpr):
    pid = tl.program_id(0)
    in_idx0 = pid * XS + tl.arange(0, XS)
    in_idx1 = tl.arange(0, RS)
    tmp0 = tl.arange(0, XS)
    tmp1 = tl.load(in_ptr1 + in_idx1)
    in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
    tmp2 = tl.load(in_ptr2 + in_idx2)
    tmp2 = tl.math.exp(tmp2)
    tmp3 = tl.load(in_ptr3 + in_idx1)
    tmp3 = tmp3 + 1 - 8
    out0_idx = in_idx0[:, None] * RS + tmp3[None, :]
    tl.store(out_ptr0 + out0_idx, tmp2)
