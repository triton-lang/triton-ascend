@triton.jit
def kernel(out_ptr0, in_ptr0, in_ptr1, in_ptr2, stride_in_r, XS: tl.constexpr, RS: tl.constexpr):
    pid = tl.program_id(0)
    in_idx0 = pid * XS + tl.arange(0, XS)
    in_idx1 = tl.arange(0, RS)
    tmp0 = tl.load(in_ptr0 + in_idx0)
    tmp1 = tl.load(in_ptr1 + in_idx1)
    in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
    tmp2 = tl.load(in_ptr2 + in_idx2)
    out0_idx = in_idx0[:, None] * RS + in_idx1[None, :]
    tl.store(out_ptr0 + out0_idx, tmp2)
