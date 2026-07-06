@triton.jit
def triton_kernel(out_ptr0, in_ptr0, in_ptr1, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(in_ptr0 + idx)
    y = tl.load(in_ptr1 + idx)
    ret = x // y
    tl.store(out_ptr0 + idx, ret)
