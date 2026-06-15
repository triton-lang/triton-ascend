@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, XB: tl.constexpr):
    idx = tl.arange(0, XB)
    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)
    ret = tl.cat(X, Y, can_reorder=True)
    oidx = tl.arange(0, XB * 2)
    tl.store(output_ptr + oidx, ret)
