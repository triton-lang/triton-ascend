@triton.jit
def trans_example():
    x = tl.zeros([2, 3, 4], dtype=tl.float32)
    y = tl.trans(x, [2, 0, 1])
    return y
