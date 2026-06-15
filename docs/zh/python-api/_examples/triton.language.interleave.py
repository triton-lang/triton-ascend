@triton.jit
def interleave_example():
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.ones([2, 3], dtype=tl.float32)
    z = tl.interleave(x, y)
    return z
