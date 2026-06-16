@triton.jit
def expand_dims_example(out_ptr):
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.expand_dims(x, axis=1)
    offs = (tl.arange(0, 2)[:, None, None] * 3 + tl.arange(0, 1)[None, :, None] * 3 + tl.arange(0, 3)[None, None, :])
    tl.store(out_ptr + offs, y)
