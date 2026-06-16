@triton.jit
def join_example(out_ptr):
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.full([2, 3], 1.0, dtype=tl.float32)
    z = tl.join(x, y)
    offs = (tl.arange(0, 2)[:, None, None] * (2 * 3) + tl.arange(0, 2)[None, :, None] * 3 +
            tl.arange(0, 3)[None, None, :])
    tl.store(out_ptr + offs, z)
