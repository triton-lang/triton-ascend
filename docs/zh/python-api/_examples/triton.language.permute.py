@triton.jit
def permute_example(out_ptr):
    x = tl.zeros([2, 3, 4], dtype=tl.float32)
    y = tl.permute(x, [2, 0, 1])
    offs = (tl.arange(0, 4)[:, None, None] * (2 * 3) + tl.arange(0, 2)[None, :, None] * 3 +
            tl.arange(0, 3)[None, None, :])
    tl.store(out_ptr + offs, y)
