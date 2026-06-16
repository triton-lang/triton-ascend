@triton.jit
def view_example(out_ptr):
    x = tl.zeros([2, 3, 4], dtype=tl.float32)
    y = tl.view(x, [6, 4])
    offs = tl.arange(0, 6)[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + offs, y)
