@triton.jit
def triton_arange(z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
    off = tl.arange(0, BLOCK)
    val = tl.arange(START, END)
    tl.store(z + off, val)
