@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, N)
    tl.store(z_ptr + offset2, z)
