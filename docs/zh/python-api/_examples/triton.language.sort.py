@triton.jit
def sort_kernel_2d(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    pid = tl.program_id(0)
    offx = tl.arange(0, M)
    offy = pid * M
    off2d = offx + offy
    x = tl.load(X + off2d)
    x = tl.sort(x, dim=0, descending=descending)
    tl.store(Z + off2d, x)
