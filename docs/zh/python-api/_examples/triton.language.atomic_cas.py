@triton.jit
def atomic_cas(in_ptr0, in_ptr1, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    val = tl.load(in_ptr0 + (x0), xmask)
    cmp = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tl.atomic_cas(out_ptr0 + (x1), cmp, val)
    tl.store(out_ptr1 + (x1), tmp1, xmask)
