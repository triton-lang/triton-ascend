@triton.jit
def flatten_kernel(x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    x = tl.load(x_ptr + offsets, mask=mask)
    x_flat = x.ravel()
    tl.store(output_ptr + offsets, x_flat, mask=mask)
