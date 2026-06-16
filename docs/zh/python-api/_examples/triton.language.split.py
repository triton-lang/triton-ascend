@triton.jit
def complex_split_kernel(complex_ptr, real_ptr, imag_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    complex_data = tl.load(complex_ptr + offsets, mask=mask)
    real_part, imag_part = complex_data.split()
    tl.store(real_ptr + offsets, real_part, mask=mask)
    tl.store(imag_ptr + offsets, imag_part, mask=mask)
