@triton.jit
def triton_max_contiguous(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    val = tl.load(A + offsets)
    # Declare that the first BLOCK_SIZE elements in offset are contiguous
    input_data = tl.max_contiguous(val, [BLOCK_SIZE])

    # The compiler can generate more efficient memory access instructions
    result = input_data * 2
    tl.store(B + offsets, result)
