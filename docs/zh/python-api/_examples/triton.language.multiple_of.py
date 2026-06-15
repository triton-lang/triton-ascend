@triton.jit
def basic_multiple_of_example(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(A + offsets)

    # Declare that the first element of the input tensor is a multiple of BLOCK_SIZE
    input_data = tl.multiple_of(input_data, [BLOCK_SIZE])
