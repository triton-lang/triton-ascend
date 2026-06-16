@triton.jit
def basic_constancy_example(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(A + offsets)

    # Use constexpr to declare the constancy pattern: every 4 consecutive values are equal
    # e.g., input pattern: [0,0,0,0,1,1,1,1,2,2,2,2,...]
    input_data = tl.max_constancy(input_data, [4])

    # The compiler can optimize based on this information
    result = input_data * 2
    tl.store(B + offsets, result)
