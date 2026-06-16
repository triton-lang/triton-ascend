@triton.jit
def optimized_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # Use static_range for small-scale loop unrolling, eliminating loop overhead
    for i in tl.static_range(BLOCK_SIZE):
        # When BLOCK_SIZE is a compile-time constant, the entire loop is unrolled
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)

    # Comparison: using range incurs loop control overhead
    for i in tl.range(BLOCK_SIZE):
        # This loop has runtime loop control logic
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)
