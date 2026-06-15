import triton.language as tl


@triton.jit
def basic_static_print_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # Print the value of constants at compile time
    tl.static_print("BLOCK_SIZE =", BLOCK_SIZE)
    tl.static_print(BLOCK_SIZE)
    # Supports f-string printing
    tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
