@triton.jit
def matrix_add_bias_kernel(x_ptr, bias_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr)
    bias_broadcast = bias.broadcast_to([BLOCK_M, BLOCK_N])
    output = x + bias_broadcast
    tl.store(output_ptr + offsets, output, mask=mask)
