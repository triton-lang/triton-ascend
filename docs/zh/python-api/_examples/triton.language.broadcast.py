@triton.jit
def broadcast_kernel(output_ptr, BLOCK_SIZE: tl.constexpr):
    scalar = 5.0
    vector = tl.arange(0, BLOCK_SIZE) * 1.0
    broadcasted_scalar = tl.broadcast(scalar, vector)
    result = vector + broadcasted_scalar
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + offsets, result)
