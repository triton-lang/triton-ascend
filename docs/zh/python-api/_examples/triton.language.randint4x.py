@triton.jit
def kernel_randint4x(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, 4)
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(0, block_size, step=4):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint4x(5, 10 + global_offset, n_rounds)  # Generate random numbers for each index
        mask = (global_offset + indices) < N
        tl.store(x_ptr + global_offset + indices, rand_vals, mask)  # Store the random numbers


y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()
kernel_randint4x[ncore, 1, 1](y_cali, 10, numel, xblock)
