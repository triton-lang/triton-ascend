@triton.jit
def kernel_randint(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint(5, 10 + global_offset, n_rounds)  # Generate a random number for each index
        tl.store(x_ptr + global_offset, rand_vals)  # Store the random number


y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()
kernel_randint[ncore, 1, 1](y_cali, 10, numel, xblock)
