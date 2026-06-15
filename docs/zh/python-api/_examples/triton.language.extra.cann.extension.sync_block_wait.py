@triton.jit
def kernel_sync_vector_to_cube():
    with al.scope(core_mode="vector"):
        al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    with al.scope(core_mode="cube"):
        al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
