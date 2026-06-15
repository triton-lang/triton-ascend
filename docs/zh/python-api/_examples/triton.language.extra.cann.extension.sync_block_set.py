@triton.jit
def kernel_sync_cube_to_vector():
    with al.scope(core_mode="cube"):
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
