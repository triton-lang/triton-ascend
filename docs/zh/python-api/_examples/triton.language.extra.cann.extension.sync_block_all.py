@triton.jit
def test_sync_block_all():
    al.sync_block_all("all_cube", 8)
    al.sync_block_all("all_vector", 9)
    al.sync_block_all("all", 10)
    al.sync_block_all("all_sub_vector", 11)
