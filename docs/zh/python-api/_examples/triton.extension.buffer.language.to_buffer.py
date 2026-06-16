@triton.jit
def to_buffer_kernel():
    a = tl.full((32, 2, 4), 0, dtype=tl.int64)
    a_buf = bl.to_buffer(a)
    b = tl.full((32, 2, 4), 0, dtype=tl.int64)
    b_buf = bl.to_buffer(b, al.ascend_address_space.UB)
    c = tl.full((32, 2, 4), 0, dtype=tl.int64)
    c_buf = bl.to_buffer(c, al.ascend_address_space.L1)
