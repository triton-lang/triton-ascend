@triton.jit
def allocate_local_buffer(XBLOCK: tl.constexpr):
    bl.alloc(tl.float32, [XBLOCK])
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L1)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0A)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0B)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0C)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB, is_mem_unique=True)
