@triton.jit
def kernel_func(XBLOCK: tl.constexpr):
    buffer1 = bl.alloc(tl.float32, [XBLOCK])
    buffer1.to_tensor(writable=True)
    buffer2 = bl.alloc(tl.float32, [XBLOCK])
    bl.to_tensor(buffer2, writable=True)
