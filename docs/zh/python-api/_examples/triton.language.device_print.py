import triton
import triton.language as tl


@triton.jit
def kernel(x_ptr):
    idx = tl.arange(0, 3)
    idy = tl.arange(0, 4)
    offset = idx[:, None] * 4 + idy[None, :]
    val = tl.load(x_ptr + offset)
    # Print the 2D tensor val
    tl.device_print("val:", val)
