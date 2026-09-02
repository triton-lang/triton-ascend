import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

# Configuration fragment and kernel definition, not a complete runnable
# example. Replace BITCODE_PATH with an existing bitcode file containing
# custom_example_func. Host launch and output validation are omitted.
BITCODE_PATH = "/absolute/path/to/custom_ops.bc"


@al.register_custom_op
class custom_example_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "custom_example_func"
    bitcode = BITCODE_PATH

    def __init__(self, x, out=None):
        assert out is not None, "out is required"


@triton.jit
def custom_example_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = al.custom("custom_example_op", x, out=y)
    tl.store(out_ptr + offsets, result, mask=mask)
