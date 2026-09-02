import triton.language.extra.cann.extension as al

# Configuration fragments, not complete runnable examples. Replace BITCODE_PATH
# with an existing bitcode file that contains both configured device functions.
BITCODE_PATH = "/absolute/path/to/custom_ops.bc"


@al.register_custom_op
class pipe_custom_op_example:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "pipe_custom_op_func"
    bitcode = BITCODE_PATH


@al.register_custom_op
class pipe_custom_macro_example:
    core = al.CORE.VECTOR
    pipe = (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V)
    mode = al.MODE.SIMD
    symbol = "pipe_custom_macro_func"
    bitcode = BITCODE_PATH
