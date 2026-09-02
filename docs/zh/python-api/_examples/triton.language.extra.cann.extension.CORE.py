import triton.language.extra.cann.extension as al

# Configuration fragment, not a complete runnable example. Replace BITCODE_PATH
# with an existing bitcode file that contains the configured device function.
BITCODE_PATH = "/absolute/path/to/custom_ops.bc"


@al.register_custom_op
class core_example_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "core_example_func"
    bitcode = BITCODE_PATH
