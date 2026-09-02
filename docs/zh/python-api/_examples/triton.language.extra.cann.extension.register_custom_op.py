import triton.language.extra.cann.extension as al

# Configuration fragment, not a complete runnable example. Replace BITCODE_PATH
# with an existing bitcode file that contains registered_example_func.
BITCODE_PATH = "/absolute/path/to/custom_ops.bc"


@al.register_custom_op
class registered_example_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "registered_example_func"
    bitcode = BITCODE_PATH

    def __init__(self, x, out=None):
        assert out is not None, "out is required"
