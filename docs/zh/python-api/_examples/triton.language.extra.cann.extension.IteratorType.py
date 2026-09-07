import triton.language.extra.cann.extension as al

# Configuration fragment, not a complete runnable example. Replace BITCODE_PATH
# with an existing bitcode file that contains the configured device function.
BITCODE_PATH = "/absolute/path/to/custom_ops.bc"


@al.register_custom_op
class iterator_type_example_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "iterator_type_example_func"
    bitcode = BITCODE_PATH
    # Logical iteration order: an independent dimension followed by a
    # reduction dimension.
    iterator_types = [
        al.IteratorType.Parallel,
        al.IteratorType.Reduction,
    ]
