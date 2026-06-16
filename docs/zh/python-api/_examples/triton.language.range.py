import triton.language as tl


@triton.jit
def basic_examples():
    # Single argument: 0 to 9
    for i in tl.range(10):
        # i = 0, 1, 2, ..., 9
        pass

    # Two arguments: 2 to 9
    for i in tl.range(2, 10):
        # i = 2, 3, ..., 9
        pass

    # Three arguments: 0 to 10, step 2
    for i in tl.range(0, 10, 2):
        # i = 0, 2, 4, 6, 8
        pass
