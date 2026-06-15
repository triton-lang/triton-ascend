@triton.jit
def cast_example():
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.cast(x, tl.int32)
    return y


@triton.jit
def cast_advanced_example():
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = x.cast(tl.int32, bitcast=True)
    z = x.cast(tl.float16, fp_downcast_rounding="rtz")
    w = x.cast(tl.int8, overflow_mode="saturate")
    return y, z, w
