@triton.jit
def bitwise_and_fn(a, b):
    return a & b


@triton.jit
def bitwise_or_fn(a, b):
    return a | b


@triton.jit
def bitwise_xor_fn(a, b):
    return a ^ b


@triton.jit
def minimum_fn(a, b):
    return tl.minimum(a, b)


@triton.jit
def maximum_fn(a, b):
    return tl.maximum(a, b)


@triton.jit
def triton_kernel_2d_scan(
    out_ptr0,
    in_ptr0,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    numel_x: tl.constexpr,
    numel_r: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    combine_fn_name: tl.constexpr,
):
    tl.static_assert(numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel")
    tl.static_assert(numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel")
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)

    if combine_fn_name == "maximum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=maximum_fn)
    elif combine_fn_name == "minimum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=minimum_fn)
    elif combine_fn_name == "bitwise_or_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_or_fn)
    elif combine_fn_name == "bitwise_xor_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_xor_fn)
    elif combine_fn_name == "bitwise_and_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_and_fn)
    tl.store(out_ptr0 + idx, ret)
