@triton.jit
def dot_scale_kernel(a_base, stride_a0: tl.constexpr, stride_a1: tl.constexpr, a_scale, b_base, stride_b0: tl.constexpr,
                     stride_b1: tl.constexpr, b_scale, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, type_a: tl.constexpr, type_b: tl.constexpr):
    PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K
    PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K
    str_a0: tl.constexpr = stride_a0
    a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0, str_a0)[None, :] * stride_a1
    b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * stride_b0 + tl.arange(0, BLOCK_N)[None, :] * stride_b1

    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if a_scale is not None:
        scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]
        a_scale = tl.load(scale_a_ptr)
    if b_scale is not None:
        scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]
        b_scale = tl.load(scale_b_ptr)
    accumulator = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b, acc=accumulator, out_dtype=tl.float32)

    out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr, accumulator.to(a.dtype))


x = torch.randn(shape, dtype=torch.bfloat16, device="npu")
y = torch.randn(shape, dtype=torch.bfloat16, device="npu")
M, K = shape[0], shape[1]
scale_x = torch.randint(min_scale - 128, max_scale - 127, (M, K // 32), dtype=torch.int8, device="npu")
scale_y = torch.randint(min_scale - 128, max_scale - 127, (N, K // 32), dtype=torch.int8, device="npu")
type_a, type_b = "bf16", "bf16"
pgm = dot_scale_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a, type_b)
