import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def dot_scale_kernel(a_base, stride_a0: tl.constexpr, stride_a1: tl.constexpr, a_scale, b_base, stride_b0: tl.constexpr,
                     stride_b1: tl.constexpr, b_scale, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, type_a: tl.constexpr, type_b: tl.constexpr):
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


def test_dot_scaled():
    M, K, N = 16, 64, 16
    x = torch.randn((M, K), dtype=torch.bfloat16, device="npu")
    y = torch.randn((K, N), dtype=torch.bfloat16, device="npu")
    type_a, type_b = "bf16", "bf16"
    # E8M0 scales: the stored byte is the biased exponent, 2 ** (scale - 127).
    # Keep the bytes in [124, 127] so they stay positive int8 values.
    scale_x = torch.randint(124, 128, (M, K // 32), dtype=torch.int8, device="npu")
    scale_y = torch.randint(124, 128, (N, K // 32), dtype=torch.int8, device="npu")
    out = x.new_empty((M, N), dtype=x.dtype)
    dot_scale_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, out, M, N, K, type_a, type_b)

    # Reference: operands are converted to BF16 first, then each 32-element K
    # block is multiplied by its E8M0 scale, and the dot product accumulates
    # in float32.
    factor_x = torch.pow(2.0, (scale_x.cpu().to(torch.int32) & 0xFF) - 127).to(torch.bfloat16)
    factor_y = torch.pow(2.0, (scale_y.cpu().to(torch.int32) & 0xFF) - 127).to(torch.bfloat16)
    a_scaled = (x.cpu() * factor_x.repeat_interleave(32, dim=1)).float()
    b_scaled = (y.cpu() * factor_y.repeat_interleave(32, dim=1).T).float()
    ref = (a_scaled @ b_scaled).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref)


if __name__ == "__main__":
    test_dot_scaled()
    print("test_dot_scaled passed!")
