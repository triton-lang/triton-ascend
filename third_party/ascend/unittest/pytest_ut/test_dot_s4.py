import torch
import triton
import triton.language as tl
import pytest
import test_common
import triton.language.extra.cann.extension as al

def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    orig_shape = tensor.shape
    last_dim = orig_shape[-1]

    if last_dim % 2 != 0:
        padding = [0] * (2 * tensor.ndim)
        padding[1] = 1
        tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)

    pairs = tensor.view(tensor.shape[:-1] + (-1, 2))
    a = pairs[..., 0]
    b = pairs[..., 1]
    a4 = a & 0x0F
    b4 = b & 0x0F
    packed = ((b4 << 4) | a4).to(torch.int8)
    return packed

@triton.jit
def dot_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * tl.cdiv(BLOCK_SIZE_N, 2) + tl.arange(0, tl.cdiv(BLOCK_SIZE_N, 2))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k2 = tl.arange(0, 2 * BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(0, K, 2 * BLOCK_SIZE_K):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (offs_k + tl.cdiv(k, 2))[None, :] * stride_ak
        b_ptrs = b_ptr + (offs_k2 + k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=((offs_m[:, None] < M) & ((offs_k + tl.cdiv(k, 2))[None, :] < tl.cdiv(K, 2))), other=0)
        b = tl.load(b_ptrs, mask=(((offs_k2 + k)[:, None] < K) & (offs_n[None, :] < tl.cdiv(N, 2))), other=0)
        accumulator += al.dot_s4(a, b)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@pytest.mark.parametrize(
    "M, N, K, BLOCK_M, BLOCK_N, BLOCK_K",
    [
        (128, 256, 128, 16, 32, 64),
        (64, 64, 64, 16, 16, 32),
        (32, 128, 32, 8, 16, 32),
    ]
)
def test_dot_s4(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8, device='npu')
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8, device='npu')
    a_packed = pack_int4(a)
    b_packed = pack_int4(b)
    c = torch.zeros((M, N), dtype=torch.int32, device='npu')

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    dot_kernel[grid](
        a_packed, b_packed, c,
        M, N, K,
        a_packed.stride(0), a_packed.stride(1),
        b_packed.stride(0), b_packed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )

    c_ref = torch.matmul(a.cpu().to(torch.int32), b.cpu().to(torch.int32)).npu()
    test_common.validate_cmp('int32', c, c_ref)

if __name__ == "__main__":
    test_dot_s4(128, 256, 128, 16, 32, 64)
