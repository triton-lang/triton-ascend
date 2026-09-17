import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    acc_dtype: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def test_matmul():
    shape = (16, 32)
    dtype = 'float32'
    M, N, K = shape[0], shape[0], shape[1]
    BLOCK_M, BLOCK_N, BLOCK_K = min(max(M, 16), 32), min(max(N, 16), 32), min(max(K, 16), 32)
    a = torch.randn((M, K), dtype=eval('torch.' + dtype)).npu()
    b = torch.randn((K, N), dtype=eval('torch.' + dtype)).npu()

    triton_res = torch.zeros((M, N), dtype=eval('torch.' + dtype)).npu()
    accumulator_type = tl.float32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    matmul_kernel[grid](a.npu(), b.npu(), triton_res, M, N, K, accumulator_type, a.stride(0), a.stride(1), b.stride(0),
                        b.stride(1), triton_res.stride(0), triton_res.stride(1), BLOCK_M, BLOCK_N, BLOCK_K)

    a_gold = a.to(torch.float32)
    b_gold = b.to(torch.float32)
    torch_res = torch.mm(a_gold, b_gold)
    torch.testing.assert_close(torch_res, triton_res)


if __name__ == "main":
    test_matmul()
