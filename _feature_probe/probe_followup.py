#!/usr/bin/env python3
"""Follow-up probes that must live in a real .py file for @triton.jit."""
from concurrent.futures import ThreadPoolExecutor

import torch
import triton
import triton.language as tl
from triton import AsyncCompileMode


def check(name, fn):
    try:
        fn()
        print("[PASS]", name)
        return True
    except Exception as e:
        print("[FAIL]", name, "::", type(e).__name__, e)
        return False


@triton.jit
def _dot_scaled_k(a_base, stride_a0, stride_a1, a_scale, b_base, stride_b0, stride_b1, b_scale, out,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type_a: tl.constexpr,
                  type_b: tl.constexpr):
    a = tl.load(a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0, BLOCK_K)[None, :] * stride_a1)
    b = tl.load(b_base + tl.arange(0, BLOCK_K)[:, None] * stride_b0 + tl.arange(0, BLOCK_N)[None, :] * stride_b1)
    SCALE_K: tl.constexpr = BLOCK_K // 32
    sa = tl.load(a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_K + tl.arange(0, SCALE_K)[None, :])
    sb = tl.load(b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_K + tl.arange(0, SCALE_K)[None, :])
    acc = tl.dot_scaled(a, sa, type_a, b, sb, type_b, out_dtype=tl.float32, fast_math=True)
    tl.store(out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], acc.to(a.dtype))


@triton.jit
def _copy_k(X, Y, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(Y + offs, tl.load(X + offs))


def probe_dot_scaled_fp16():
    M = N = K = 32
    x = torch.randn((M, K), dtype=torch.float16, device="npu").clamp_(-8, 7)
    y = torch.randn((K, N), dtype=torch.float16, device="npu").clamp_(-8, 7)
    sx = torch.randint(124, 131, (M, K // 32), dtype=torch.uint8, device="npu")
    sy = torch.randint(124, 131, (N, K // 32), dtype=torch.uint8, device="npu")
    z = x.new_empty((M, N))
    _dot_scaled_k[(1, )](x, *x.stride(), sx, y, *y.stride(), sy, z, M, N, K, "fp16", "fp16", num_warps=4)
    assert torch.isfinite(z).all()


def probe_async():
    x = torch.randn(32, device="npu")
    y = torch.empty_like(x)
    with ThreadPoolExecutor(max_workers=2) as ex:
        with AsyncCompileMode(ex):
            fut = _copy_k[(1, )](x, y, 32)
            if hasattr(fut, "result"):
                fut.result()
    torch.npu.synchronize()
    assert torch.allclose(x, y)


if __name__ == "__main__":
    check("dot_scaled fp16", probe_dot_scaled_fp16)
    check("AsyncCompileMode+executor", probe_async)
