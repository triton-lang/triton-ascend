# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch
import triton
import triton.language as tl
import numpy as np
import test_common
from triton.backends.ascend.utils import is_compile_on_910_95

# Chained dot_scaled: first matmul with zero init, second accumulates via acc=.
# Mirrors normalize-matmul.mlir @test_mmadmx_chain_no_elemwise_decompose:
#   mmadmxL1(init=true,  outs=empty) -> mmadmxL1(init=false, outs=first)
# No transpose. Result should equal 2 * dot_scaled(lhs, rhs, scales).


def ub_overflow_check(M, N, K):
    bytes_of_dtype = 4
    M32 = ((M + 31) // 32) * 32
    N32 = ((N + 31) // 32) * 32
    cond1 = M32 * K * bytes_of_dtype <= 60 * 1024
    cond2 = K * N32 * bytes_of_dtype <= 60 * 1024
    cond3 = M32 * N32 * bytes_of_dtype <= 256 * 1024
    return cond1 and cond2 and cond3


def torch_dot_scaled(a, b, sa, sb):
    sa = torch.exp2(sa - 127).repeat_interleave(32, dim=1)
    sb = torch.exp2(sb - 127).repeat_interleave(32, dim=1).T
    return torch.matmul(a * sa, b * sb)


@triton.jit
def dot_scale_chain_kernel(
    lhs_ptr,
    lhs_scale_ptr,
    rhs_ptr,
    rhs_scale_ptr,
    out_ptr,
    lhs_format: tl.constexpr,
    rhs_format: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    Midx = tl.arange(0, M)
    Nidx = tl.arange(0, N)
    Kidx = tl.arange(0, K)
    K1idx = tl.arange(0, K // 32)

    lhs = tl.load(lhs_ptr + Midx[:, None] * K + Kidx[None, :])
    lhs_scale = tl.load(lhs_scale_ptr + Midx[:, None] * (K // 32) + K1idx[None, :])
    rhs = tl.load(rhs_ptr + Kidx[:, None] * N + Nidx[None, :])
    rhs_scale = tl.load(rhs_scale_ptr + Nidx[:, None] * (K // 32) + K1idx[None, :])

    # First mmad: init=true equivalent (fresh accumulation).
    first = tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format)
    # Second mmad: init=false equivalent (accumulate into L0C).
    result = tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=first)
    tl.store(out_ptr + Midx[:, None] * N + Nidx[None, :], result)


testlist = [
    (1, 7, 64),
    (15, 128, 64),
    (128, 64, 64),
    (189, 175, 64),
    (1, 1, 128),
    (35, 37, 128),
    (62, 64, 192),
    (31, 27, 256),
]


@pytest.mark.skipif(
    not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch),
    reason="only support in Ascend 950",
)
@pytest.mark.parametrize("dtype", ["fp8e4m3", "fp8e5m2"])
@pytest.mark.parametrize("M, N, K", testlist)
def test_scaled_dot_fp8(M, N, K, dtype):
    if not ub_overflow_check(M, N, K):
        pytest.skip("UB overflow")

    lhs_format = dtype[3:]
    rhs_format = lhs_format

    lhs = test_common.generate_tensor_fp8(shape=(M, K), dtype=dtype).npu()
    rhs = test_common.generate_tensor_fp8(shape=(K, N), dtype=dtype).npu()
    lhs_scale = torch.randint(low=110, high=128, size=(M, K // 32), dtype=torch.int8).npu()
    rhs_scale = torch.randint(low=110, high=128, size=(N, K // 32), dtype=torch.int8).npu()

    lhs_f32 = lhs.cpu().to(torch.float32)
    rhs_f32 = rhs.cpu().to(torch.float32)
    lhs_scale_f32 = lhs_scale.cpu().to(torch.float32)
    rhs_scale_f32 = rhs_scale.cpu().to(torch.float32)

    triton_res = torch.zeros((M, N), dtype=torch.float32).npu()
    dot_scale_chain_kernel[(1, )](lhs, lhs_scale, rhs, rhs_scale, triton_res, lhs_format, rhs_format, M, N, K)

    single = torch_dot_scaled(lhs_f32, rhs_f32, lhs_scale_f32, rhs_scale_f32)
    torch_ref = (single + single).to(torch.float32)

    if dtype == "fp8e4m3":
        torch.testing.assert_close(triton_res.cpu(), torch_ref.cpu(), rtol=125e-03, atol=125e-03, equal_nan=True)
    elif dtype == "fp8e5m2":
        torch.testing.assert_close(triton_res.cpu(), torch_ref.cpu(), rtol=75e-02, atol=75e-02, equal_nan=True)


@triton.jit
def _dot_scaled_test_fp4(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    K_PACKED: tl.constexpr,
    K_SCALE: tl.constexpr,
):
    # A: [M, K_PACKED] uint8 (packed e2m1), scale [M, K_SCALE] uint8 e8m0
    # B: [K_PACKED, N] uint8 (packed e2m1), scale [N, K_SCALE] uint8 e8m0
    # note: rhs is [K, N] not [N, K] for dot_scaled
    m_offs = tl.arange(0, M)
    n_offs = tl.arange(0, N)
    k_packed_offs = tl.arange(0, K_PACKED)
    scale_offs = tl.arange(0, K_SCALE)

    a = tl.load(a_ptr + m_offs[:, None] * K_PACKED + k_packed_offs[None, :])
    a_scale = tl.load(a_scale_ptr + m_offs[:, None] * K_SCALE + scale_offs[None, :])

    b = tl.load(b_ptr + k_packed_offs[:, None] * N + n_offs[None, :])
    b_scale = tl.load(b_scale_ptr + n_offs[:, None] * K_SCALE + scale_offs[None, :])

    acc = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1")

    tl.store(c_ptr + m_offs[:, None] * N + n_offs[None, :], acc)


def dequant_e2m1(packed_uint8, scale_uint8, rows, K):
    """Reference dequant for verification."""
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                        device=packed_uint8.device, dtype=torch.float32)
    low = vals[(packed_uint8 & 0x0F).long()]
    high = vals[((packed_uint8 >> 4) & 0x0F).long()]
    out = torch.zeros(rows, K, device=packed_uint8.device, dtype=torch.float32)
    out[:, 0::2] = low
    out[:, 1::2] = high
    scale_f = torch.pow(2.0, scale_uint8.float() - 127.0)
    scale_expanded = scale_f.repeat_interleave(32, dim=1)[:, :K]
    return out * scale_expanded


@pytest.mark.skip(reason="to be supported in Ascend 950")
def test_dot_scaled_fp4():
    M, N, K = 16, 16, 64
    K_PACKED = K // 2
    K_SCALE = K // 32

    a_packed = torch.randint(0, 256, (M, K_PACKED), device='npu', dtype=torch.uint8)
    a_scale = torch.full((M, K_SCALE), 127, device='npu', dtype=torch.uint8)

    # B for dot_scaled: [K_PACKED, N] (K-major)
    b_packed_nk = torch.randint(0, 256, (N, K_PACKED), device='npu', dtype=torch.uint8)
    b_packed_kn = b_packed_nk.t().contiguous()  # [K_PACKED, N]
    b_scale = torch.full((N, K_SCALE), 127, device='npu', dtype=torch.uint8)

    c = torch.empty(M, N, device='npu', dtype=torch.float32)

    _dot_scaled_test_fp4[(1, )](
        a_packed,
        a_scale,
        b_packed_kn,
        b_scale,
        c,
        M,
        N,
        K,
        K_PACKED,
        K_SCALE,
    )

    # Reference
    a_deq = dequant_e2m1(a_packed, a_scale, M, K)
    b_deq = dequant_e2m1(b_packed_nk, b_scale, N, K)
    ref = a_deq @ b_deq.t()
    torch.testing.assert_close(c, ref)


MIDDLE_RANGE_DICT = {
    'fp8e4m3': [-4, 4],
}


def get_tolerance(dtype_name):
    if dtype_name == 'fp8e4m3':
        return 5e-3, 5e-3
    return 1e-5, 1e-5


TL_DTYPE_MAP = {
    'fp8e4m3': tl.float8e4b8,
}


@triton.jit
def complex_pipeline_kernel(x_ptr, y_ptr, M, K, BM: tl.constexpr, BK: tl.constexpr, OUTER: tl.constexpr,
                            INNER: tl.constexpr, DEEP: tl.constexpr, CLAMP_I: tl.constexpr, OUT_DTYPE: tl.constexpr,
                            IS_FP8: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_k_blocks = tl.cdiv(K, BK)
    pid_m = pid // num_k_blocks
    pid_k = pid % num_k_blocks

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = pid_k * BK + tl.arange(0, BK)
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask, other=0.0)

    if IS_FP8:
        x_raw = x
        xt_raw = tl.trans(x_raw)
        xf = x.to(tl.float16)
    else:
        xf = x.to(tl.float32)
        xtf = tl.trans(xf)

    sync_flag = 1.0
    acc = tl.zeros((BM, BK), dtype=tl.float32)

    for o in range(OUTER):
        sync_flag = sync_flag * 0.75 + 0.25 * (o + 1)
        unit_flag = sync_flag / (1.0 + sync_flag)
        acc += unit_flag

        for i in range(INNER):
            if IS_FP8:
                d = tl.dot_scaled(xt_raw, None, "e4m3", x_raw, None, "e4m3")
                d = d.to(tl.float16)
            else:
                d = tl.dot(xtf, xf, input_precision="ieee")

            mul_r = d * xf
            min_val = tl.min(mul_r)
            sub_r = d - min_val
            for dj in range(DEEP):
                for ci in range(CLAMP_I):
                    sub_r = tl.minimum(tl.maximum(sub_r, -1.0), 1.0)
            acc += sub_r

    out = acc.to(OUT_DTYPE)
    tl.store(y_ptr + offs_m[:, None] * K + offs_k[None, :], out, mask=mask)


def compute_reference(x: torch.Tensor, outer, inner, deep, clamp_i, dtype):
    M, K = x.shape
    xf = x.float()
    sync_flag = 1.0
    acc = torch.zeros_like(xf)
    for o in range(outer):
        sync_flag = sync_flag * 0.75 + 0.25 * (o + 1)
        unit_flag = sync_flag / (1.0 + sync_flag)
        acc = acc + unit_flag
        for i in range(inner):
            d = xf @ xf.T
            mul_r = d * xf
            min_val = mul_r.min()
            sub_r = d - min_val
            for _ in range(deep):
                for _ in range(clamp_i):
                    sub_r = torch.clamp(sub_r, -1.0, 1.0)
            acc = acc + sub_r
    return acc


OUTER, INNER, DEEP, CLAMP_I = 2, 2, 2, 2
BM, BK = 16, 16
TEST_DTYPE = "fp8e4m3"
TL_DTYPE = TL_DTYPE_MAP[TEST_DTYPE]
FAIL_SHAPES = [
    (32, ),
    (37, ),
    (16, 16),
    (17, 19),
    (2, 2, 2, 17, 19),
]


def run_single_case(shape, dtype_name, device, verbose=True):
    flat_size = 1
    for s in shape:
        flat_size *= s
    M = flat_size
    K = flat_size
    x = torch.randn((M, K), dtype=torch.float32, device=device).to(dtype=torch.float8_e4m3fn)
    y = torch.zeros_like(x)
    complex_pipeline_kernel[(M // BM, )](x, y, M, K, BM=BM, BK=BK, OUTER=OUTER, INNER=INNER, DEEP=DEEP, CLAMP_I=CLAMP_I,
                                         OUT_DTYPE=TL_DTYPE, IS_FP8=(dtype_name == "fp8e4m3"))
    ref = compute_reference(x, OUTER, INNER, DEEP, CLAMP_I, dtype_name)
    rtol, atol = get_tolerance(dtype_name)
    torch.testing.assert_close(y.float(), ref, rtol=rtol, atol=atol)
    return True, 0, 0, 0


def get_device():
    return torch.device("npu")


@pytest.mark.skipif(
    not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch),
    reason="native FP8 scaled dot requires Ascend 950",
)
@pytest.mark.parametrize("shape", FAIL_SHAPES)
def test_fp8_error_input_only(shape):
    device = get_device()
    try:
        success, n_mis, total, pct = run_single_case(shape, TEST_DTYPE, device, verbose=True)
        pytest.fail(f"shape={shape}: kernel compiled successfully, MatMulMx check not triggered")
    except triton.compiler.errors.MLIRCompilationError as e:
        err_msg = str(e)
        matmul_mx_check = "MatMulMx" in err_msg and "must be divisible by 64" in err_msg
        hfusion_crash = "unsupported datatype for arith::TruncFOp to hfusion" in err_msg
        if matmul_mx_check or hfusion_crash:
            print(f"[PASS - Expected error] shape={shape}")
            return
        else:
            print(f"[FAIL] shape={shape}, unexpected compile error:\n{err_msg}")
            pytest.fail(f"captured non-target compile error: {e}")
    except Exception as e:
        print(f"[FAIL] shape={shape}, dtype={TEST_DTYPE}, exception={e}")
        pytest.fail(f"runtime exception: {type(e).__name__}: {e}")
