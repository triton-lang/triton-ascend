import pytest
import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95


@triton.jit
def _dot_scaled_optional_scale(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, out_ptr, FORMAT: tl.constexpr, M: tl.constexpr,
                               N: tl.constexpr, K: tl.constexpr):
    PACKED_K: tl.constexpr = K // 2 if FORMAT == "e2m1" else K
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, PACKED_K)
    scale_k = tl.arange(0, K // 32)
    a = tl.load(a_ptr + m[:, None] * PACKED_K + k[None, :])
    b = tl.load(b_ptr + k[:, None] * N + n[None, :])
    a_scale = None
    b_scale = None
    if a_scale_ptr is not None:
        a_scale = tl.load(a_scale_ptr + m[:, None] * (K // 32) + scale_k[None, :])
    if b_scale_ptr is not None:
        b_scale = tl.load(b_scale_ptr + n[:, None] * (K // 32) + scale_k[None, :])
    result = tl.dot_scaled(a, a_scale, FORMAT, b, b_scale, FORMAT, out_dtype=tl.float32)
    tl.store(out_ptr + m[:, None] * N + n[None, :], result)


@pytest.mark.skipif(
    not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch),
    reason="native FP8/FP4 scaled dot requires Ascend 950",
)
@pytest.mark.parametrize("dtype", ["e4m3", "e5m2", "e2m1"])
@pytest.mark.parametrize("missing_scale", ["lhs", "rhs", "both", "neither"])
def test_dot_scaled_optional_scale(dtype, missing_scale):
    # Different M/N and four scale groups exercise both default scale shapes.
    m, n, k = 16, 32, 128
    values = torch.tensor([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=torch.float32)
    a_indices = (3 * torch.arange(m)[:, None] + torch.arange(k)[None, :]) % len(values)
    b_indices = (5 * torch.arange(k)[:, None] + torch.arange(n)[None, :]) % len(values)
    a_ref = values[a_indices]
    b_ref = values[b_indices]
    if dtype == "e2m1":
        codes = torch.tensor([0xC, 0xA, 0x9, 0x1, 0x2, 0x4], dtype=torch.uint8)
        a_codes = codes[a_indices]
        b_codes = codes[b_indices]
        # Low/high nibbles hold consecutive even/odd logical K elements.
        a = a_codes[:, 0::2] | (a_codes[:, 1::2] << 4)
        b = b_codes[0::2, :] | (b_codes[1::2, :] << 4)
    else:
        torch_dtype = torch.float8_e4m3fn if dtype == "e4m3" else torch.float8_e5m2
        a = a_ref.to(torch_dtype).view(torch.uint8)
        b = b_ref.to(torch_dtype).view(torch.uint8)

    # Native scales use E8M0: 126/127/128 represent 0.5/1/2.
    groups = torch.arange(k // 32)[None, :]
    a_scale = (126 + (torch.arange(m)[:, None] + groups) % 3).to(torch.uint8)
    b_scale = (126 + (torch.arange(n)[:, None] + groups + 1) % 3).to(torch.uint8)
    if missing_scale in ("lhs", "both"):
        a_scale = None
    if missing_scale in ("rhs", "both"):
        b_scale = None

    scaled_a = a_ref
    scaled_b = b_ref
    if a_scale is not None:
        scaled_a = a_ref * torch.exp2(a_scale.float() - 127).repeat_interleave(32, dim=1)
    if b_scale is not None:
        scaled_b = b_ref * torch.exp2(b_scale.float() - 127).repeat_interleave(32, dim=1).T
    expected = scaled_a @ scaled_b

    output = torch.full((m, n), float("nan"), dtype=torch.float32, device="npu")
    _dot_scaled_optional_scale[(1, )](
        a.contiguous().npu(),
        b.contiguous().npu(),
        None if a_scale is None else a_scale.npu(),
        None if b_scale is None else b_scale.npu(),
        output,
        dtype,
        m,
        n,
        k,
    )
    # Inputs, scales and sums are exactly representable in FP32.
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
