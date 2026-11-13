"""Opt-in device checks of ordinary JIT calls, without the offline selector.

Run on an installed Ascend backend with TRITON_BLOCK_AUTO_DEVICE_TEST=1.
Device visibility is owned by the test runner, never changed by the selector.
"""
import json
import os
from pathlib import Path

import pytest

if os.environ.get("TRITON_BLOCK_AUTO_DEVICE_TEST") != "1":
    pytest.skip("explicit NPU integration run required", allow_module_level=True)

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _multiply(X, Y, Z, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0)
    y = tl.load(Y + offsets, offsets < N, other=0)
    tl.store(Z + offsets, x * y, offsets < N)


@triton.jit
def _batch_matmul(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    batch = tl.program_id(0)
    m, n, k = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K)
    a = tl.load(A + batch * M * K + m[:, None] * K + k[None, :])
    b = tl.load(B + batch * K * N + k[:, None] * N + n[None, :])
    result = tl.dot(a, b, input_precision="ieee")
    tl.store(C + batch * M * N + m[:, None] * N + n[None, :], result)


@triton.jit
def _rms_norm(X, W, Y, RSTD, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + columns, columns < N, other=0).to(tl.float32)
    weight = tl.load(W + columns, columns < N, other=0)
    inverse = tl.rsqrt(tl.sum(x * x, 0) / N + 1e-6)
    tl.store(Y + row * N + columns, x * inverse * weight, columns < N)
    tl.store(RSTD + row, inverse)


def _record(name, kernel, *, allow_unmodeled_row_coalescing=False):
    report = kernel.block_auto_selection
    directory = os.environ.get("TRITON_BLOCK_AUTO_TEST_OUTPUT")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(
            json.dumps(
                {
                    "selection": report,
                    "launch": kernel.last_launch,
                    "metadata": kernel.metadata._asdict(),
                    "device_executed": True,
                }, default=str, indent=2) + "\n")
    assert report["device_executed"] is False  # Ranking itself did not launch.
    assert report["selected_factor"] in (1, 2, 4)
    assert report["candidate_count"] <= 3
    assert kernel.last_launch["applied_factor"] == report["selected_factor"]
    if not report["model_complete"] and allow_unmodeled_row_coalescing:
        # The normal pipeline can coalesce the baseline before block selection.
        # That extra schedule transform is intentionally outside this model.
        assert kernel.baseline.metadata.row_coalescing_applied is True
        assert report["selected_factor"] == 1
        assert report["candidates"]["unmerged"]["error"] == (
            "Unsupported: additional grid transform row_coalescing_applied")
    else:
        assert report["model_complete"], report


def test_vv_ordinary_jit_and_memory_cache():
    torch.npu.set_device(0)
    count = 1 << 20
    x = torch.randn(count, dtype=torch.float32, device="npu")
    y = torch.randn_like(x)
    actual, baseline = torch.empty_like(x), torch.empty_like(x)
    kernel = _multiply[(triton.cdiv(count, 1024), )](x, y, actual, count, 1024)
    torch.npu.synchronize()
    _record("vv", kernel, allow_unmodeled_row_coalescing=True)
    torch.testing.assert_close(actual.cpu(), x.cpu() * y.cpu(), rtol=1e-5, atol=1e-6)
    # A new address with the same launch contract must reuse the decision.
    reused = _multiply[(triton.cdiv(count, 1024), )](x, y, baseline, count, 1024)
    torch.npu.synchronize()
    assert reused is kernel
    torch.testing.assert_close(baseline, actual, rtol=0, atol=0)
    manual = _multiply[(triton.cdiv(count, 1024), )](x, y, baseline, count, 1024, block_batching_factor=1)
    torch.npu.synchronize()
    assert not hasattr(manual, "block_auto_selection")
    torch.testing.assert_close(baseline, actual, rtol=0, atol=0)


def test_cv_ordinary_jit():
    torch.npu.set_device(0)
    a = torch.randn((4, 16, 32), dtype=torch.float32, device="npu")
    b = torch.randn((4, 32, 16), dtype=torch.float32, device="npu")
    actual = torch.empty((4, 16, 16), dtype=torch.float32, device="npu")
    kernel = _batch_matmul[(4, )](a, b, actual, 16, 16, 32)
    torch.npu.synchronize()
    _record("cv", kernel)
    expected = torch.bmm(a.cpu().double(), b.cpu().double()).float()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)


def test_bf16_rms_norm_ordinary_jit():
    torch.npu.set_device(0)
    x = torch.randn((4096, 512), dtype=torch.bfloat16, device="npu")
    weight = torch.randn(512, dtype=torch.float32, device="npu")
    actual = torch.empty_like(x)
    inverse = torch.empty(4096, dtype=torch.float32, device="npu")
    kernel = _rms_norm[(4096, )](x, weight, actual, inverse, 512, 512)
    torch.npu.synchronize()
    _record("bf16_rms", kernel)
    assert kernel.last_launch["applied_factor"] in (2, 4)
    host = x.cpu().float()
    reference_inverse = torch.rsqrt((host * host).mean(dim=1) + 1e-6)
    expected = (host * reference_inverse[:, None] * weight.cpu()).to(torch.bfloat16)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(inverse.cpu(), reference_inverse, rtol=1e-4, atol=1e-5)
    other, other_inverse = torch.empty_like(actual), torch.empty_like(inverse)
    reused = _rms_norm[(4096, )](x, weight, other, other_inverse, 512, 512)
    torch.npu.synchronize()
    assert reused is kernel
    assert reused.last_launch["applied_factor"] == kernel.block_auto_selection["selected_factor"]
    torch.testing.assert_close(other.cpu(), actual.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(other_inverse.cpu(), inverse.cpu(), rtol=0, atol=0)
