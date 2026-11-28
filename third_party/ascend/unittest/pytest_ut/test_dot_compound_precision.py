import pytest
import torch
import numpy as np

import triton
import triton.language as tl


@triton.jit
def _compound_dot(A, B, C, D, PRECISION: tl.constexpr, BATCHED: tl.constexpr):
    r = tl.arange(0, 16)
    offset = r[:, None] * 16 + r[None, :]
    if BATCHED:
        offset = tl.arange(0, 2)[:, None, None] * 256 + offset[None, :, :]
    a = tl.load(A + offset)
    b = tl.load(B + offset)
    c = tl.load(C + offset)
    result = tl.dot(a, b, c, input_precision=PRECISION)
    tl.store(D + offset, result)


def _tf32(value, nearest):
    bits = value.view(torch.int32).to(torch.int64) & 0xffffffff
    special = (bits & 0x7fffffff) >= 0x7f800000
    rounded = ((bits + (0x1000 if nearest else 0)) & 0xffffe000)
    return torch.where(special, bits, rounded).to(torch.int32).view(torch.float32)


def _reference(a, b, c, precision):
    # CPU products in FP64 provide an independent oracle for the decomposition;
    # convert each accumulated partial result back to the required FP32.
    def dot(x, y, acc):
        return (x.double() @ y.double() + acc.double()).float()

    if precision == "tf32x3":
        ah, bh = _tf32(a, True), _tf32(b, True)
        al, bl = _tf32(a - ah, False), _tf32(b - bh, False)
        parts = [(al, bh), (ah, bl)]
    else:

        def split(value):
            result = []
            for _ in range(3 if precision == "bf16x6" else 2):
                part = value.to(torch.bfloat16).float()
                result.append(part)
                value = value - part
            return result

        aa, bb = split(a), split(b)
        ah, bh = aa[0], bb[0]
        parts = []
        if precision == "bf16x6":
            parts += [(aa[1], bb[1]), (aa[2], bb[0]), (aa[0], bb[2])]
        parts += [(aa[1], bb[0]), (aa[0], bb[1])]
    result = torch.zeros_like(c)
    for lhs, rhs in parts:
        result = dot(lhs, rhs, result)
    result = torch.where(torch.isnan(result), 0., result)
    return dot(ah, bh, result) + c


@pytest.mark.parametrize("precision", ["tf32x3", "bf16x3", "bf16x6"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("nonzero_acc", [False, True])
@pytest.mark.parametrize("data", ["random", "rounding", "cancellation", "tiny", "inf", "nan"])
def test_dot_compound_precision(precision, batched, nonzero_acc, data):
    generator = torch.Generator().manual_seed(42)
    shape = (2, 16, 16) if batched else (16, 16)
    a = torch.randn(shape, generator=generator)
    b = torch.randn(shape, generator=generator)
    c = torch.randn(shape, generator=generator) if nonzero_acc else torch.zeros(shape)
    if data == "rounding":
        # The low x low contribution distinguishes BF16x3 from BF16x6/IEEE;
        # the residual terms distinguish every mode from one HF32/BF16 dot.
        a.fill_(1.00390625)
        b.fill_(1.00390625)
    elif data == "cancellation":
        a = 1 + a * 0.01
        b = 1 + b * 0.01
        b[..., 1::2, :] *= -1
    elif data == "tiny":
        a *= 2.**-60
        b *= 2.**40
    elif data in ("inf", "nan"):
        a.fill_(1.)
        b.fill_(1.)
        a[..., 0, 0] = float(data)
    expected = _reference(a, b, c, precision)
    actual = torch.empty(shape, dtype=torch.float32, device="npu")
    _compound_dot[(1, )](a.npu(), b.npu(), c.npu(), actual, precision, batched)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-6, equal_nan=True)


@triton.jit
def _chained_compound_dot(A, B, W, D, PRECISION: tl.constexpr, SIZE: tl.constexpr):
    r = tl.arange(0, SIZE)
    offset = r[:, None] * SIZE + r[None, :]
    a = tl.load(A + offset)
    b = tl.load(B + offset)
    w = tl.load(W + offset)
    intermediate = tl.dot(a, b, input_precision=PRECISION)
    result = tl.dot(intermediate, w, input_precision=PRECISION)
    tl.store(D + offset, result)


@pytest.mark.parametrize("precision", ["tf32x3", "bf16x3", "bf16x6"])
@pytest.mark.parametrize("size", [16, 32, 64])
def test_chained_compound_dot(precision, size):
    generator = np.random.RandomState(17)
    a, b, w = [torch.from_numpy(generator.randn(size, size).astype(np.float32)) for _ in range(3)]
    a *= .1
    b *= .1
    zero = torch.zeros_like(a)
    intermediate = _reference(a, b, zero, precision)
    expected = _reference(intermediate, w, zero, precision)
    actual = torch.empty_like(a, device="npu")
    _chained_compound_dot[(1, )](a.npu(), b.npu(), w.npu(), actual, precision, size)
    # Two K-term FP32 reductions may use different accumulation orders than
    # the CPU oracle. Use the componentwise gamma_(2K) * |A| |B| |W| budget,
    # gamma_n = n*u/(1-n*u), u = 2^-24, instead of a fixed absolute tolerance.
    # Form the budget in FP64, including the scale for cancellation-heavy sums.
    unit_roundoff = 2.**-24
    gamma = (2 * size * unit_roundoff) / (1 - 2 * size * unit_roundoff)
    scale = a.double().abs() @ b.double().abs() @ w.double().abs()
    scaled_error = (actual.cpu().double() - expected.double()) / scale
    torch.testing.assert_close(scaled_error, torch.zeros_like(scale), rtol=0, atol=gamma)
