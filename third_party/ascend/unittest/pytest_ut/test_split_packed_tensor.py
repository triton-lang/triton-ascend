# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Split packed values must preserve all 64 bits and reduction carries."""
import numpy as np
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _compare_exchange(X, O, B, AXIS: tl.constexpr):
    offsets = tl.arange(0, 128)
    values = tl.load(X + offsets)
    if AXIS == 0:
        shaped = values.reshape(2, 64)
        mask = tl.arange(0, 2)[:, None].to(tl.int64)
    else:
        shaped = values.reshape(64, 2)
        mask = tl.arange(0, 2)[None, :].to(tl.int64)
    left = tl.sum(shaped * (1 - mask), AXIS)
    right = tl.sum(shaped * mask, AXIS)
    choose = left > right
    difference = left ^ right
    largest = right ^ tl.where(choose, difference, 0)
    smallest = left ^ tl.where(choose, difference, 0)
    out_offsets = tl.arange(0, 64)
    tl.store(O + out_offsets, largest)
    tl.store(O + 64 + out_offsets, smallest)
    # An extra wide consumer exercises materialization of a shared pair.
    tl.store(O + 128 + out_offsets, difference)
    tl.store(B + out_offsets, largest.to(tl.int32))
    tl.store(B + 64 + out_offsets, (largest.to(tl.uint64) >> 32).to(tl.int32))


def _values(reverse):
    # Equal high halves with unsigned-low crossing, high sign crossing, ties,
    # extrema, and negative values with nonzero low halves.
    pairs = [
        (-(1 << 63), (1 << 63) - 1),
        ((7 << 32) | 0x80000000, (7 << 32) | 0x7fffffff),
        (-1, -(1 << 32)),
        (0, 0),
        ((1 << 63) - 1, (1 << 63) - 1),
        (1 << 32, (1 << 32) - 1),
        (-(1 << 32) - 1, -(1 << 32) + 1),
        (-(1 << 63) + 0xffffffff, -(1 << 63) + 0x80000000),
    ] * 8
    data = torch.tensor(pairs, device="cpu", dtype=torch.int64)
    return data.flip(1) if reverse else data


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_compare_exchange_exact(axis, reverse):
    pairs = _values(reverse)
    shaped = pairs.T.contiguous() if axis == 0 else pairs
    inputs = shaped.to("npu")
    outputs = torch.empty((3, 64), device=inputs.device, dtype=torch.int64)
    bits = torch.empty((2, 64), device=inputs.device, dtype=torch.int32)
    _compare_exchange[(1, )](inputs, outputs, bits, AXIS=axis, enable_graph_optimize=True)
    left, right = pairs[:, 0], pairs[:, 1]
    largest = torch.maximum(left, right)
    expected = torch.stack((largest, torch.minimum(left, right), left ^ right))
    expected_bits = torch.stack((largest.to(torch.int32), (largest >> 32).to(torch.int32)))
    torch.testing.assert_close(outputs.cpu(), expected, rtol=0, atol=0, check_device=True)
    torch.testing.assert_close(bits.cpu(), expected_bits, rtol=0, atol=0, check_device=True)


@triton.jit
def _loop_and_boundaries(X, F, Y, TOTAL):
    pid = tl.program_id(0)
    offsets = pid * 128 + tl.arange(0, 128)
    original = tl.load(X + offsets)
    # Unknown runtime multiplier may be -1 or 2. General i64 multiplication
    # must remain exact at this boundary rather than applying the 0/1 rule.
    factor = tl.load(F + pid)
    values = original * factor
    for iteration in range(TOTAL):
        shaped = values.reshape(64, 2)
        mask = tl.arange(0, 2)[None, :].to(tl.int64)
        left = tl.sum(shaped * (1 - mask), 1)
        right = tl.sum(shaped * mask, 1)
        choose = left > right
        delta = tl.where(choose, left ^ right, 0)
        maximum = right ^ delta
        minimum = left ^ delta
        if iteration % 2 == 0:
            values = tl.join(maximum, minimum).reshape(128)
        else:
            values = tl.join(minimum, maximum).reshape(128)
    # This general sum shares the final values but can have a carry. It must
    # not receive the one-hot no-carry reduction transformation.
    total = tl.sum(values.reshape(64, 2), 1)
    cols = tl.arange(0, 64)
    tl.store(Y + pid * 192 + tl.arange(0, 128), values)
    tl.store(Y + pid * 192 + 128 + cols, total)


def _data(seed, width):
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 1 << 64, (64, width), dtype=np.uint64).view(np.int64)
    boundaries = np.array([
        0,
        1,
        -1,
        -(1 << 63),
        (1 << 63) - 1,
        (7 << 32) | 0x80000000,
        (7 << 32) | 0x7fffffff,
        -(1 << 32),
        -(1 << 32) - 1,
        (1 << 32) - 1,
        1 << 32,
    ], dtype=np.int64)
    values[0] = np.resize(boundaries, width)
    values[1].fill(-1)
    values[2].fill((1 << 63) - 1)
    return values


@pytest.mark.parametrize('iterations', [1, 2, 3, 5])
def test_loop_general_multiply_and_carry(iterations):
    for seed in range(10):
        original = _data(95000 + seed, 128)
        factors = np.resize(np.array([-1, 0, 1, 2], dtype=np.int64), 64)
        source = torch.from_numpy(original.copy()).to('npu')
        multiplier = torch.from_numpy(factors.copy()).to('npu')
        result = torch.empty((64, 192), device=source.device, dtype=torch.int64)
        _loop_and_boundaries[(64, )](source, multiplier, result, iterations, enable_graph_optimize=True)
        # uint64 reference makes modulo-2^64 wrap explicit and avoids Python
        # signed-overflow or dtype-promotion ambiguities.
        expected = (original.view(np.uint64) * factors.view(np.uint64)[:, None]).view(np.int64)
        pairs = expected.reshape(64, 64, 2)
        smaller, larger = np.minimum(pairs[..., 0], pairs[..., 1]), np.maximum(pairs[..., 0], pairs[..., 1])
        order = (larger, smaller) if iterations % 2 else (smaller, larger)
        expected_values = np.stack(order, axis=-1).reshape(64, 128)
        sums = expected_values.view(np.uint64).reshape(64, 64, 2).sum(axis=-1, dtype=np.uint64).view(np.int64)
        expected = np.concatenate((expected_values, sums), axis=1)
        actual = result.cpu().numpy()
        np.testing.assert_array_equal(actual, expected, err_msg=f'loops={iterations},seed={seed}')
