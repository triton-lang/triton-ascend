# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Fold parked bin-zero counts without losing genuine zeros or other users."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _parking(X, M, O, N: tl.constexpr, BINS: tl.constexpr, PARK: tl.constexpr, WRONG_MASK: tl.constexpr,
             KEEP_RAW: tl.constexpr):
    i = tl.arange(0, N)
    mask = tl.load(M + i) != 0
    x = tl.load(X + i)
    hist = tl.histogram(tl.where(mask, x, PARK), BINS)
    counted_mask = mask if not WRONG_MASK else (i % 3 == 0)
    count = tl.sum((~counted_mask).to(tl.int32), 0)
    bins = tl.arange(0, BINS)
    corrected = hist - tl.where(bins == 0, count, 0)
    tl.store(O + bins, corrected)
    tl.store(O + BINS + bins, hist if KEEP_RAW else corrected)


def _check_parking(size, bins, park, wrong_mask, mask_kind, keep_raw=False):
    torch.manual_seed(37)
    x = torch.randint(-8, bins + 8, (size, ), dtype=torch.int32)
    x[::7] = 0
    mask = torch.randint(0, 2, (size, ), dtype=torch.int32)
    if mask_kind != "random":
        mask.fill_(int(mask_kind == "all"))
    device_x, device_mask = x.npu(), mask.npu()
    output = torch.empty((2, bins), dtype=torch.int32, device=device_x.device)
    _parking[(1, )](device_x, device_mask, output, size, bins, park, wrong_mask, keep_raw, enable_graph_optimize=True)
    parked = torch.where(mask.bool(), x, park)
    hist = torch.bincount(parked[(parked >= 0) & (parked < bins)].long(), minlength=bins)
    count_mask = torch.arange(size) % 3 == 0 if wrong_mask else mask.bool()
    expected = hist.clone()
    expected[0] -= (~count_mask).sum()
    second = hist if keep_raw else expected
    torch.testing.assert_close(output.cpu().long(), torch.stack((expected, second)), rtol=0, atol=0)


@pytest.mark.parametrize("size", [256, 8192])
@pytest.mark.parametrize("bins", [16, 256])
@pytest.mark.parametrize("park,wrong_mask", [(0, False), (1, False), (0, True)])
@pytest.mark.parametrize("mask_kind", ["random", "none", "all"])
def test_histogram_parking(size, bins, park, wrong_mask, mask_kind):
    _check_parking(size, bins, park, wrong_mask, mask_kind)


def test_histogram_parking_other_users():
    _check_parking(8192, 256, 0, False, "random", keep_raw=True)
