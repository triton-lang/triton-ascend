# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the MIT license.

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'])
def _diagonal_tile_store(A, T, H: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, NT_OFFSET, NC_OFFSET, BH_OFFSET):
    t = tl.program_id(0).to(tl.int64) + NT_OFFSET
    i = tl.program_id(1) + NC_OFFSET
    bh = tl.program_id(2).to(tl.int64) + BH_OFFSET
    b, h = bh // H, bh % H
    r = tl.arange(0, BC)
    row = t * BT + i * BC + r
    col = i * BC + r
    ptr = A + ((b * T + row[:, None]) * H + h) * BT + col[None, :]
    mask = row[:, None] < T
    tl.store(ptr, 3.0, mask=mask)
    tl.debug_barrier()
    tl.store(ptr, 7.0, mask=mask & (r[:, None] < r[None, :]))


@pytest.mark.parametrize('B,T,H,offsets', [
    (1, 63, 1, (0, 0, 0)),
    (2, 64, 2, (0, 0, 0)),
    (2, 64, 2, (0, 1, 1)),
    (2, 128, 3, (0, 0, 0)),
    (2, 128, 3, (0, 1, 1)),
    (2, 128, 3, (1, 0, 0)),
])
def test_diagonal_tile_ownership(B, T, H, offsets):
    bt, bc = 64, 16
    ot, oi, obh = offsets
    grid = (triton.cdiv(T, bt) - ot, bt // bc - oi, B * H - obh)
    output = torch.full((B, T, H, bt), -13.0, device='npu')
    expected = output.clone()
    for t in range(ot, triton.cdiv(T, bt)):
        for i in range(oi, bt // bc):
            lo, hi = t * bt + i * bc, min(T, t * bt + (i + 1) * bc)
            for bh in range(obh, B * H):
                b, h = divmod(bh, H)
                if lo >= hi:
                    continue
                expected[b, lo:hi, h, i * bc:(i + 1) * bc] = 3.0
                for r in range(hi - lo):
                    expected[b, lo + r, h, i * bc + r + 1:(i + 1) * bc] = 7.0
    for _ in range(5):
        output.fill_(-13.0)
        _diagonal_tile_store[grid](output, T, H, bt, bc, ot, oi, obh, compile_mode='simd')
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'])
def _masked_update(X, A, T, H: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, NT_OFFSET, NC_OFFSET, BH_OFFSET):
    t = tl.program_id(0).to(tl.int64) + NT_OFFSET
    i = tl.program_id(1) + NC_OFFSET
    bh = tl.program_id(2).to(tl.int64) + BH_OFFSET
    b, h = bh // H, bh % H
    lane = tl.arange(0, BC)
    row = t * BT + i * BC + lane
    base = ((b * T + row[:, None]) * H + h) * BT
    data = tl.load(X + base + tl.arange(0, BT)[None, :])
    value = tl.sum(data, 1)[:, None] + lane[None, :] + 7.0
    ptr = A + base + i * BC + lane[None, :]
    tl.store(ptr, value, mask=lane[:, None] < lane[None, :])


@pytest.mark.parametrize('alias', [False, True])
@pytest.mark.parametrize('misaligned', [False, True])
def test_pointer_alias_guard(alias, misaligned):
    shape = (2, 64, 2, 64)
    size = 2 * 64 * 2 * 64
    start = int(misaligned)
    source = torch.arange(size + start, dtype=torch.float32, device='npu')[start:].view(shape)
    output = source if alias else torch.full((size + start, ), -13.0, device='npu')[start:].view(shape)
    expected = output.clone()
    sums = source.sum(-1)
    for i in range(4):
        for lane in range(16):
            row = i * 16 + lane
            columns = torch.arange(lane + 1, 16, device='npu')
            expected[:, row, :, i * 16 + lane + 1:(i + 1) * 16] = sums[:, row, :, None] + columns + 7.0
    _masked_update[(1, 4, 4)](source, output, 64, 2, 64, 16, 0, 0, 0, compile_mode='simd')
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@triton.jit
def _overlapping_windows(A, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    pid = tl.program_id(0).to(tl.int64)
    # The old heuristic mistakes this value expression for an address proof.
    value = pid * BLOCK + lane
    address = pid * (BLOCK // 2) + lane
    tl.store(A + address, value.to(tl.float32), mask=(lane % 2) == (pid % 2))


def test_overlapping_rmw_windows():
    # Selected lanes are disjoint; the full RMW windows overlap by half a tile.
    programs, block = 8, 64
    size = (programs + 1) * block // 2
    expected = torch.full((size, ), -13.0)
    for pid in range(programs):
        for lane in range(block):
            if lane % 2 == pid % 2:
                expected[pid * (block // 2) + lane] = pid * block + lane
    expected = expected.to('npu')
    output = torch.empty_like(expected)
    for _ in range(20):
        output.fill_(-13.0)
        _overlapping_windows[(programs, )](output, block, compile_mode='simd')
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
