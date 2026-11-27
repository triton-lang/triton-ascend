"""Shared helpers for the extended SIMD unary/binary tests."""
import torch
import triton.language as tl

MASK_ROWS = 6
TAIL_BASE = MASK_ROWS
MASK_ROWS_C = tl.constexpr(MASK_ROWS)
TAIL_BASE_C = tl.constexpr(TAIL_BASE)


def masks(n, active):
    idx = torch.arange(n)
    thr = (torch.arange(n) % 5) * (n // 5)
    return {"prefix": idx < active, "arb": idx < thr, "none": torch.zeros(n, dtype=torch.bool)}, thr


def apply(value, mask):
    # torch.where on CPU is not implemented for the unsigned integer dtypes
    # (uint16/uint32/uint64); keep the bit pattern via int64 math instead.
    if value.dtype in (torch.uint16, torch.uint32, torch.uint64):
        return (value.to(torch.int64) * mask.to(torch.int64)).to(value.dtype)
    return torch.where(mask, value, torch.zeros((), dtype=value.dtype))


def ref_rows(value, mask_set):
    return torch.stack([
        value,
        apply(value, mask_set["prefix"]),
        apply(value, mask_set["arb"]),
        apply(value, mask_set["none"]),
        apply(value, mask_set["prefix"]),
        apply(value, mask_set["arb"]),
    ])


_SIGNED_TWIN = {torch.uint16: torch.int16, torch.uint32: torch.int32, torch.uint64: torch.int64}


def inactive_zero(got, mask_set):
    for row, name in ((1, "prefix"), (2, "arb"), (3, "none"), (4, "prefix"), (5, "arb")):
        keep = mask_set[name]
        # Boolean indexing has no CPU kernel for the unsigned dtypes; the bit
        # pattern is what we test, so read the row through its signed twin.
        values = got[row]
        twin = _SIGNED_TWIN.get(values.dtype)
        if twin is not None:
            values = values.view(twin)
        inactive = values[~keep] if keep.any() else values
        assert (inactive == 0).all(), f"row {row} ({name}) has a non-zero inactive lane"


def padded_unary(op, x, tiles, lanes, total, sentinel, dtype=None):
    out = torch.full((tiles, lanes), sentinel, dtype=dtype or x.dtype)
    for t in range(tiles):
        if t == tiles - 1:
            n = total - (tiles - 1) * lanes
            out[t, :n] = op(x[t][:n])
        else:
            out[t] = op(x[t])
    return out


def padded_binary(op, x, y, tiles, lanes, total, sentinel, dtype=None):
    out = torch.full((tiles, lanes), sentinel, dtype=dtype or x.dtype)
    for t in range(tiles):
        if t == tiles - 1:
            n = total - (tiles - 1) * lanes
            out[t, :n] = op(x[t][:n], y[t][:n])
        else:
            out[t] = op(x[t], y[t])
    return out
