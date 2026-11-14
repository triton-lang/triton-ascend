# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
"""TopK kernels for Ascend 950, including radix selection and tiled reduction."""

import math
import threading

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import extension as al
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction
from triton.runtime import driver

simd = tla.simd
_tiled_launch = threading.local()


@triton.jit
def radix_topk(X, Values, Indices, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, LARGEST: tl.constexpr):
    """Write unsorted values and indices for FP32 (excluding NaNs) or int32 rows.

    Ties at the cutoff prefer the lower input index. Launch one program per row.
    """
    tl.static_assert(K > 0 and K <= N and N <= BLOCK)
    tl.static_assert(X.dtype.element_ty == tl.float32 or X.dtype.element_ty == tl.int32)
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    valid = lane < N
    x = tl.load(X + row * N + lane, valid, other=0)
    bits = x.to(tl.uint32, bitcast=True)
    if x.dtype == tl.float32:
        # Give signed zeros the same key and preserve the original output value.
        bits = tl.where(bits & 0x7FFFFFFF == 0, 0, bits)
        key = bits ^ tl.where(bits & 0x80000000 != 0, 0xFFFFFFFF, 0x80000000)
    else:
        key = bits ^ 0x80000000
    if not LARGEST:
        key = ~key

    bins = tl.arange(0, 256)
    prefix = tl.full((), 0, tl.uint32)
    rank = tl.full((), K, tl.int32)
    active = valid
    for digit in tl.static_range(3, -1, -1):
        byte = ((key >> (digit * 8)) & 255).to(tl.int32)
        freq = tl.histogram(byte, 256, mask=active)
        cumulative = tl.cumsum(freq)
        larger = tl.sum(freq, 0) - cumulative
        selected = tl.sum((larger >= rank).to(tl.int32), 0)
        rank -= tl.sum(tl.where(bins == selected, larger, 0), 0)
        prefix |= selected.to(tl.uint32) << (digit * 8)
        active &= byte == selected

    greater = valid & (key > prefix)
    equal = valid & (key == prefix)
    greater_rank = tl.cumsum(greater.to(tl.int32)) - 1
    equal_rank = tl.cumsum(equal.to(tl.int32)) - 1
    greater_count = tl.sum(greater.to(tl.int32), 0)
    keep = greater | (equal & (equal_rank < K - greater_count))
    pos = row * K + tl.where(greater, greater_rank, greater_count + equal_rank)
    tl.store(Values + pos, x, keep)
    tl.store(Indices + pos, lane, keep)


@TLAJITFunction
def native_topk(
    X,
    Values,
    Indices,
    ROWS,
    N: tl.constexpr,
    K: tl.constexpr,
    LARGEST: tl.constexpr,
    LAST_N: tl.constexpr = 0,
    WIDE_INDEX: tl.constexpr = False,
    Previous=None,
    TILE_INDICES: tl.constexpr = False,
    FinalValues=None,
    FinalIndices=None,
    TILES: tl.constexpr = 0,
):
    """Select each row and optionally merge the tiles in the same launch."""
    tl.static_assert(LAST_N == 0 or (K <= LAST_N and LAST_N < N))
    tl.static_assert(TILES * K <= 8192)
    dtype: tl.constexpr = X.dtype.element_ty
    tl.static_assert(dtype == tl.float32 or dtype == tl.float16 or dtype == tl.bfloat16 or dtype == tl.int32
                     or dtype == tl.int8 or dtype == tl.uint8)
    bits: tl.constexpr = dtype.primitive_bitwidth
    tl.static_assert(0 < K and K <= N and N <= 8192)
    floating: tl.constexpr = dtype.is_floating()
    narrow_keys: tl.constexpr = bits == 16
    input_lanes: tl.constexpr = 128 if narrow_keys else 64
    key_dtype: tl.constexpr = tl.uint16 if narrow_keys else tl.uint32
    output_lanes: tl.constexpr = 128 if bits == 16 else 64
    output_padded: tl.constexpr = ((K + output_lanes - 1) // output_lanes) * output_lanes
    top_digit: tl.constexpr = bits // 8 - 1

    widths: tl.constexpr = (N, TILES * K)
    tails: tl.constexpr = (LAST_N, 0)
    word_counts: tl.constexpr = (2 if WIDE_INDEX else 1, 2)
    padded_widths: tl.constexpr = (((N + 255) // 256) * 256, ((TILES * K + 255) // 256) * 256)
    compact_widths: tl.constexpr = (((N + 63) // 64) * 64, ((TILES * K + 63) // 64) * 64)

    # Large 1D inputs use the same row selector twice. All AIVs publish
    # tile candidates, then AIV 0 selects from that single candidate row.
    for phase in tl.static_range(2 if TILES else 1):
        if phase == 0:
            source, out_values, out_indices, previous = X, Values, Indices, Previous
            row_count, step = ROWS, tl.num_programs(0)
        else:
            source, out_values, out_indices, previous = Values, FinalValues, FinalIndices, Indices
            row_count, step = 1, 1
        for row in tl.range(tl.program_id(0), row_count, step):
            with tla.scope.vector():
                data = tla.alloc((padded_widths[phase], ), dtype, space="ub")
                keys = tla.alloc((padded_widths[phase], ), key_dtype, space="ub")
                if not narrow_keys:
                    bytes_by_digit = tla.alloc((bits // 8, padded_widths[phase]), tl.uint8, space="ub")
                picked = tla.alloc((output_padded, ), tl.uint32, space="ub")
                indices = tla.alloc((K * word_counts[phase], ), tl.int32, space="ub")
                values = tla.alloc((K, ), dtype, space="ub")
                if previous is not None:
                    source_indices = tla.alloc((padded_widths[phase], ), tl.int32, space="ub")
                if tails[phase]:
                    if row == ROWS - 1:
                        tla.load_gm_to_ub(tla.view(source, (tails[phase], ), offsets=(row * widths[phase], )),
                                          tla.subview(data, offsets=(0, ), sizes=(tails[phase], )))
                        if previous is not None:
                            tla.load_gm_to_ub(tla.view(previous, (tails[phase], ), offsets=(row * widths[phase], )),
                                              tla.subview(source_indices, offsets=(0, ), sizes=(tails[phase], )))
                    else:
                        tla.load_gm_to_ub(tla.view(source, (widths[phase], ), offsets=(row * widths[phase], )),
                                          tla.subview(data, offsets=(0, ), sizes=(widths[phase], )))
                        if previous is not None:
                            tla.load_gm_to_ub(tla.view(previous, (widths[phase], ), offsets=(row * widths[phase], )),
                                              tla.subview(source_indices, offsets=(0, ), sizes=(widths[phase], )))
                    row_n = tl.where(row == ROWS - 1, tails[phase], widths[phase])
                else:
                    tla.load_gm_to_ub(tla.view(source, (widths[phase], ), offsets=(row * widths[phase], )),
                                      tla.subview(data, offsets=(0, ), sizes=(widths[phase], )))
                    if previous is not None:
                        tla.load_gm_to_ub(tla.view(previous, (widths[phase], ), offsets=(row * widths[phase], )),
                                          tla.subview(source_indices, offsets=(0, ), sizes=(widths[phase], )))
                    row_n = widths[phase]

                with tla.scope.simd():
                    lanes = simd.vbitcast(simd.vci(0), "uint32")
                    zero32 = simd.vdup(0, tl.uint32)
                    one32 = simd.vdup(1, tl.uint32)
                    sign = simd.vdup(1 << (bits - 1), key_dtype)
                    ones = simd.vdup((1 << bits) - 1, key_dtype)
                    zero = simd.vdup(0, key_dtype)
                    # Map numeric order to unsigned key order. Keep the input
                    # values in their original dtype for the final gather.
                    for base in range(0, padded_widths[phase], input_lanes):
                        if bits == 8:
                            raw = simd.vunpack(simd.vlds(data[base], dist="UNPK4_B8"), tl.uint32)
                        elif narrow_keys:
                            raw = simd.vbitcast(simd.vlds(data[base]), "uint16")
                        else:
                            raw = simd.vbitcast(simd.vlds(data[base]), "uint32")
                        if floating:
                            magnitude = simd.vand(raw, simd.vdup((1 << (bits - 1)) - 1, key_dtype))
                            raw = simd.vsel(simd.vcmp(magnitude, zero), zero, raw)
                            flip = simd.vsel(simd.vcmp(raw, sign, op=simd.CMP.GE), ones, sign)
                            key = simd.vxor(raw, flip)
                        elif dtype == tl.uint8:
                            key = raw
                        else:
                            key = simd.vxor(raw, sign)
                        if not LARGEST:
                            key = simd.vxor(key, ones)
                        simd.vsts(keys[base], key, dist="NORM_B16" if narrow_keys else "NORM_B32")
                        if not narrow_keys:
                            for plane in tl.static_range(0, bits // 8):
                                component = simd.vshr(key, simd.vdup(plane * 8, tl.uint32))
                                simd.vsts(bytes_by_digit[plane, base], component, dist="PK4_B32")
                    simd.membar("VST_VLD")

                    zero16 = simd.vdup(0, tl.uint16)
                    one16 = simd.vdup(1, tl.uint16)
                    rank = simd.vdup(K, tl.uint16)
                    prefix = zero
                    lanes16, _ = simd.vdintlv(simd.vbitcast(lanes, "uint16"),
                                              simd.vbitcast(simd.vadds(lanes, 64), "uint16"))
                    if not narrow_keys:
                        lanes8, _ = simd.vdintlv(simd.vbitcast(lanes16, "uint8"),
                                                 simd.vbitcast(simd.vadds(lanes16, 128), "uint8"))
                        selected_bytes = ()
                    # Refine the cutoff from the most significant digit. Each
                    # pass sees only keys matching the preceding digits.
                    for digit in tl.static_range(top_digit, -1, -1):
                        hist_low = zero16
                        hist_high = zero16
                        if narrow_keys:
                            # Deinterleave data and validity in the same order so the
                            # histogram receives a predicate with one lane per byte.
                            for base in range(0, padded_widths[phase], 256):
                                bytes_pair = ()
                                flags = ()
                                for group in tl.static_range(2):
                                    offset = base + group * 128
                                    key = simd.vlds(keys[offset])
                                    if tails[phase] or widths[phase] % 256 or digit != top_digit:
                                        valid = simd.pset(16)
                                        if tails[phase] or widths[phase] % 256:
                                            valid = simd.vcmps(simd.vadds(lanes16, tla.cast(offset, tl.uint16)),
                                                               tla.cast(row_n, tl.uint16), op=simd.CMP.LT)
                                        if digit != top_digit:
                                            valid = simd.vcmp(simd.vand(key, simd.vdup(0xFF00, tl.uint16)), prefix,
                                                              mask=valid)
                                        flags += (simd.vbitcast(simd.vsel(valid, one16, zero16), "uint8"), )
                                    bytes_pair += (simd.vbitcast(key, "uint8"), )
                                low_bytes, high_bytes = simd.vdintlv(bytes_pair[0], bytes_pair[1])
                                packed_bytes = high_bytes if digit == top_digit else low_bytes
                                if tails[phase] or widths[phase] % 256 or digit != top_digit:
                                    packed_flags, _unused_flags = simd.vdintlv(flags[0], flags[1])
                                    active_bytes = simd.vcmps(packed_flags, 0, op=simd.CMP.NE)
                                else:
                                    active_bytes = simd.pset(8)
                                hist_low = simd.chistv2(hist_low, packed_bytes, active_bytes, 0)
                                hist_high = simd.chistv2(hist_high, packed_bytes, active_bytes, 1)
                        else:
                            # Each 256-lane load contains one byte per input element.
                            for base in range(0, row_n, 256):
                                last_lane = tla.cast(tl.minimum(255, row_n - base - 1), tl.uint8)
                                valid8 = simd.vcmps(lanes8, last_lane, op=simd.CMP.LE)
                                for prior in tl.static_range(top_digit, digit, -1):
                                    previous_byte = simd.vlds(bytes_by_digit[prior, base])
                                    valid8 = simd.vcmp(previous_byte, selected_bytes[top_digit - prior], mask=valid8)
                                digit_bytes = simd.vlds(bytes_by_digit[digit, base])
                                hist_low = simd.chistv2(hist_low, digit_bytes, valid8, 0)
                                hist_high = simd.chistv2(hist_high, digit_bytes, valid8, 1)
                        total = simd.vdupv(hist_high, pos=simd.POS.HIGHEST)
                        # Convert the descending rank to an ascending ordinal.
                        # The first cumulative bucket reaching it is the cutoff.
                        target = simd.vadd(simd.vsub(total, rank), one16)
                        below_lo = simd.vsel(simd.vcmp(hist_low, target, op=simd.CMP.LT), one16, zero16)
                        below_hi = simd.vsel(simd.vcmp(hist_high, target, op=simd.CMP.LT), one16, zero16)
                        bucket = simd.vdupv(simd.vreduce(simd.vadd(below_lo, below_hi), "add"))
                        lane16 = simd.vand(bucket, simd.vdup(127, tl.uint16))
                        count = simd.vsel(simd.vcmps(bucket, 128, op=simd.CMP.GE), simd.vselr(hist_high, lane16),
                                          simd.vselr(hist_low, lane16))
                        rank = simd.vsub(rank, simd.vsub(total, count))
                        digit_value = bucket if narrow_keys else simd.vunpack(bucket, tl.uint32)
                        prefix = simd.vor(prefix, simd.vshls(digit_value, digit * 8))
                        if not narrow_keys and digit > 0:
                            repetitions = simd.vmul(bucket, simd.vdup(257, tl.uint16))
                            selected_bytes += (simd.vbitcast(repetitions, "uint8"), )

                    cutoff = simd.vunpack(prefix, tl.uint32) if narrow_keys else prefix
                    strict_end = simd.vsub(simd.vdup(K, tl.uint32), simd.vunpack(rank, tl.uint32))
                    # Pack keys above the cutoff first. Reserve the remaining
                    # slots for the earliest equal keys in this row.
                    strict_pos = zero32
                    equal_pos = strict_end
                    for base in range(0, compact_widths[phase], 64):
                        if narrow_keys:
                            key = simd.vunpack(simd.vlds(keys[base], dist="UNPK_B16"), tl.uint32)
                        else:
                            key = simd.vlds(keys[base])
                        index = simd.vadds(lanes, tla.cast(base, tl.uint32))
                        valid = simd.pset(32)
                        if tails[phase] or widths[phase] % 64:
                            valid = simd.vcmps(index, row_n, op=simd.CMP.LT)
                        greater = simd.vcmp(key, cutoff, mask=valid, op=simd.CMP.GT)
                        equal = simd.vcmp(key, cutoff, mask=valid)
                        greater_count = simd.vdupv(simd.vreduce(one32, "add", mask=greater))
                        equal_count = simd.vdupv(simd.vreduce(one32, "add", mask=equal))
                        strict_offsets = simd.vadd(lanes, strict_pos)
                        equal_offsets = simd.vadd(lanes, equal_pos)
                        strict_mask = simd.vcmp(lanes, greater_count, op=simd.CMP.LT)
                        equal_mask = simd.vcmps(equal_offsets, K, mask=simd.vcmp(lanes, equal_count, op=simd.CMP.LT),
                                                op=simd.CMP.LT)
                        simd.vscatter(picked, strict_offsets, simd.vsqz(index, greater), strict_mask)
                        simd.vscatter(picked, equal_offsets, simd.vsqz(index, equal), equal_mask)
                        strict_pos = simd.vadd(strict_pos, greater_count)
                        equal_pos = simd.vadd(equal_pos, equal_count)

                    simd.membar("VST_VLD")
                    # Gather original values and translate candidate indices
                    # back to positions in the original input during merging.
                    if bits == 16:
                        for start in range(0, K, 128):
                            first = simd.vbitcast(simd.vlds(picked[start]), "uint16")
                            second = simd.vbitcast(simd.vlds(picked[start + 64]), "uint16")
                            selection, _ = simd.vdintlv(first, second)
                            out_mask = simd.vcmps(simd.vadds(lanes16, tla.cast(start, tl.uint16)), K, op=simd.CMP.LT)
                            simd.vsts(values[start], simd.vgather(data, selection, out_mask), dist="NORM_B16",
                                      mask=out_mask)

                    for start in range(0, K, 64):
                        out_mask = simd.vcmps(simd.vadds(lanes, tla.cast(start, tl.uint32)), K, op=simd.CMP.LT)
                        selection = simd.vlds(picked[start])
                        if bits == 32:
                            simd.vsts(values[start], simd.vgather(data, selection, out_mask), mask=out_mask)
                        elif bits == 8:
                            raw = simd.vgather(keys, selection, out_mask)
                            if not LARGEST:
                                raw = simd.vxor(raw, ones)
                            if dtype == tl.int8:
                                raw = simd.vxor(raw, sign)
                            simd.vsts(values[start], raw, dist="PK4_B32", mask=out_mask)
                        if previous is not None:
                            selection = simd.vbitcast(simd.vgather(source_indices, selection, out_mask), "uint32")
                        elif phase == 0 and TILE_INDICES:
                            selection = simd.vadds(selection, tla.cast(row * widths[phase], tl.uint32))
                        if word_counts[phase] == 2:
                            # Store each nonnegative index as two consecutive 32-bit words.
                            low, high = simd.vintlv(selection, zero32)
                            positions = simd.vadds(lanes, tla.cast(start * 2, tl.uint32))
                            low_mask = simd.vcmps(positions, K * 2, op=simd.CMP.LT)
                            high_mask = simd.vcmps(simd.vadds(positions, 64), K * 2, op=simd.CMP.LT)
                            simd.vsts(indices[start * 2], simd.vbitcast(low, "int32"), mask=low_mask)
                            simd.vsts(indices[start * 2 + 64], simd.vbitcast(high, "int32"), mask=high_mask)
                        else:
                            simd.vsts(indices[start], simd.vbitcast(selection, "int32"), mask=out_mask)

                tla.store_ub_to_gm(values, tla.view(out_values, (K, ), offsets=(row * K, )))
                tla.store_ub_to_gm(
                    indices, tla.view(out_indices, (K * word_counts[phase], ),
                                      offsets=(row * K * word_counts[phase], )))

        if TILES and phase == 0:
            tla.sync.sync_all(False)


@triton.jit
def _full_topk(X, V, I, SIZE: tl.constexpr, N: tl.constexpr, INNER: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < SIZE
    value = tl.load(X + offset, valid, other=0)
    tl.store(V + offset, value, valid)
    tl.store(I + offset, offset // INNER % N, valid)


def _native_launch(x, rows, n, k, largest, cores, last_n=0, wide_index=False, previous=None, tile_indices=False,
                   shape=None):
    shape = (rows, k) if shape is None else shape
    values = x.new_empty(shape)
    indices = x.new_empty(shape, dtype=torch.int64 if wide_index else torch.int32)
    storage = triton.reinterpret(indices, tl.int32) if wide_index else indices
    native_topk[(min(rows, cores), )](x, values, storage, rows, n, k, largest, last_n, wide_index, previous,
                                      tile_indices)
    return values, indices


def _tiled_topk(x, k, largest, cores, chunk):
    """Select K candidates per tile and propagate their original indices through each merge."""
    n = x.numel()
    rows = triton.cdiv(n, chunk)
    values, indices = _native_launch(x, rows, chunk, k, largest, cores, n % chunk, wide_index=rows == 1,
                                     tile_indices=True)
    while rows > 1:
        tiles_per_group = min(rows, 8192 // k)
        width = tiles_per_group * k
        tail = values.numel() % width
        rows = triton.cdiv(rows, tiles_per_group)
        values, indices = _native_launch(values, rows, width, k, largest, cores, tail, wide_index=rows == 1,
                                         previous=indices)
    return values.flatten(), indices.flatten()


@triton.jit
def _restore_output(Values, Indices, V, I, K: tl.constexpr, INNER: tl.constexpr, ROW_TILE: tl.constexpr,
                    K_TILE: tl.constexpr, WRITE_VALUES: tl.constexpr):
    groups = tl.cdiv(INNER, ROW_TILE)
    tile = tl.program_id(0)
    outer = tile // groups
    row = tile % groups * ROW_TILE + tl.arange(0, ROW_TILE)
    col = tl.program_id(1) * K_TILE + tl.arange(0, K_TILE)
    source = (outer * INNER + row[:, None]) * K + col[None, :]
    valid = (row[:, None] < INNER) & (col[None, :] < K)
    index = tl.load(Indices + source, valid, other=0)
    destination = (outer * K + col[:, None]) * INNER + row[None, :]
    if WRITE_VALUES:
        value = tl.load(Values + source, valid, other=0)
        tl.store(V + destination, tl.trans(value), tl.trans(valid))
    tl.store(I + destination, tl.trans(index), tl.trans(valid))


def _restore_layout(x, values, indices, k, axis, fused):
    inner = math.prod(x.shape[axis + 1:])
    outer = values.shape[0] // inner
    shape = list(x.shape)
    shape[axis] = k
    out_indices = x.new_empty(shape, dtype=torch.int64)
    if fused:
        out = x.new_empty(shape)
    else:
        # The large strided path transposes values separately and casts indices
        # as part of their transpose, avoiding a full int64 intermediate.
        row_shape = (*x.movedim(axis, -1).shape[:-1], k)
        out = values.view(row_shape).movedim(-1, axis).to(x.dtype).contiguous()
    row_tile, k_tile = (64, 16) if fused and inner % 32 else (32, 32) if fused else (64, 64)
    row_tile = min(row_tile, triton.next_power_of_2(inner))
    k_tile = min(k_tile, triton.next_power_of_2(k))
    grid = (outer * triton.cdiv(inner, row_tile), triton.cdiv(k, k_tile))
    _restore_output[grid](values, indices, out, out_indices, k, inner, row_tile, k_tile, fused)
    return out, out_indices


def topk(x, k, dim=-1, largest=True, sorted=True, *, implementation="auto", chunk=4096, bounded_int64=False,
         fused_output=None):
    """Select K values and their int64 indices, ordered by value by default."""
    if not -x.ndim <= dim < x.ndim or not x.is_contiguous():
        raise ValueError("expected a valid axis and contiguous input")
    axis = dim % x.ndim
    n = x.shape[axis]
    if not 0 < k <= n:
        raise ValueError("expected 0 < k <= axis length")
    if implementation == "auto":
        implementation = "full" if k == n else "tiles" if n > 8192 else "radix"
    if implementation == "full" and k != n:
        raise ValueError("full selection requires k == n")
    if implementation == "sort" or (sorted and implementation == "full"):
        return _sort_output(x, None, k, axis, largest, bounded_int64)
    if x.dtype == torch.int64 and not bounded_int64 and implementation == "radix":
        return _sort_output(x, None, k, axis, largest)
    values, indices = _select_topk(x, k, axis, largest, implementation, chunk, bounded_int64, fused_output)
    if sorted and k > 1:
        return _sort_output(values, indices, k, axis, largest, bounded_int64)
    return values, indices


def _select_topk(x, k, axis, largest, implementation, chunk, bounded_int64, fused_output):
    n = x.shape[axis]
    if implementation == "full":
        values = torch.empty_like(x)
        indices = torch.empty_like(x, dtype=torch.int64)
        block = 2048
        _full_topk[(triton.cdiv(x.numel(), block), )](x, values, indices, x.numel(), n, math.prod(x.shape[axis + 1:]),
                                                      block)
        return values, indices

    if implementation == "byte":
        if axis != x.ndim - 1 or x.dtype not in (torch.uint8, torch.int8):
            raise ValueError("byte selection requires contiguous int8 or uint8 rows")
        shape = (*x.shape[:-1], k)
        values = x.new_empty(shape)
        indices = x.new_empty(shape, dtype=torch.int64)
        _byte_topk[(x.numel() // n, )](x, values, indices, n, k, triton.next_power_of_2(n), largest)
        return values, indices

    if implementation == "tiles":
        if x.ndim != 1 or n >= 2**31 or k > 4096 or not k <= chunk <= 8192 or 0 < n % chunk < k:
            raise ValueError("tiled selection requires 1D input, int32 offsets and a merge of at least two tiles")
        rows = triton.cdiv(n, chunk)
        cores = driver.active.utils.get_aivector_core_num()
        if x.element_size() == 2 and 1 < rows and rows * k <= 8192:
            device = x.device
            stream = driver.active.get_current_stream(device.index)
            capturing = torch.npu.is_current_stream_capturing()
            # Contiguous slices can start at an unaligned address. Keep their
            # specialization separate from the regular allocator's outputs.
            key = (device.index, stream, x.dtype, n, chunk, k, largest, cores, x.data_ptr() % 16 == 0)
            cached = getattr(_tiled_launch, "entry", None)
            values = x.new_empty((k, ))
            indices = x.new_empty((k, ), dtype=torch.int64)
            if not capturing and cached is not None and cached[0] == key:
                _, partial_values, partial_indices, launch = cached
            else:
                partial_values = x.new_empty((rows, k))
                partial_indices = x.new_empty((rows, k), dtype=torch.int32)
                launch = None

            # Launch only resident blocks. Every block must reach the barrier.
            grid = (min(rows, cores), 1, 1)
            args = (x, partial_values, partial_indices, rows, chunk, k, largest, n % chunk, False, None, True, values,
                    triton.reinterpret(indices, tl.int32), rows)
            if launch is None:
                compiled = native_topk[grid](*args)
                if not capturing and compiled is not None:
                    # Keep one compiled launch and workspace per thread, keyed by stream.
                    _tiled_launch.entry = (key, partial_values, partial_indices, compiled[grid])
            else:
                launch(*args, stream=stream)
            return values, indices
        if x.dtype == torch.int64:
            if not bounded_int64:
                raise ValueError("int64 radix requires the caller to guarantee the int32 range")
            values, indices = _tiled_topk(x.to(torch.int32), k, largest, cores, chunk)
            return values.to(x.dtype), indices
        return _tiled_topk(x, k, largest, cores, chunk)

    cores = driver.active.utils.get_aivector_core_num()
    if x.dtype == torch.int64 and not bounded_int64:
        raise ValueError("int64 radix requires the caller to guarantee the int32 range")
    dtype = torch.int32 if x.dtype == torch.int64 else x.dtype
    inner = math.prod(x.shape[axis + 1:])
    if fused_output is None:
        # Fuse output conversion for small strided floating-point outputs.
        fused_output = inner >= 32 and 32 <= k <= 128 and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if implementation == "radix" and axis == x.ndim - 1 and not fused_output and n <= 8192 and x.dtype == dtype:
        return _native_launch(x, x.numel() // n, n, k, largest, cores, wide_index=True, shape=(*x.shape[:-1], k))
    if implementation != "radix" or n > 8192:
        raise ValueError(f"unsupported implementation for this axis: {implementation}")
    rows = x.movedim(axis, -1).to(dtype).contiguous()
    values, indices = _native_launch(rows,
                                     rows.numel() // n, n, k, largest, cores, wide_index=inner == 1
                                     and not fused_output)
    if fused_output:
        return _restore_layout(x, values, indices, k, axis, fused=True)
    if inner % 64 == 0 and k % 64 == 0 and indices.numel() >= 1024**2:
        return _restore_layout(x, values, indices, k, axis, fused=False)
    shape = (*rows.shape[:-1], k)
    return (values.view(shape).movedim(-1, axis).to(x.dtype).contiguous(),
            indices.view(shape).movedim(-1, axis).to(torch.int64).contiguous())


@triton.jit
def _byte_topk(X, V, I, N: tl.constexpr, K: tl.constexpr, WIDTH: tl.constexpr, LARGEST: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, WIDTH)
    valid = lane < N
    x = tl.load(X + row * N + lane, valid, other=0)
    key = x.to(tl.int32)
    if X.dtype.element_ty == tl.int8:
        key += 128
    if not LARGEST:
        key = 255 - key
    cumulative = tl.cumsum(tl.histogram(key, 256, mask=valid))
    bucket = tl.sum((cumulative <= N - K).to(tl.int32), 0)
    greater = valid & (key > bucket)
    equal = valid & (key == bucket)
    greater_count = tl.sum(greater.to(tl.int32), 0)
    equal_pos = tl.cumsum(equal.to(tl.int32)) - 1
    pos = tl.where(greater, tl.cumsum(greater.to(tl.int32)) - 1, greater_count + equal_pos)
    keep = greater | (equal & (equal_pos < K - greater_count))
    tl.store(V + row * K + pos, x, keep)
    tl.store(I + row * K + pos, lane, keep)


def _sort_output(values, indices, k, axis, largest, bounded_int64=False):
    shape = list(values.shape)
    n = shape[axis]
    shape[axis] = k
    out_values = values.new_empty(shape)
    out_indices = values.new_empty(shape, dtype=torch.int64)
    if not values.numel():
        return out_values, out_indices
    inner = math.prod(values.shape[axis + 1:])
    rows = values.numel() // n
    width = triton.next_power_of_2(n)
    if values.dtype == torch.int64 and not bounded_int64 and indices is None and width > 1024:
        rows_view = values.movedim(axis, -1).contiguous().view(rows, n)
        ordered_values, ordered_indices = _sort_wide_rows(rows_view, k, largest)
        row_shape = (*values.movedim(axis, -1).shape[:-1], k)
        return (ordered_values.view(row_shape).movedim(-1, axis).contiguous(),
                ordered_indices.view(row_shape).movedim(-1, axis).contiguous())

    rows_per_block = min(32, max(1, 2048 // width))
    if rows == 1 or (values.dtype == torch.int64 and not bounded_int64):
        rows_per_block = 1
    if width <= 32 and (values.dtype != torch.int64 or bounded_int64):
        _rank_order[(triton.cdiv(rows, rows_per_block), )](values, indices, out_values, out_indices, n, k, inner, rows,
                                                           width, rows_per_block, largest)
        return out_values, out_indices
    _order_topk[(triton.cdiv(rows, rows_per_block), )](values, indices, out_values, out_indices, n, k, inner, rows,
                                                       width, rows_per_block, largest, bounded_int64)
    return out_values, out_indices


def _sort_wide_rows(values, k, largest):
    rows, n = values.shape
    tile = 1024
    keep = min(k, tile)
    tiles = triton.cdiv(n, tile)
    out_values = values.new_empty((rows, tiles, keep))
    out_indices = values.new_empty((rows, tiles, keep), dtype=torch.int64)
    _sort_int64_tiles[(rows * tiles, )](values, None, out_values, out_indices, tiles * keep, n, tile, keep, largest)

    while tiles > 1:
        input_width = keep
        keep = min(k, 2 * keep)
        groups = triton.cdiv(tiles, 2)
        new_values = values.new_empty((rows, groups, keep))
        new_indices = values.new_empty((rows, groups, keep), dtype=torch.int64)
        _sort_int64_tiles[(rows * groups, )](out_values, out_indices, new_values, new_indices, groups * keep,
                                             tiles * input_width, 2 * input_width, keep, largest)
        out_values, out_indices, tiles = new_values, new_indices, groups
    return out_values.view(rows, k), out_indices.view(rows, k)


@triton.jit
def _sort_int64_tiles(X, Indices, V, I, OUT_WIDTH: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr, K: tl.constexpr,
                      LARGEST: tl.constexpr):
    tiles: tl.constexpr = triton.cdiv(N, TILE)
    block = tl.program_id(0)
    row = block // tiles
    tile = block % tiles
    width: tl.constexpr = triton.next_power_of_2(TILE)
    lanes = tl.arange(0, width)
    local = tile * TILE + lanes
    valid = (lanes < TILE) & (local < N)
    offsets = row * N + local
    x = tl.load(X + offsets, valid, other=0)
    index = local if Indices is None else tl.load(Indices + offsets, valid, other=-1)
    order = _order_int64(x, valid & (index >= 0), lanes, width, LARGEST)
    source = row * N + tile * TILE + width - order
    out_mask = lanes < K
    valid_out = out_mask & (order > 0)
    value = tl.load(X + source, valid_out, other=0)
    if Indices is None:
        original = tl.where(valid_out, source - row * N, -1)
    else:
        original = tl.load(Indices + source, valid_out, other=-1)
    dst = row * OUT_WIDTH + tile * K + lanes
    tl.store(V + dst, value, out_mask)
    tl.store(I + dst, original, out_mask)


@triton.jit
def _order_int64(x, valid, lane, WIDTH: tl.constexpr, LARGEST: tl.constexpr):
    # Sort the low word first. Its sorted rank breaks ties in the high word.
    bits: tl.constexpr = WIDTH.bit_length()
    suffix_mask: tl.constexpr = (1 << bits) - 1
    key = x if LARGEST else ~x
    low = key.to(tl.uint32).to(tl.int64)
    high = (key >> 32) + 0x80000000
    low_pair = (low << bits) | (WIDTH - lane).to(tl.int64)
    ordered_low = al.sort(tl.where(valid, low_pair, 0), descending=True)
    low_order = (ordered_low & suffix_mask).to(tl.int32)
    source = tl.maximum(0, WIDTH - low_order - (low_order == 0).to(tl.int32))
    high = tl.gather(high, source, axis=0)
    high_pair = (high << bits) | (WIDTH - lane).to(tl.int64)
    ordered_high = al.sort(tl.where(low_order > 0, high_pair, 0), descending=True)
    high_order = (ordered_high & suffix_mask).to(tl.int32)
    source = tl.maximum(0, WIDTH - high_order - (high_order == 0).to(tl.int32))
    return tl.where(high_order > 0, tl.gather(low_order, source, axis=0), 0)


@triton.jit
def _rank_order(X, Indices, V, I, N: tl.constexpr, K: tl.constexpr, INNER: tl.constexpr, ROWS: tl.constexpr,
                WIDTH: tl.constexpr, BATCH: tl.constexpr, LARGEST: tl.constexpr):
    row = tl.program_id(0) * BATCH + tl.arange(0, BATCH)
    lane = tl.arange(0, WIDTH)
    base = row // INNER * N * INNER + row % INNER
    source = base[:, None] + lane[None, :] * INNER
    valid = (row[:, None] < ROWS) & (lane[None, :] < N)
    value = tl.load(X + source, valid, other=0)
    key = _sort_key(value.to(tl.int32) if X.dtype.element_ty == tl.int64 else value)
    if not LARGEST:
        key = ~key
    current = key[:, :, None]
    other = key[:, None, :]
    earlier = lane[None, None, :] < lane[None, :, None]
    ahead = (other > current) | ((other == current) & earlier)
    rank = tl.sum((ahead & (lane[None, None, :] < N)).to(tl.int32), axis=2)
    output_mask = valid & (rank < K)
    index = lane[None, :] if Indices is None else tl.load(Indices + source, valid, other=0)
    destination = row[:, None] // INNER * K * INNER + row[:, None] % INNER + rank * INNER
    tl.store(V + destination, value, output_mask)
    tl.store(I + destination, index, output_mask)


@triton.jit
def _order_topk(X, Indices, V, I, N: tl.constexpr, K: tl.constexpr, INNER: tl.constexpr, ROWS: tl.constexpr,
                WIDTH: tl.constexpr, BATCH: tl.constexpr, LARGEST: tl.constexpr, BOUNDED_INT64: tl.constexpr):
    lane = tl.arange(0, WIDTH)
    if X.dtype.element_ty == tl.int64 and not BOUNDED_INT64:
        row = tl.program_id(0)
        base = row // INNER * N * INNER + row % INNER
        x = tl.load(X + base + lane * INNER, lane < N, other=0)
        local = WIDTH - _order_int64(x, lane < N, lane, WIDTH, LARGEST)
        out = lane
        valid = out < K
    else:
        row = tl.program_id(0) * BATCH + tl.arange(0, BATCH)
        row = row[:, None]
        base = row // INNER * N * INNER + row % INNER
        lane = lane[None, :]
        x = tl.load(X + base + lane * INNER, (row < ROWS) & (lane < N), other=0)
        key = _sort_key(x.to(tl.int32) if BOUNDED_INT64 else x)
        if not LARGEST:
            key = ~key
        index_bits: tl.constexpr = WIDTH.bit_length()
        if X.dtype.element_ty.primitive_bitwidth <= 16 and index_bits <= 15:
            pair = ((key & 0xFFFF).to(tl.int32) << index_bits) | (WIDTH - lane)
        else:
            pair = (key.to(tl.int64) << index_bits) | (WIDTH - lane).to(tl.int64)
        selected: tl.constexpr = triton.next_power_of_2(K)
        ordered = al.sort(tl.where(lane < N, pair, 0), descending=True)
        ordered = tl.gather(ordered, tl.broadcast_to(tl.arange(0, selected)[None, :], (BATCH, selected)), axis=1)
        local = WIDTH - (ordered & ((1 << index_bits) - 1)).to(tl.int32)
        out = tl.arange(0, selected)[None, :]
        valid = (row < ROWS) & (out < K)
    source = base + local * INNER
    value = tl.load(X + source, valid, other=0)
    index = local if Indices is None else tl.load(Indices + source, valid, other=0)
    destination = row // INNER * K * INNER + row % INNER + out * INNER
    tl.store(V + destination, value, valid)
    tl.store(I + destination, index, valid)


@triton.jit
def _sort_key(value):
    if value.dtype == tl.float32:
        bits = value.to(tl.uint32, bitcast=True)
        return bits ^ tl.where(bits & 0x80000000 != 0, 0xFFFFFFFF, 0x80000000)
    if value.dtype == tl.float16 or value.dtype == tl.bfloat16:
        bits = value.to(tl.uint16, bitcast=True).to(tl.uint32)
        return bits ^ tl.where(bits & 0x8000 != 0, 0xFFFF, 0x8000)
    if value.dtype == tl.uint8:
        return value.to(tl.uint32)
    if value.dtype == tl.int8:
        return value.to(tl.uint8).to(tl.uint32) ^ 0x80
    return value.to(tl.uint32) ^ 0x80000000
