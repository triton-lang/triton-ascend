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

import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from triton.runtime import driver

simd = tla.simd


@tla.jit
def _sort_proposals(scores, positions, proposals, workspace, width: tl.constexpr, backend: tl.constexpr):
    if backend == "template":
        tla.sort_pairs(scores, positions, proposals, workspace)
    else:
        rounds: tl.constexpr = (width.bit_length() - 5) // 2
        if rounds % 2 == 0:
            simd.vbs(scores, positions, proposals)
        else:
            simd.vbs(scores, positions, workspace)
        for stage in tl.static_range(rounds):
            if (rounds - stage) % 2 == 0:
                simd.vmrgsort4(proposals, workspace, 32 * (4**stage))
            else:
                simd.vmrgsort4(workspace, proposals, 32 * (4**stage))


@tla.jit
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
    SORT_BACKEND: tl.constexpr = None,
    DIRECT: tl.constexpr = False,
    TILED_WIDE_INDEX: tl.constexpr = True,
):
    """Select each row and optionally merge the tiles in the same launch."""
    tl.static_assert(LAST_N == 0 or (K <= LAST_N and LAST_N < N))
    tl.static_assert(TILES * K <= 8192)
    tl.static_assert(not DIRECT or (SORT_BACKEND is not None and not LAST_N))
    tl.static_assert(SORT_BACKEND is None or (not TILES and not TILE_INDICES))
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
    direct_width: tl.constexpr = max(128, triton.next_power_of_2(N if DIRECT else K))
    sort_count: tl.constexpr = N if DIRECT else K

    widths: tl.constexpr = (N, TILES * K)
    tails: tl.constexpr = (LAST_N, 0)
    word_counts: tl.constexpr = (2 if WIDE_INDEX else 1, 2 if TILED_WIDE_INDEX else 1)
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
        if TILES:
            row_start = tl.program_id(0)
            row_end = row_count
            row_step = step
        else:
            rows_per_core = tla.cdiv(row_count, step)
            row_start = tl.program_id(0) * rows_per_core
            row_end = tla.minimum(row_count, row_start + rows_per_core)
            row_step = 1
        for row in tl.range(row_start, row_end, row_step):
            with tla.scope.vector():
                data = tla.alloc((padded_widths[phase], ), dtype, space="ub")
                keys = tla.alloc((padded_widths[phase], ), key_dtype, space="ub")
                if not narrow_keys:
                    bytes_by_digit = tla.alloc((bits // 8, padded_widths[phase]), tl.uint8, space="ub")
                picked_width: tl.constexpr = output_padded if SORT_BACKEND is None else direct_width
                picked = tla.alloc((picked_width, ), tl.uint32, space="ub")
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
                    row_n = tla.select(row == ROWS - 1, tails[phase], widths[phase])
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
                    if not DIRECT or bits == 8:
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
                    if not DIRECT:
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
                                    last_lane = tla.cast(tla.minimum(255, row_n - base - 1), tl.uint8)
                                    valid8 = simd.vcmps(lanes8, last_lane, op=simd.CMP.LE)
                                    for prior in tl.static_range(top_digit, digit, -1):
                                        previous_byte = simd.vlds(bytes_by_digit[prior, base])
                                        valid8 = simd.vcmp(previous_byte, selected_bytes[top_digit - prior],
                                                           mask=valid8)
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
                            equal_mask = simd.vcmps(equal_offsets, K,
                                                    mask=simd.vcmp(lanes, equal_count, op=simd.CMP.LT), op=simd.CMP.LT)
                            simd.vscatter(picked, strict_offsets, simd.vsqz(index, greater), strict_mask)
                            simd.vscatter(picked, equal_offsets, simd.vsqz(index, equal), equal_mask)
                            strict_pos = simd.vadd(strict_pos, greater_count)
                            equal_pos = simd.vadd(equal_pos, equal_count)

                        simd.membar("VST_VLD")

                    if SORT_BACKEND is None:
                        # Gather original values and translate candidate indices
                        # back to positions in the original input during merging.
                        if bits == 16:
                            for start in range(0, K, 128):
                                first = simd.vbitcast(simd.vlds(picked[start]), "uint16")
                                second = simd.vbitcast(simd.vlds(picked[start + 64]), "uint16")
                                selection, _ = simd.vdintlv(first, second)
                                out_mask = simd.vcmps(simd.vadds(lanes16, tla.cast(start, tl.uint16)), K,
                                                      op=simd.CMP.LT)
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

                if SORT_BACKEND is not None:
                    # Build FP32 comparison scores and keep original positions
                    # as the proposal payload. Narrow types use an exact encoding.
                    width: tl.constexpr = direct_width
                    scores = tla.alloc((width, ), tl.float32, space="ub")
                    positions = tla.alloc((width, ), tl.uint32, space="ub")
                    proposals = tla.alloc((2 * width, ), tl.uint32, space="ub")
                    workspace = tla.alloc((2 * width, ), tl.uint32, space="ub")
                    if bits == 16:
                        selected_keys = tla.alloc((width, ), tl.uint16, space="ub")
                        with tla.scope.simd():
                            lanes = simd.vbitcast(simd.vci(0), "uint32")
                            lane16, _ = simd.vdintlv(simd.vbitcast(lanes, "uint16"),
                                                     simd.vbitcast(simd.vadds(lanes, 64), "uint16"))
                            for start in range(0, width, 128):
                                valid = simd.vcmps(simd.vadds(lane16, tla.cast(start, tl.uint16)), sort_count,
                                                   op=simd.CMP.LT)
                                if DIRECT:
                                    raw = simd.vbitcast(simd.vlds(data[start]), "uint16")
                                else:
                                    first = simd.vbitcast(simd.vlds(picked[start]), "uint16")
                                    second = simd.vbitcast(simd.vlds(picked[start + 64]), "uint16")
                                    selection, _ = simd.vdintlv(first, second)
                                    raw = simd.vbitcast(simd.vgather(data, selection, valid), "uint16")
                                sign = simd.vdup(0x8000, tl.uint16)
                                ones = simd.vdup(0xffff, tl.uint16)
                                flip = simd.vsel(simd.vcmp(raw, sign, op=simd.CMP.GE), ones, sign)
                                key = simd.vxor(raw, flip)
                                if not LARGEST:
                                    key = simd.vxor(key, ones)
                                # Preserve the original sign of zero in the score.
                                simd.vsts(selected_keys[start], key, mask=valid)
                    with tla.scope.simd():
                        lanes = simd.vbitcast(simd.vci(0), "uint32")
                        for start in range(0, width, 64):
                            order = simd.vadds(lanes, tla.cast(start, tl.uint32))
                            selected = order if DIRECT else simd.vlds(picked[start])
                            valid = simd.vcmps(order, sort_count, op=simd.CMP.LT)
                            if bits == 32:
                                original = simd.vlds(data[start]) if DIRECT else simd.vgather(data, selected, valid)
                                sign = simd.vdup(0 if LARGEST else 0x80000000, tl.uint32)
                                score = simd.vxor(simd.vbitcast(original, "uint32"), sign)
                                score = simd.vsel(valid, score, simd.vdup(0xff800000, tl.uint32))
                            else:
                                if bits == 16:
                                    key = simd.vunpack(simd.vlds(selected_keys[start], dist="UNPK_B16"), tl.uint32)
                                else:
                                    key = simd.vgather(keys, selected, valid)
                                score = simd.vadds(key, tla.cast(0x00800000, tl.uint32))
                                score = simd.vsel(valid, score, simd.vdup(0, tl.uint32))
                            simd.vsts(scores[start], simd.vbitcast(score, "float32"))
                            simd.vsts(positions[start], simd.vsel(valid, selected, simd.vdup(0xffffffff, tl.uint32)))
                    _sort_proposals(scores, positions, proposals, workspace, width, SORT_BACKEND)
                    if dtype == tl.float32 and sort_count != width:
                        # Padding and genuine worst infinities share a score. Keep
                        # all real positions even if sorting interleaves the two.
                        sorted_positions = tla.alloc((width, ), tl.uint32, space="ub")
                        with tla.scope.simd():
                            lane = simd.vbitcast(simd.vci(0), "uint32")
                            written = simd.vdup(0, tl.uint32)
                            for start in range(0, width, 64):
                                _, pos = simd.vdintlv(simd.vlds(proposals[2 * start]),
                                                      simd.vlds(proposals[2 * start + 64]))
                                real = simd.vcmps(pos, 0xffffffff, op=simd.CMP.NE)
                                count = simd.vdupv(simd.vreduce(simd.vdup(1, tl.uint32), "add", mask=real))
                                simd.vscatter(sorted_positions, simd.vadd(lane, written), simd.vsqz(pos, real),
                                              simd.vcmp(lane, count, op=simd.CMP.LT))
                                written = simd.vadd(written, count)
                    # Split the interleaved score/position pairs for output.
                    with tla.scope.simd():
                        lanes = simd.vbitcast(simd.vci(0), "uint32")
                        for start in range(0, K, 64):
                            pair_first = simd.vlds(proposals[2 * start])
                            pair_second = simd.vlds(proposals[2 * start + 64])
                            ordered, index = simd.vdintlv(pair_first, pair_second)
                            pos = simd.vadds(lanes, tla.cast(start, tl.uint32))
                            valid = simd.vcmps(pos, K, op=simd.CMP.LT)
                            if dtype == tl.float32:
                                if sort_count == width:
                                    sign = simd.vdup(0 if LARGEST else 0x80000000, tl.uint32)
                                    value = simd.vbitcast(simd.vxor(ordered, sign), "float32")
                                else:
                                    index = simd.vlds(sorted_positions[start])
                                    value = simd.vgather(data, index, valid)
                                simd.vsts(values[start], value, mask=valid)
                            else:
                                key = simd.vsub(ordered, simd.vdup(0x00800000, tl.uint32))
                                if not LARGEST:
                                    key = simd.vxor(key, simd.vdup((1 << bits) - 1, tl.uint32))
                                if bits == 8:
                                    if dtype == tl.int8:
                                        key = simd.vxor(key, simd.vdup(128, tl.uint32))
                                    simd.vsts(values[start], key, dist="PK4_B32", mask=valid)
                                else:
                                    sign = simd.vdup(0x8000, tl.uint32)
                                    flip = simd.vsel(simd.vcmp(key, sign, op=simd.CMP.GE), sign,
                                                     simd.vdup(0xffff, tl.uint32))
                                    raw = simd.vxor(key, flip)
                                    simd.vsts(values[start], raw, dist="PK_B32", mask=valid)
                            if previous is not None:
                                index = simd.vbitcast(simd.vgather(source_indices, index, valid), "uint32")
                            if WIDE_INDEX:
                                low, high = simd.vintlv(index, simd.vdup(0, tl.uint32))
                                words = simd.vadds(lanes, tla.cast(start * 2, tl.uint32))
                                low_mask = simd.vcmps(words, K * 2, op=simd.CMP.LT)
                                high_mask = simd.vcmps(simd.vadds(words, 64), K * 2, op=simd.CMP.LT)
                                simd.vsts(indices[2 * start], simd.vbitcast(low, "int32"), mask=low_mask)
                                simd.vsts(indices[2 * start + 64], simd.vbitcast(high, "int32"), mask=high_mask)
                            else:
                                simd.vsts(indices[start], simd.vbitcast(index, "int32"), mask=valid)
                tla.store_ub_to_gm(values, tla.view(out_values, (K, ), offsets=(row * K, )))
                tla.store_ub_to_gm(
                    indices, tla.view(out_indices, (K * word_counts[phase], ),
                                      offsets=(row * K * word_counts[phase], )))

        if TILES and phase == 0:
            tla.sync.sync_all(False)


def _native_launch(x, rows, n, k, largest, cores, last_n=0, wide_index=False, previous=None, tile_indices=False,
                   shape=None, sort_backend=None, direct=False):
    shape = (rows, k) if shape is None else shape
    values = x.new_empty(shape)
    indices = x.new_empty(shape, dtype=torch.int64 if wide_index else torch.int32)
    storage = triton.reinterpret(indices, tl.int32) if wide_index else indices
    native_topk[(min(rows, cores), )](x, values, storage, rows, n, k, largest, last_n, wide_index, previous,
                                      tile_indices, SORT_BACKEND=sort_backend, DIRECT=direct)
    return values, indices


def _tiled_topk(x, k, largest, cores, chunk):
    """Select K candidates per tile and propagate their original indices through each merge."""
    n = x.numel()
    rows = triton.cdiv(n, chunk)
    if x.element_size() == 2 and 1 < rows and rows * k <= 8192:
        # Every launched block participates in the barrier before the final merge.
        partial_values = x.new_empty((rows, k))
        partial_indices = x.new_empty((rows, k), dtype=torch.int32)
        values = x.new_empty((k, ))
        indices = x.new_empty((k, ), dtype=torch.int32)
        native_topk[(min(rows, cores), )](x, partial_values, partial_indices, rows, chunk, k, largest, LAST_N=n % chunk,
                                          TILE_INDICES=True, FinalValues=values, FinalIndices=indices, TILES=rows,
                                          TILED_WIDE_INDEX=False)
        return values, indices

    values, indices = _native_launch(x, rows, chunk, k, largest, cores, n % chunk, wide_index=False, tile_indices=True)
    while rows > 1:
        tiles_per_group = min(rows, 8192 // k)
        width = tiles_per_group * k
        tail = values.numel() % width
        rows = triton.cdiv(rows, tiles_per_group)
        values, indices = _native_launch(values, rows, width, k, largest, cores, tail, wide_index=False,
                                         previous=indices)
    return values.flatten(), indices.flatten()


@tla.jit
def _transpose_tile(X, Y, outer, row, col, N: tl.constexpr, INNER: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
                    VALID_R: tl.constexpr, VALID_C: tl.constexpr, WORDS: tl.constexpr, WIDEN: tl.constexpr):
    dtype: tl.constexpr = X.dtype.element_ty
    byte: tl.constexpr = dtype.primitive_bitwidth == 8
    narrow: tl.constexpr = dtype.primitive_bitwidth == 16
    bits: tl.constexpr = 16 if narrow else 32
    index_dtype: tl.constexpr = tl.uint16 if narrow else tl.uint32
    lanes: tl.constexpr = 2048 // bits
    output_words: tl.constexpr = 2 if WIDEN else WORDS
    row_words: tl.constexpr = R * output_words
    col_words: tl.constexpr = C * WORDS
    src = tla.alloc((R, col_words), dtype, space="ub")
    dst = tla.alloc((C, row_words), dtype, space="ub")
    tla.load_gm_to_ub(
        tla.view(X, (VALID_R, VALID_C * WORDS), strides=(INNER * WORDS, 1), offsets=(outer * N + row, col * WORDS)),
        tla.subview(src, (0, 0), (VALID_R, VALID_C * WORDS)))
    src_flat = tla.reshape(src, (R * col_words, ))
    dst_flat = tla.reshape(dst, (C * row_words, ))
    if byte:
        expanded = tla.alloc((R * C, ), tl.uint32, space="ub")
    with tla.scope.simd():
        lane = simd.vbitcast(simd.vci(0), "uint32")
        if narrow:
            lane16, _ = simd.vdintlv(simd.vbitcast(lane, "uint16"), simd.vbitcast(simd.vadds(lane, 64), "uint16"))
        if byte:
            for start in range(0, R * C, lanes):
                raw = simd.vunpack(simd.vlds(src_flat[start], dist="UNPK4_B8"), tl.uint32)
                simd.vsts(expanded[start], raw)
            simd.membar("VST_VLD")
        for start in range(0, C * row_words, lanes):
            out = simd.vadds(lane16 if narrow else lane, tla.cast(start, index_dtype))
            row_word = simd.vand(out, simd.vdup(row_words - 1, index_dtype))
            dst_row = simd.vshr(out, simd.vdup(row_words.bit_length() - 1, index_dtype))
            if WIDEN:
                # Input indices are nonnegative int32. The high word is zero.
                word = simd.vand(row_word, simd.vdup(1, index_dtype))
                row_index = simd.vshr(row_word, simd.vdup(1, index_dtype))
                addr = simd.vadd(simd.vshls(row_index, C.bit_length() - 1), dst_row)
            elif WORDS == 1:
                addr = simd.vadd(simd.vshls(row_word, C.bit_length() - 1), dst_row)
            else:
                # Transpose adjacent 32-bit words as one 64-bit element.
                low_bit = simd.vand(row_word, simd.vdup(1, index_dtype))
                row_even = simd.vsub(row_word, low_bit)
                col_word = simd.vadd(simd.vshls(dst_row, 1), low_bit)
                addr = simd.vadd(simd.vshls(row_even, C.bit_length() - 1), col_word)
            if WIDEN:
                mask = simd.vcmps(word, 0, op=simd.CMP.EQ)
                simd.vsts(dst_flat[start], simd.vgather(src_flat, addr, mask))
            elif byte:
                simd.vsts(dst_flat[start], simd.vgather(expanded, addr), dist="PK4_B32")
            else:
                simd.vsts(dst_flat[start], simd.vgather(src_flat, addr))
    tla.store_ub_to_gm(
        tla.subview(dst, (0, 0), (VALID_C, VALID_R * output_words)),
        tla.view(Y, (VALID_C, VALID_R * output_words), strides=(N * output_words, 1),
                 offsets=(outer * INNER + col, row * output_words)))


@tla.jit
def _transpose_rows(X, Y, N: tl.constexpr, INNER: tl.constexpr, R: tl.constexpr, C: tl.constexpr, OUTER: tl.constexpr,
                    WORDS: tl.constexpr, WIDEN: tl.constexpr):
    rows: tl.constexpr = triton.cdiv(N, R)
    cols: tl.constexpr = triton.cdiv(INNER, C)
    for tile in range(tl.program_id(0), OUTER * rows * cols, tl.num_programs(0)):
        outer = tile // (rows * cols)
        row = tile // cols % rows * R
        col = tile % cols * C
        with tla.scope.vector():
            # Each edge has its actual DMA extent and the same aligned UB pitch.
            for tail_r in tl.static_range(1 + (N % R != 0)):
                for tail_c in tl.static_range(1 + (INNER % C != 0)):
                    if ((not N % R or (row == (rows - 1) * R) == tail_r)
                            and (not INNER % C or (col == (cols - 1) * C) == tail_c)):
                        _transpose_tile(X, Y, outer, row, col, N, INNER, R, C, N % R if tail_r else R,
                                        INNER % C if tail_c else C, WORDS, WIDEN)


@tla.jit
def _digit_order_row(X, Previous, V, I, begin, base_index, dst_row, N: tl.constexpr, K: tl.constexpr,
                     LARGEST: tl.constexpr, FLOAT: tl.constexpr, WORDS: tl.constexpr, WIDE_INDEX: tl.constexpr,
                     SORT_BACKEND: tl.constexpr):
    """Stable LSD ordering of one tile, using the old rank to break ties per digit."""
    width: tl.constexpr = max(64, triton.next_power_of_2(N))
    index_words: tl.constexpr = 2 if WIDE_INDEX else 1
    data = tla.alloc((width * WORDS, ), tl.uint32, space="ub")
    scores = tla.alloc((width, ), tl.float32, space="ub")
    order = tla.alloc((width, ), tl.uint32, space="ub")
    proposals = tla.alloc((2 * width, ), tl.uint32, space="ub")
    workspace = tla.alloc((2 * width, ), tl.uint32, space="ub")
    out = tla.alloc((WORDS * K, ), tl.uint32, space="ub")
    indices = tla.alloc((index_words * K, ), tl.uint32, space="ub")
    tla.load_gm_to_ub(tla.view(X, (N * WORDS, ), offsets=(begin * WORDS, )), tla.subview(data, (0, ), (N * WORDS, )))
    if Previous is not None:
        prev = tla.alloc((width, ), tl.uint32, space="ub")
        tla.load_gm_to_ub(tla.view(Previous, (N, ), offsets=(begin, )), tla.subview(prev, (0, ), (N, )))

    for digit in tl.static_range(WORDS * 4):
        with tla.scope.simd():
            lane = simd.vbitcast(simd.vci(0), "uint32")
            zero = simd.vdup(0, tl.uint32)
            for start in range(0, width, 64):
                rank = simd.vadds(lane, tla.cast(start, tl.uint32))
                if digit == 0:
                    pos = rank
                else:
                    _, pos = simd.vdintlv(simd.vlds(proposals[2 * start]), simd.vlds(proposals[2 * start + 64]))
                valid = simd.vcmps(pos, N, op=simd.CMP.LT)
                if WORDS == 2:
                    address = simd.vadds(simd.vshls(pos, 1), digit // 4)
                else:
                    address = pos
                key = simd.vgather(data, address, valid)
                if FLOAT:
                    sign = simd.vdup(0x80000000, tl.uint32)
                    flip = simd.vsel(simd.vcmp(key, sign, op=simd.CMP.GE), simd.vdup(0xffffffff, tl.uint32), sign)
                    key = simd.vxor(key, flip)
                elif digit == WORDS * 4 - 1:
                    key = simd.vxor(key, simd.vdup(0x80000000, tl.uint32))
                if not LARGEST:
                    key = simd.vxor(key, simd.vdup(0xffffffff, tl.uint32))
                byte = simd.vand(simd.vshr(key, simd.vdup((digit % 4) * 8, tl.uint32)), simd.vdup(255, tl.uint32))
                tie = simd.vsub(simd.vdup(width - 1, tl.uint32), rank)
                # The score bits represent a positive normal FP32 exactly.
                packed = simd.vadds(simd.vadd(simd.vshls(byte, width.bit_length() - 1), tie), 0x00800000)
                simd.vsts(scores[start], simd.vbitcast(simd.vsel(valid, packed, zero), "float32"))
                simd.vsts(order[start], pos)
        _sort_proposals(scores, order, proposals, workspace, width, SORT_BACKEND)

    with tla.scope.simd():
        lane = simd.vbitcast(simd.vci(0), "uint32")
        zero = simd.vdup(0, tl.uint32)
        for start in range(0, K, 64):
            _, pos = simd.vdintlv(simd.vlds(proposals[2 * start]), simd.vlds(proposals[2 * start + 64]))
            rank = simd.vadds(lane, tla.cast(start, tl.uint32))
            valid = simd.vcmps(rank, K, op=simd.CMP.LT)
            if WORDS == 2:
                address = simd.vshls(pos, 1)
                low = simd.vgather(data, address, valid)
                high = simd.vgather(data, simd.vadds(address, 1), valid)
                _store_words(out, start, K, low, high, rank)
            else:
                simd.vsts(out[start], simd.vgather(data, pos, valid), mask=valid)
            if Previous is not None:
                pos = simd.vgather(prev, pos, valid)
            else:
                pos = simd.vadds(pos, tla.cast(base_index, tl.uint32))
            if WIDE_INDEX:
                _store_words(indices, start, K, pos, zero, rank)
            else:
                simd.vsts(indices[start], pos, mask=valid)
    tla.store_ub_to_gm(out, tla.view(V, (K * WORDS, ), offsets=(dst_row * K * WORDS, )))
    tla.store_ub_to_gm(indices, tla.view(I, (K * index_words, ), offsets=(dst_row * K * index_words, )))


@tla.jit
def _store_words(dst, start, k: tl.constexpr, low, high, positions):
    first, second = simd.vintlv(low, high)
    word = simd.vadds(positions, tla.cast(start, tl.uint32))
    first_mask = simd.vcmps(word, k * 2, op=simd.CMP.LT)
    second_mask = simd.vcmps(simd.vadds(word, 64), k * 2, op=simd.CMP.LT)
    simd.vsts(dst[2 * start], first, mask=first_mask)
    simd.vsts(dst[2 * start + 64], second, mask=second_mask)


@tla.jit
def _digit_order(X, Previous, V, I, ROWS, N: tl.constexpr, K: tl.constexpr, LENGTH: tl.constexpr, LARGEST: tl.constexpr,
                 FLOAT: tl.constexpr, WORDS: tl.constexpr, WIDE_INDEX: tl.constexpr, SORT_BACKEND: tl.constexpr):
    tiles: tl.constexpr = triton.cdiv(LENGTH, N)
    for tile in range(tl.program_id(0), ROWS * tiles, tl.num_programs(0)):
        tile_col = tile % tiles
        offset = tile_col * N
        begin = tile // tiles * LENGTH + offset
        with tla.scope.vector():
            if LENGTH % N:
                if tile_col == tiles - 1:
                    _digit_order_row(X, Previous, V, I, begin, offset, tile, LENGTH % N, K, LARGEST, FLOAT, WORDS,
                                     WIDE_INDEX, SORT_BACKEND)
                else:
                    _digit_order_row(X, Previous, V, I, begin, offset, tile, N, K, LARGEST, FLOAT, WORDS, WIDE_INDEX,
                                     SORT_BACKEND)
            else:
                _digit_order_row(X, Previous, V, I, begin, offset, tile, N, K, LARGEST, FLOAT, WORDS, WIDE_INDEX,
                                 SORT_BACKEND)


def _unsigned_view(x):
    return triton.reinterpret(x, tl.uint32)


def _digit_launch(x, n, k, tile, largest, cores, sort_backend, previous=None):
    rows = x.numel() // n
    tiles = triton.cdiv(n, tile)
    values = x.new_empty((rows, tiles * k))
    indices = x.new_empty(values.shape, dtype=torch.int64 if tiles == 1 else torch.int32)
    _digit_order[(min(rows * tiles,
                      cores), )](_unsigned_view(x), None if previous is None else _unsigned_view(previous),
                                 _unsigned_view(values), _unsigned_view(indices), rows, tile, k, n, largest,
                                 x.dtype == torch.float32, x.element_size() // 4, tiles == 1, sort_backend)
    return values, indices


def _order_int64(x, n, k, largest, cores, sort_backend):
    # Balanced initial tiles avoid a final tile with fewer than K real entries.
    if k > 2048:
        raise ValueError("int64 TopK supports K up to 2048")
    tiles = max(1, n // max(2048, k))
    tile = triton.cdiv(n, tiles)
    if tile > 4096:
        raise ValueError("int64 TopK tile exceeds the UB sorter capacity")
    values, indices = _digit_launch(x, n, k, tile, largest, cores, sort_backend)
    while tiles > 1:
        groups = triton.cdiv(tiles, max(2, 2048 // k))
        tile = triton.cdiv(tiles, groups) * k
        values, indices = _digit_launch(values, tiles * k, k, tile, largest, cores, sort_backend, previous=indices)
        tiles = triton.cdiv(tiles * k, tile)
    return values, indices


def _transpose(x, outer, rows, cols, cores, *, widen_indices=False):
    dtype_size = x.element_size()
    tile = 64 if dtype_size < 8 else 32
    output = x.new_empty((outer, cols, rows), dtype=torch.int64 if widen_indices else x.dtype)
    work = outer * triton.cdiv(rows, tile) * triton.cdiv(cols, tile)
    words = 2 if dtype_size == 8 else 1
    src, dst = (_unsigned_view(x), _unsigned_view(output)) if words == 2 or widen_indices else (x, output)
    _transpose_rows[(min(cores, work), )](src, dst, rows, cols, tile, tile, outer, words, widen_indices)
    return output


def topk(x, k, dim=-1, largest=True, sorted=True, *, implementation="auto", chunk=4096, sort_backend="template"):
    """TopK through TLA, returning values and int64 indices in the requested layout."""
    if not -x.ndim <= dim < x.ndim or not x.is_contiguous():
        raise ValueError("expected a valid axis and contiguous input")
    if sort_backend not in ("template", "simd"):
        raise ValueError("sort_backend must be 'template' or 'simd'")
    if implementation not in ("auto", "radix", "sort", "tiles", "full", "byte"):
        raise ValueError(f"unknown selection implementation: {implementation}")
    axis = dim % x.ndim
    n = x.shape[axis]
    if not 0 < k <= n or (implementation == "full" and k != n):
        raise ValueError("expected 0 < k <= axis length, or k == axis length for full selection")
    shape = list(x.shape)
    shape[axis] = k
    if not x.numel():
        return x.new_empty(shape), x.new_empty(shape, dtype=torch.int64)
    cores = driver.active.utils.get_aivector_core_num()
    inner = math.prod(x.shape[axis + 1:])
    outer = x.numel() // (n * inner)
    rows = _transpose(x, outer, n, inner, cores) if inner != 1 else x
    row_count = outer * inner
    if x.dtype == torch.int64:
        values, indices = _order_int64(rows, n, k, largest, cores, sort_backend)
    elif n > 8192 or implementation == "tiles":
        if row_count != 1 or not k <= chunk <= 8192 or 0 < n % chunk < k:
            raise ValueError("wide radix selection requires one row and K real elements in each tile")
        values, indices = _tiled_topk(rows, k, largest, cores, chunk)
        if sorted and k > 1:
            if x.dtype == torch.int32:
                values, indices = _digit_launch(values, k, k, k, largest, cores, sort_backend, previous=indices)
            else:
                values, indices = _native_launch(values, 1, k, k, largest, cores, wide_index=True, previous=indices,
                                                 sort_backend=sort_backend, direct=True)
        else:
            values, indices = _native_launch(values, 1, k, k, largest, cores, wide_index=True, previous=indices)
    elif x.dtype == torch.int32 and sorted and k > 1:
        values, indices = _native_launch(rows, row_count, n, k, largest, cores)
        values, indices = _digit_launch(values, k, k, k, largest, cores, sort_backend, previous=indices)
    else:
        backend = sort_backend if sorted and k > 1 else None
        direct = backend is not None and n <= 4096 and (implementation == "sort" or k * 2 >= n and n <= 256)
        values, indices = _native_launch(rows, row_count, n, k, largest, cores, wide_index=inner == 1,
                                         sort_backend=backend, direct=direct)
    if inner != 1:
        values = _transpose(values, outer, inner, k, cores)
        indices = _transpose(indices, outer, inner, k, cores, widen_indices=indices.dtype == torch.int32)
    return values.view(shape), indices.view(shape)
