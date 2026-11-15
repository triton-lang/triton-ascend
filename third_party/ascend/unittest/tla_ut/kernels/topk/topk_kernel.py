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

import hashlib
import math
import shutil
import subprocess
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import extension as al
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction
from triton.runtime import driver
from triton.runtime.cache import get_cache_manager

simd = tla.simd
_tiled_launch = threading.local()


@lru_cache(maxsize=1)
def _proposal_sort_bitcode():
    """Build the small UB intrinsic adapter with the CCE compiler."""
    source = Path(__file__).with_name("topk_proposal_sort.cce")
    compiler = shutil.which("bisheng")
    if compiler is None or not source.is_file():
        return None
    flags = (
        "-x",
        "cce",
        "-emit-llvm",
        "-c",
        "-O2",
        "--cce-aicore-only",
        "--cce-aicore-arch=dav-c310-vec",
        "--cce-generic-addrspace=off",
        "--cce-auto-sync=off",
        "-mllvm",
        "-disable-llvm-optzns",
        "-std=c++17",
    )
    version = subprocess.check_output([compiler, "--version"])
    cache = get_cache_manager(hashlib.sha256(source.read_bytes() + version + repr(flags).encode()).hexdigest())
    name = "topk_proposal_sort.bc"
    if cached := cache.get_file(name):
        return cached
    with tempfile.TemporaryDirectory(prefix="triton-topk-") as folder:
        binary = Path(folder) / name
        subprocess.run([compiler, *flags, str(source), "-o", str(binary)], check=True)
        return cache.put(binary.read_bytes(), name)


@al.builtin
def _sort_proposals(scores, positions, width, proposals, workspace, bitcode, _semantic=None):
    builder = _semantic.builder
    symbol = "sort_proposals"
    attrs = {
        "symbol": builder.get_string_attr(symbol),
        "bitcode": builder.get_string_attr(bitcode.value),
        "gm_addr_args_indices": builder.parse_attr("array<i32>"),
        "hivm.tcore_type": builder.parse_attr("#hivm.tcore_type<VECTOR>"),
        "hivm.vf_mode": builder.parse_attr("#hivm.vf_mode<SIMD>"),
        "hivm.pipe": builder.parse_attr("#hivm.pipe<PIPE_V>"),
    }
    builder.create_custom_op(
        symbol, attrs,
        [scores.handle, positions.handle, builder.get_int32(width.value)], [proposals.handle, workspace.handle], [])


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
    SORT_BITCODE: tl.constexpr = None,
    DIRECT: tl.constexpr = False,
):
    """Select each row and optionally merge the tiles in the same launch."""
    tl.static_assert(LAST_N == 0 or (K <= LAST_N and LAST_N < N))
    tl.static_assert(TILES * K <= 8192)
    tl.static_assert(not DIRECT or SORT_BITCODE is not None)
    tl.static_assert(SORT_BITCODE is None or (not TILES and Previous is None and not TILE_INDICES))
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
        if TILES:
            row_start = tl.program_id(0)
            row_end = row_count
            row_step = step
        else:
            rows_per_core = tl.cdiv(row_count, step)
            row_start = tl.program_id(0) * rows_per_core
            row_end = tl.minimum(row_count, row_start + rows_per_core)
            row_step = 1
        for row in tl.range(row_start, row_end, row_step):
            with tla.scope.vector():
                data = tla.alloc((padded_widths[phase], ), dtype, space="ub")
                keys = tla.alloc((padded_widths[phase], ), key_dtype, space="ub")
                if not narrow_keys:
                    bytes_by_digit = tla.alloc((bits // 8, padded_widths[phase]), tl.uint8, space="ub")
                picked_width: tl.constexpr = output_padded if SORT_BITCODE is None else direct_width
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
                    if DIRECT:
                        # Use all local positions when the selected fraction
                        # is large enough to make the histogram pass redundant.
                        for base in range(0, direct_width, 64):
                            seq = simd.vadds(lanes, tla.cast(base, tl.uint32))
                            simd.vsts(picked[base], seq)
                    else:
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

                    if SORT_BITCODE is None:
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

                if SORT_BITCODE is not None:
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
                                first = simd.vbitcast(simd.vlds(picked[start]), "uint16")
                                second = simd.vbitcast(simd.vlds(picked[start + 64]), "uint16")
                                selection, _ = simd.vdintlv(first, second)
                                valid = simd.vcmps(simd.vadds(lane16, tla.cast(start, tl.uint16)), sort_count,
                                                   op=simd.CMP.LT)
                                simd.vsts(selected_keys[start], simd.vgather(keys, selection, valid), mask=valid)
                    with tla.scope.simd():
                        lanes = simd.vbitcast(simd.vci(0), "uint32")
                        for start in range(0, width, 64):
                            selected = simd.vlds(picked[start])
                            order = simd.vadds(lanes, tla.cast(start, tl.uint32))
                            valid = simd.vcmps(order, sort_count, op=simd.CMP.LT)
                            if bits == 32:
                                original = simd.vgather(data, selected, valid)
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
                            simd.vsts(positions[start], simd.vsel(valid, selected, simd.vdup(0, tl.uint32)))
                    _sort_proposals(scores, positions, width, proposals, workspace, SORT_BITCODE)
                    # Split the interleaved score/position pairs for output.
                    with tla.scope.simd():
                        lanes = simd.vbitcast(simd.vci(0), "uint32")
                        for start in range(0, K, 64):
                            pair_first = simd.vlds(proposals[2 * start])
                            pair_second = simd.vlds(proposals[2 * start + 64])
                            ordered, index = simd.vdintlv(pair_first, pair_second)
                            pos = simd.vadds(lanes, tla.cast(start, tl.uint32))
                            valid = simd.vcmps(pos, K, op=simd.CMP.LT)
                            if bits == 32:
                                sign = simd.vdup(0 if LARGEST else 0x80000000, tl.uint32)
                                value = simd.vbitcast(simd.vxor(ordered, sign), "float32")
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


@triton.jit
def _full_topk(X, V, I, SIZE: tl.constexpr, N: tl.constexpr, INNER: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < SIZE
    value = tl.load(X + offset, valid, other=0)
    tl.store(V + offset, value, valid)
    tl.store(I + offset, offset // INNER % N, valid)


def _native_launch(x, rows, n, k, largest, cores, last_n=0, wide_index=False, previous=None, tile_indices=False,
                   shape=None, sort_bitcode=None, direct=False):
    shape = (rows, k) if shape is None else shape
    values = x.new_empty(shape)
    indices = x.new_empty(shape, dtype=torch.int64 if wide_index else torch.int32)
    storage = triton.reinterpret(indices, tl.int32) if wide_index else indices
    native_topk[(min(rows, cores), )](x, values, storage, rows, n, k, largest, last_n, wide_index, previous,
                                      tile_indices, SORT_BITCODE=sort_bitcode, DIRECT=direct)
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
                    K_TILE: tl.constexpr):
    groups = tl.cdiv(INNER, ROW_TILE)
    tile = tl.program_id(0)
    outer = tile // groups
    row = tile % groups * ROW_TILE
    col = tl.program_id(1) * K_TILE
    base = outer * INNER * K
    # Boundary checks retain contiguous DMA even on a partial tile.
    index_ptr = tl.make_block_ptr(Indices + base, (INNER, K), (K, 1), (row, col), (ROW_TILE, K_TILE), (1, 0))
    index_out_ptr = tl.make_block_ptr(I + base, (K, INNER), (INNER, 1), (col, row), (K_TILE, ROW_TILE), (1, 0))
    index = tl.load(index_ptr, boundary_check=(0, 1), padding_option="zero")
    value_ptr = tl.make_block_ptr(Values + base, (INNER, K), (K, 1), (row, col), (ROW_TILE, K_TILE), (1, 0))
    value_out_ptr = tl.make_block_ptr(V + base, (K, INNER), (INNER, 1), (col, row), (K_TILE, ROW_TILE), (1, 0))
    value = tl.load(value_ptr, boundary_check=(0, 1), padding_option="zero")
    tl.store(value_out_ptr, tl.trans(value), boundary_check=(0, 1))
    tl.store(index_out_ptr, tl.trans(index).to(I.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _pack_rows(X, Y, N: tl.constexpr, INNER: tl.constexpr, WIDTH: tl.constexpr, COLUMNS: tl.constexpr):
    groups: tl.constexpr = tl.cdiv(INNER, COLUMNS)
    program = tl.program_id(0)
    outer = program // groups
    column = program % groups * COLUMNS
    row = tl.program_id(1) * WIDTH
    shape: tl.constexpr = (N, INNER)
    source = tl.make_block_ptr(X + outer * N * INNER, shape, (INNER, 1), (row, column), (WIDTH, COLUMNS), (1, 0))
    destination = tl.make_block_ptr(Y + outer * N * INNER, (INNER, N), (N, 1), (column, row), (COLUMNS, WIDTH), (1, 0))
    tile = tl.load(source, boundary_check=(0, 1), padding_option="zero")
    tl.store(destination, tl.trans(tile).to(Y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _cast(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + offset, tl.load(X + offset, offset < N, other=0), offset < N)


def _cast_tensor(x, dtype):
    if x.dtype == dtype:
        return x
    output = x.new_empty(x.shape, dtype=dtype)
    if x.numel():
        block = 2048
        _cast[(triton.cdiv(x.numel(), block), )](x, output, x.numel(), block)
    return output


def _contiguous_rows(x, axis, dtype=None):
    n = x.shape[axis]
    inner = math.prod(x.shape[axis + 1:])
    dtype = x.dtype if dtype is None else dtype
    if inner == 1:
        return _cast_tensor(x, dtype).view(-1, n)
    row_count = x.numel() // n
    result = x.new_empty((row_count, n), dtype=dtype)
    if row_count:
        width = min(64, triton.next_power_of_2(n))
        cols = min(64, triton.next_power_of_2(inner))
        if inner % 32:
            width = min(width, 64 if x.element_size() == 2 else 32)
            cols = min(cols, 32 if x.element_size() == 2 else 64)
        grid = (row_count // inner * triton.cdiv(inner, cols), triton.cdiv(n, width))
        _pack_rows[grid](x, result, n, inner, width, cols)
    return result


def _restore_layout(x, values, indices, k, axis, *, index_dtype=torch.int64):
    inner = math.prod(x.shape[axis + 1:])
    shape = list(x.shape)
    shape[axis] = k
    if inner == 1:
        return _cast_tensor(values, x.dtype).view(shape), _cast_tensor(indices, index_dtype).view(shape)
    out = x.new_empty(shape)
    out_indices = x.new_empty(shape, dtype=index_dtype)
    if values.numel():
        outer = values.numel() // (k * inner)
        if inner % 64 == 0 and k % 64 == 0:
            row_tile, k_tile = 64, 64
        elif inner % 32:
            row_tile, k_tile = 128, 16
        else:
            row_tile, k_tile = 32, 32
        row_tile = min(row_tile, triton.next_power_of_2(inner))
        k_tile = min(k_tile, triton.next_power_of_2(k))
        grid = (outer * triton.cdiv(inner, row_tile), triton.cdiv(k, k_tile))
        _restore_output[grid](values, indices, out, out_indices, k, inner, row_tile, k_tile)
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
    # Work on contiguous rows through selection and sorting, then restore
    # the requested axis once. Candidate indices stay in int32 until then.
    if sorted and k > 1 and fused_output is None and implementation in ("radix", "full", "sort"):
        return _ordered_topk(x, k, axis, largest, implementation, bounded_int64)
    if implementation == "sort" or (sorted
                                    and implementation == "full") or (x.dtype == torch.int64 and not bounded_int64
                                                                      and implementation == "radix"):
        return _sort_output(x, None, k, axis, largest, bounded_int64)
    values, indices = _select_topk(x, k, axis, largest, implementation, chunk, bounded_int64, fused_output)
    if sorted and k > 1:
        return _sort_output(values, indices, k, axis, largest, bounded_int64)
    return values, indices


def _ordered_topk(x, k, axis, largest, implementation, bounded_int64):
    n = x.shape[axis]
    inner = math.prod(x.shape[axis + 1:])
    if n <= 32 and inner >= 128 and x.dtype in (torch.float16, torch.bfloat16):
        return _short_strided_topk(x, k, axis, largest, inner)
    if not x.numel():
        shape = list(x.shape)
        shape[axis] = k
        return x.new_empty(shape), x.new_empty(shape, dtype=torch.int64)
    index_dtype = torch.int64 if axis == x.ndim - 1 else torch.int32
    wide_int64 = x.dtype == torch.int64 and not bounded_int64
    select_first = implementation == "radix" and not wide_int64
    source_dtype = torch.int32 if select_first and bounded_int64 and x.dtype == torch.int64 else x.dtype
    rows = _contiguous_rows(x, axis, source_dtype)
    values, indices = rows, None
    if select_first:
        if n > 8192:
            raise ValueError("radix selection supports axes of at most 8192 elements")
        cores = driver.active.utils.get_aivector_core_num()
        # Sort narrow, short BF16 candidate rows with the existing batched
        # packed-key path when no output transpose is needed.
        minimum_k = 32 if x.dtype == torch.float32 else 96
        use_sorter = (minimum_k <= k <= 2048 and (n <= 4096 or k <= 1024)
                      and x.dtype in (torch.int8, torch.uint8, torch.float16, torch.bfloat16, torch.float32)
                      and not (x.dtype == torch.bfloat16 and inner == 1 and k <= 128))
        bitcode = _proposal_sort_bitcode() if use_sorter else None
        if bitcode is not None:
            values, indices = _native_launch(rows, rows.shape[0], n, k, largest, cores, wide_index=axis == x.ndim - 1,
                                             sort_bitcode=bitcode, direct=n <= 256 and k * 4 >= n)
            return _restore_layout(x, values, indices, k, axis)
        values, indices = _native_launch(rows, rows.shape[0], n, k, largest, cores)
    values, indices = _sort_output(values, indices, k, 1, largest, bounded_int64, index_dtype=index_dtype)
    if axis != x.ndim - 1:
        return _restore_layout(x, values, indices, k, axis)
    shape = (*x.shape[:-1], k)
    return _cast_tensor(values, x.dtype).view(shape), indices.view(shape)


def _short_strided_topk(x, k, axis, largest, inner):
    n = x.shape[axis]
    shape = list(x.shape)
    shape[axis] = k
    values = x.new_empty(shape)
    indices = x.new_empty(shape, dtype=torch.int64)
    if x.numel():
        columns = 128
        grid = (x.numel() // (n * inner) * triton.cdiv(inner, columns), )
        _sort_short_columns[grid](x, values, indices, inner, n, k, triton.next_power_of_2(n), columns, largest)
    return values, indices


@triton.constexpr_function
def _bitonic_peer(row, step):
    return row ^ (1 << step)


@triton.jit
def _sort_short_columns(X, V, I, INNER: tl.constexpr, N: tl.constexpr, K: tl.constexpr, WIDTH: tl.constexpr,
                        COLUMNS: tl.constexpr, LARGEST: tl.constexpr):
    groups: tl.constexpr = triton.cdiv(INNER, COLUMNS)
    tile = tl.program_id(0)
    outer = tile // groups
    column = tile % groups * COLUMNS + tl.arange(0, COLUMNS)
    valid = column < INNER
    bits: tl.constexpr = WIDTH.bit_length()
    packed = ()
    # Each register contains the same axis position from neighboring columns.
    for row in tl.static_range(WIDTH):
        if row < N:
            value = tl.load(X + (outer * N + row) * INNER + column, valid, other=0)
            key = _sort_key(value)
            if not LARGEST:
                key = ~key
            score = ((key & 0xFFFF).to(tl.int32) << bits) | (WIDTH - row)
        else:
            score = tl.full((COLUMNS, ), 0, tl.int32)
        packed += (score, )
    for stage in tl.static_range(1, bits):
        for step in tl.static_range(stage - 1, -1, -1):
            next_scores = ()
            for row in tl.static_range(WIDTH):
                other = packed[_bitonic_peer(row, step)]
                if (((row >> stage) ^ (row >> step)) & 1) == 0:
                    score = tl.maximum(packed[row], other)
                else:
                    score = tl.minimum(packed[row], other)
                next_scores += (score, )
            packed = next_scores
    for row in tl.static_range(K):
        ordered = packed[row]
        index = WIDTH - (ordered & ((1 << bits) - 1))
        key = (ordered >> bits).to(tl.uint16)
        if not LARGEST:
            key = ~key
        value = _decode_narrow_key(key, X.dtype.element_ty)
        destination = (outer * K + row) * INNER + column
        tl.store(V + destination, value, valid)
        tl.store(I + destination, index, valid)


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
                    triton.reinterpret(indices, tl.int32), rows, None, False)
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
            values, indices = _tiled_topk(_cast_tensor(x, torch.int32), k, largest, cores, chunk)
            return _cast_tensor(values, x.dtype), indices
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
    rows = _contiguous_rows(x, axis, dtype)
    values, indices = _native_launch(rows, rows.numel() // n, n, k, largest, cores, wide_index=inner == 1)
    return _restore_layout(x, values, indices, k, axis)


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


def _sort_output(values, indices, k, axis, largest, bounded_int64=False, index_dtype=torch.int64):
    shape = list(values.shape)
    n = shape[axis]
    shape[axis] = k
    out_values = values.new_empty(shape)
    out_indices = values.new_empty(shape, dtype=index_dtype)
    if not values.numel():
        return out_values, out_indices
    inner = math.prod(values.shape[axis + 1:])
    rows = values.numel() // n
    width = triton.next_power_of_2(n)
    if values.dtype == torch.int64 and not bounded_int64 and indices is None and width > 1024:
        rows_view = _contiguous_rows(values, axis)
        ordered_values, ordered_indices = _sort_wide_rows(rows_view, k, largest)
        return _restore_layout(values, ordered_values, ordered_indices, k, axis, index_dtype=index_dtype)

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
def _sort_packed(pair):
    # Order-preserving encoding in the positive normal FP32 bit patterns.
    score = (pair + 0x00800000).to(tl.float32, bitcast=True)
    return al.sort(score, descending=True).to(tl.int32, bitcast=True) - 0x00800000


@triton.jit
def _order_int64(x, valid, lane, WIDTH: tl.constexpr, LARGEST: tl.constexpr):
    key = x.to(tl.uint64, bitcast=True) ^ 0x8000000000000000
    if not LARGEST:
        key = ~key
    # If the tile's key range is small, subtract its minimum to retain
    # the full ordering and pack the result with the lane in one pass.
    return _sort_key_range(key, valid, lane, WIDTH, 4)


@triton.jit
def _sort_key_range(key, valid, lane, WIDTH: tl.constexpr, DIGITS: tl.constexpr):
    axis: tl.constexpr = len(key.shape) - 1
    upper = ~tl.full((), 0, key.dtype)
    first = tl.min(tl.where(valid, key, upper), axis=axis)
    last = tl.max(tl.where(valid, key, 0), axis=axis)
    bits: tl.constexpr = WIDTH.bit_length()
    span = tl.max(last - first, axis=0) if axis else last - first
    if span < (1 << (30 - bits)):
        shifted = key - tl.expand_dims(first, axis) if axis else key - first
        pair = (shifted.to(tl.int32) << bits) | (WIDTH - lane)
        order = _sort_packed(tl.where(valid, pair, 0)) & ((1 << bits) - 1)
    else:
        order = _sort_digits(key, valid, lane, WIDTH, DIGITS)
    return order


@triton.jit
def _sort_digits(key, valid, lane, WIDTH: tl.constexpr, DIGITS: tl.constexpr):
    # Stable low-to-high radix sorting. The previous rank breaks ties in
    # each 16-bit digit, retaining the order of all less significant bits.
    bits: tl.constexpr = WIDTH.bit_length()
    suffix_mask: tl.constexpr = (1 << bits) - 1
    order = tl.broadcast_to(tl.where(valid, WIDTH - lane, 0), key.shape)
    for shift in tl.static_range(0, DIGITS * 16, 16):
        source = tl.minimum(WIDTH - order, WIDTH - 1)
        current = key if shift == 0 else tl.gather(key, source, axis=len(key.shape) - 1)
        digit = ((current >> shift) & 0xFFFF).to(tl.int32)
        pair = (digit << bits) | (WIDTH - lane)
        ranked = _sort_packed(tl.where(order > 0, pair, 0))
        prev_order = (ranked & suffix_mask).to(tl.int32)
        source = tl.minimum(WIDTH - prev_order, WIDTH - 1)
        if shift == 0:
            order = prev_order
        else:
            order = tl.where(prev_order > 0, tl.gather(order, source, axis=len(key.shape) - 1), 0)
    return order


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
        selected: tl.constexpr = triton.next_power_of_2(K)
        if index_bits <= 14:
            if X.dtype.element_ty.primitive_bitwidth <= 16:
                pair = ((key & 0xFFFF).to(tl.int32) << index_bits) | (WIDTH - lane)
                ordered = _sort_packed(tl.where(lane < N, pair, 0))
            elif WIDTH <= 256:
                # Wider tiles rarely fit once their position uses more bits.
                ordered = _sort_key_range(key, lane < N, lane, WIDTH, 2)
            else:
                ordered = _sort_digits(key, lane < N, lane, WIDTH, 2)
            order_lane = tl.broadcast_to(tl.arange(0, selected)[None, :], (BATCH, selected))
            ordered = tl.gather(ordered, order_lane, axis=1)
        else:
            pair = (key.to(tl.int64) << index_bits) | (WIDTH - lane).to(tl.int64)
            ordered = tl.topk(tl.where(lane < N, pair, 0), selected)
        local = WIDTH - (ordered & ((1 << index_bits) - 1)).to(tl.int32)
        out = tl.arange(0, selected)[None, :]
        valid = (row < ROWS) & (out < K)
    if INNER == 1 and X.dtype.element_ty != tl.int64:
        axis: tl.constexpr = len(x.shape) - 1
        if X.dtype.element_ty.primitive_bitwidth <= 16 and WIDTH.bit_length() <= 14:
            sorted_key = (ordered >> index_bits).to(tl.uint32)
            if not LARGEST:
                sorted_key = ~sorted_key
            value = _decode_narrow_key(sorted_key, X.dtype.element_ty)
        else:
            value = tl.gather(x, tl.minimum(local, WIDTH - 1), axis=axis)
        if Indices is None:
            index = local
        else:
            idx = tl.load(Indices + base + lane, (row < ROWS) & (lane < N), other=0)
            index = tl.gather(idx, tl.minimum(local, WIDTH - 1), axis=axis)
    else:
        source = base + local * INNER
        value = tl.load(X + source, valid, other=0)
        index = local if Indices is None else tl.load(Indices + source, valid, other=0)
    destination = row // INNER * K * INNER + row % INNER + out * INNER
    tl.store(V + destination, value, valid)
    tl.store(I + destination, index, valid)


@triton.jit
def _decode_narrow_key(key, DTYPE: tl.constexpr):
    if DTYPE == tl.float16 or DTYPE == tl.bfloat16:
        bits = key.to(tl.uint16)
        raw = bits ^ tl.where(bits & 0x8000 != 0, 0x8000, 0xFFFF).to(tl.uint16)
        return raw.to(DTYPE, bitcast=True)
    if DTYPE == tl.int8:
        key ^= 0x80
    return key.to(DTYPE)


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
