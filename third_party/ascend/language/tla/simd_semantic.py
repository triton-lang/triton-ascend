# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""SIMD vector-register contracts shared by the frontend.

This module is deliberately dependency-free (pure Python, no triton/torch
imports) so the validation rules stay importable without an installed Triton
build.

Contents:

- dist / part / pp / pge-pattern whitelists mirroring the AVE dialect enums
  (``bishengir/Dialect/HIVMAVE/IR/HIVMAVEAttrs.td``);
- the 2048-bit vector-register invariant (i8 -> 256 lanes, bf16 -> 128,
  f32 -> 64), via :func:`lanes_for`;
- predication-mode (:func:`check_mask_mode`) and vsstb stride
  (:func:`check_vsstb_stride`) contracts for the masked binary ops and the
  stride store.
"""

__all__ = [
    "VL_BITS",
    "LOAD_DISTS",
    "STORE_DISTS",
    "PGE_PATTERNS",
    "VCVT_PARTS",
    "VSSTB_BLOCK_BITS",
    "MASK_MODES",
    "check_dist",
    "check_part",
    "check_pge_pattern",
    "check_total_bits",
    "check_mask_mode",
    "check_vsstb_stride",
    "lanes_for",
]

#: Total bits of one physical vector register (hivm::util::VL_BITS).
VL_BITS = 2048

#: One vsstb stride unit: 32B blocks (hivm::util::vectorBlockSizeBit).
VSSTB_BLOCK_BITS = 256

#: Accepted predication-mode spellings.  ``ZEROING`` writes 0 to masked-off
#: lanes (an explicit select enforces this for masked operations);
#: ``MERGING`` keeps the destination's previous value
#: (``.m`` variants, not yet reachable -- see HIVMAVEToAVEIntrin.cpp
#: "TODO: predication mode merging").
MASK_MODES = frozenset(["zeroing", "merging"])

# AVE LoadDist whitelist (strings accepted by symbolizeLoadDist).
LOAD_DISTS = frozenset([
    "NORM", "UNPK_B8", "UNPK_B16", "UNPK4_B8", "BLK",
    "BRC_B8", "BRC_B16", "BRC_B32", "BRC_B64",
    "DINTLV_B8", "DINTLV_B16", "DINTLV_B32", "BDINTLV", "UNPK_B32",
    "E2B_B16", "E2B_B32",
    "SPLT2CHN_B8", "SPLT2CHN_B16", "SPLT4CHN_B8",
    "US_B8", "US_B16", "DS_B8", "DS_B16", "US", "DS", "PK",
])

# AVE StoreDist whitelist (strings accepted by symbolizeStoreDist).
STORE_DISTS = frozenset([
    "NORM_B8", "NORM_B16", "NORM_B32", "NORM_B64",
    "ONEPT_B8", "ONEPT_B16", "ONEPT_B32", "ONEPT_B64",
    "PK_B16", "PK_B32", "PK_B64", "PK4_B32",
    "INTLV_B8", "INTLV_B16", "INTLV_B32",
    "MRG4CHN_B8", "MRG2CHN_B8", "MRG2CHN_B16",
])

# AVE PgePattern whitelist (strings accepted by symbolizePgePattern).
PGE_PATTERNS = frozenset([
    "ALL", "VL1", "VL2", "VL3", "VL4", "VL8", "VL16", "VL32", "VL64", "VL128",
    "M3", "M4", "H", "Q", "ALLF",
])

# AVE VCVT_PartType whitelist (vextf part selector; assembly strings).
VCVT_PARTS = frozenset(["part_even", "part_odd"])


def check_dist(dist, whitelist, what="load"):
    """Validate a distribution string against a whitelist; return it unchanged."""
    if not isinstance(dist, str) or dist not in whitelist:
        allowed = ", ".join(sorted(whitelist))
        raise ValueError(
            f"invalid {what} dist {dist!r}: expected one of [{allowed}]")
    return dist


def check_part(part):
    """Validate a vextf part selector; return the canonical assembly string."""
    part = str(part).lower()
    if part not in VCVT_PARTS:
        raise ValueError(f"invalid vextf part {part!r}: expected part_even/part_odd")
    return part


def check_pge_pattern(pattern):
    """Validate a pge pattern; return its uppercase canonical form."""
    pattern = str(pattern).upper()
    if pattern not in PGE_PATTERNS:
        allowed = ", ".join(sorted(PGE_PATTERNS))
        raise ValueError(
            f"invalid pge pattern {pattern!r}: expected one of [{allowed}]")
    return pattern


def check_total_bits(lanes, elem_bits, what="vector"):
    """Enforce the 2048-bit register invariant."""
    if lanes <= 0:
        raise ValueError(f"{what} lane count must be positive, got {lanes}")
    if lanes * elem_bits > VL_BITS:
        raise ValueError(
            f"{what} needs {lanes}x{elem_bits} = {lanes * elem_bits} bits, "
            f"exceeding the {VL_BITS}-bit vector register")


def check_mask_mode(mode, what="simd op"):
    """Normalize a predication-mode argument; return the lowercase name.

    Accepts the ``MODE`` enum members exported by :mod:`.simd`, or the
    TileKernels-style strings ``ZEROING`` / ``MODE_MERGING`` / ``merging``.
    Raises ``NotImplementedError`` for merging modes: the intrinsic ``.m``
    variants exist but the HIVMAVE ops carry no mode attribute and the
    HIVMAVEToAVEIntrin dispatch only emits the ``.x`` variants. The caller
    explicitly selects zero for inactive lanes to enforce ZEROING semantics.
    """
    name = getattr(mode, "value", mode)          # MODE enum -> its value
    name = str(name).strip().lower()
    if name.startswith("mode_"):
        name = name[len("mode_"):]
    if name not in MASK_MODES:
        allowed = ", ".join(sorted(MASK_MODES))
        raise ValueError(
            f"invalid {what} predication mode {mode!r}: expected one of "
            f"[{allowed}] (or MODE.ZEROING / MODE.MERGING)")
    if name == "merging":
        raise NotImplementedError(
            f"{what} mode=MODE.MERGING is not supported yet: masked-off "
            f"lanes would keep the destination's previous value, which the "
            f"AVE ops cannot express (see HIVMAVEToAVEIntrin.cpp "
            f"\"TODO: predication mode merging\").  Use "
            f"tla.simd.vsel(mask, op(...), src0) as a merging substitute "
            f"when the destination equals src0.")
    return name


def check_vsstb_stride(stride, elem_bits, what="vsstb"):
    """Validate a vsstb compile-time element stride; return it unchanged.

    The lowering (HIVMStoreStrideOpLowering) resolves the stride from an
    ``arith.constant`` index, so it must be a compile-time integer, and it
    must be a whole number of 32B blocks: the register is written one
    32-byte block at a time, with consecutive blocks landing ``stride``
    elements apart.  For bank-conflict avoidance prefer strides aligned to
    (512*N + 32) bytes (see bishengir docs stride_align.md).
    """
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError(
            f"{what} stride must be a positive compile-time integer, "
            f"got {stride!r}")
    if elem_bits not in (8, 16, 32):
        raise ValueError(f"{what} requires 8-, 16-, or 32-bit elements")
    block = VSSTB_BLOCK_BITS // elem_bits
    if stride % block != 0:
        raise ValueError(
            f"{what} stride {stride} elements must be a multiple of the "
            f"32B block size ({block} elements for a {elem_bits}-bit type)")
    if stride // block > 0xffff:
        raise ValueError(f"{what} stride exceeds the 16-bit block-stride field")
    return stride


def lanes_for(elem_bits):
    """Full-register lane count for an element width (2048 / elem_bits)."""
    if VL_BITS % elem_bits != 0:
        raise ValueError(
            f"element width {elem_bits} does not divide {VL_BITS} bits")
    return VL_BITS // elem_bits
