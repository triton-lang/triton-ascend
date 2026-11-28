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

- per-operation element types and type-dependent conversion parameters;
- dist / part / pp / pge-pattern whitelists mirroring the AVE dialect enums
  (``bishengir/Dialect/HIVMAVE/IR/HIVMAVEAttrs.td``);
- the 2048-bit vector-register invariant (i8 -> 256 lanes, bf16 -> 128,
  f32 -> 64), via :func:`lanes_for`.
"""

from ._type_contract import ParameterRule, TypeContract, check_type_contract

__all__ = [
    "check_element_type",
    "check_conversion",
    "VL_BITS",
    "LOAD_DISTS",
    "STORE_DISTS",
    "PGE_PATTERNS",
    "VCVT_PARTS",
    "MODES",
    "CMP_TYPES",
    "check_dist",
    "check_part",
    "check_pge_pattern",
    "check_mode",
    "check_cmp",
    "check_total_bits",
    "lanes_for",
]

#: Total bits of one physical vector register (hivm::util::VL_BITS).
VL_BITS = 2048

# AVE LoadDist whitelist (strings accepted by symbolizeLoadDist).
LOAD_DISTS = frozenset([
    "NORM",
    "UNPK_B8",
    "UNPK_B16",
    "UNPK4_B8",
    "BLK",
    "BRC_B8",
    "BRC_B16",
    "BRC_B32",
    "BRC_B64",
    "DINTLV_B8",
    "DINTLV_B16",
    "DINTLV_B32",
    "BDINTLV",
    "UNPK_B32",
    "E2B_B16",
    "E2B_B32",
    "SPLT2CHN_B8",
    "SPLT2CHN_B16",
    "SPLT4CHN_B8",
    "US_B8",
    "US_B16",
    "DS_B8",
    "DS_B16",
    "US",
    "DS",
    "PK",
])

# AVE StoreDist whitelist (strings accepted by symbolizeStoreDist).
STORE_DISTS = frozenset([
    "NORM_B8",
    "NORM_B16",
    "NORM_B32",
    "NORM_B64",
    "ONEPT_B8",
    "ONEPT_B16",
    "ONEPT_B32",
    "ONEPT_B64",
    "PK_B16",
    "PK_B32",
    "PK_B64",
    "PK4_B32",
    "INTLV_B8",
    "INTLV_B16",
    "INTLV_B32",
    "MRG4CHN_B8",
    "MRG2CHN_B8",
    "MRG2CHN_B16",
])

# AVE PgePattern whitelist (strings accepted by symbolizePgePattern).
PGE_PATTERNS = frozenset([
    "ALL",
    "VL1",
    "VL2",
    "VL3",
    "VL4",
    "VL8",
    "VL16",
    "VL32",
    "VL64",
    "VL128",
    "M3",
    "M4",
    "H",
    "Q",
    "ALLF",
])

# AVE VCVT_PartType whitelist (vextf part selector; assembly strings).
VCVT_PARTS = frozenset(["part_even", "part_odd"])

# Predicate handling modes.  The AVE register ops currently lower with
# MODE_ZEROING; MERGING is accepted by the frontend and emulated with a select
# for elementwise ops (inactive lanes keep the first source operand).
MODES = frozenset(["ZEROING", "MERGING"])

# AVE CmpType whitelist.  The unsigned variants compare the raw integer bit
# pattern and are used by vaddc and by unsigned integer operands.
CMP_TYPES = frozenset([
    "EQ",
    "NE",
    "GT",
    "GE",
    "LT",
    "LE",
    "UGT",
    "UGE",
    "ULT",
    "ULE",
])


def check_dist(dist, whitelist, what="load"):
    """Validate a distribution string against a whitelist; return it unchanged."""
    if not isinstance(dist, str) or dist not in whitelist:
        allowed = ", ".join(sorted(whitelist))
        raise ValueError(f"invalid {what} dist {dist!r}: expected one of [{allowed}]")
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
        raise ValueError(f"invalid pge pattern {pattern!r}: expected one of [{allowed}]")
    return pattern


def check_mode(mode):
    """Validate a predicate mode; return its uppercase canonical form."""
    mode = str(mode).upper()
    if mode not in MODES:
        raise ValueError(f"invalid predicate mode {mode!r}: expected ZEROING/MERGING")
    return mode


def check_cmp(cmp_type):
    """Validate a compare type; return its uppercase canonical form."""
    cmp_type = str(cmp_type).upper()
    if cmp_type not in CMP_TYPES:
        allowed = ", ".join(sorted(CMP_TYPES))
        raise ValueError(f"invalid cmp type {cmp_type!r}: expected one of [{allowed}]")
    return cmp_type


def check_total_bits(lanes, elem_bits, what="vector"):
    """Enforce the 2048-bit register invariant."""
    if lanes <= 0:
        raise ValueError(f"{what} lane count must be positive, got {lanes}")
    if lanes * elem_bits > VL_BITS:
        raise ValueError(f"{what} needs {lanes}x{elem_bits} = {lanes * elem_bits} bits, "
                         f"exceeding the {VL_BITS}-bit vector register")


def lanes_for(elem_bits):
    """Full-register lane count for an element width (2048 / elem_bits)."""
    if VL_BITS % elem_bits != 0:
        raise ValueError(f"element width {elem_bits} does not divide {VL_BITS} bits")
    return VL_BITS // elem_bits


def check_vsstb_stride(stride, elem_bits, what="vsstb"):
    """Validate the positive, 32-byte-block-aligned immediate element stride."""
    if type(stride) is not int or stride <= 0:
        raise ValueError(f"{what} stride must be a positive compile-time integer")
    if elem_bits not in (8, 16, 32):
        raise ValueError(f"{what} requires 8-, 16-, or 32-bit elements")
    block = 256 // elem_bits
    if stride % block or stride // block > 0xffff:
        raise ValueError(f"{what} stride must be a multiple of the 32B block size and fit 16 bits")
    return stride


# Per-op element-type support, derived from the AVE lowering rather than from
# the TileLang function names:
#   * HIVMAVEToAVEIntrin registries (vector-vector / unary / vector-scalar)
#     pick the *XInstr checks per element width,
#   * the regbase library (bishengir/lib/Template/.../RegBase) only provides
#     64-bit vector-vector / vector-scalar wrappers.
# "unsigned" is the frontend dtype's signedness (``dtype.is_int_unsigned()``);
# MLIR element types are signless, so the frontend must select the unsigned AVE
# op / cast attribute itself.
_OP_ELEM_SUPPORT = {
    # VFDivOp: f16/f32 lower to the VdivSX intrinsic.  HIVMAVEToStandard
    # forces 16/32-bit integer VFDivOp to the library wrappers
    # _mlir_ciface_vdiv_{int,uint}{16,32}_t (not to VdivS/VdivU intrinsics),
    # and i64/u64 use _mlir_ciface_vdiv_{int,uint}64_t.  i8/u8/bf16/f8 have
    # no lowering.
    "vdiv": {
        "float": ("f16", "f32"),
        "signed": ("i16", "i32", "i64"),
        "unsigned": ("i16", "i32", "i64"),
    },
    # VFExpOp lowers only for F16/F32 (no lib fallback: the op has no
    # getOpLibraryCallName override), so everything else is rejected up front.
    "vexp": {
        "float": ("f16", "f32"),
        "signed": (),
        "unsigned": (),
    },
    # VFAbsOp lowers to VabsXInstrOp for 8/16/32-bit elements and to the
    # 64-bit library wrapper for i64.  It is a signed absolute value, so
    # unsigned integer inputs are rejected rather than silently using signed
    # hardware semantics.
    "vabs": {
        "float": ("f16", "f32"),
        "signed": ("i8", "i16", "i32", "i64"),
        "unsigned": (),
    },
    # VFNegOp is two's-complement negation; the bit pattern is the same for
    # signed and unsigned integers, so both frontend signedness variants are
    # accepted.  i64/u64 use the dedicated 64-bit library wrappers.
    "vneg": {
        "float": ("f16", "f32"),
        "signed": ("i8", "i16", "i32", "i64"),
        "unsigned": ("i8", "i16", "i32", "i64"),
    },
    # VFNotOp is bitwise-not; all integer widths and float bit patterns are
    # accepted, including the 64-bit library wrappers.
    "vnot": {
        "float": ("f16", "f32"),
        "signed": ("i8", "i16", "i32", "i64"),
        "unsigned": ("i8", "i16", "i32", "i64"),
    },
    # VFReluOp has no dedicated 8/16/64-bit lowering; the direct RegBase
    # intrinsic covers f16/f32/i32 only.  Unsigned i32 would use signed
    # semantics, so it is rejected as well.
    "vrelu": {
        "float": ("f16", "f32"),
        "signed": ("i32", ),
        "unsigned": (),
    },
    # VFSqrtOp lowers only to the f16/f32 SFU intrinsic; there is no integer
    # or 64-bit library fallback.
    "vsqrt": {
        "float": ("f16", "f32"),
        "signed": (),
        "unsigned": (),
    },
    # VFAbsDiffOp covers f16/f32 through VabsdifSXInstrOp.  The unsigned
    # intrinsic exists in the dialect but is not wired into the lowering yet.
    "vabsdif": {
        "float": ("f16", "f32"),
        "signed": (),
        "unsigned": (),
    },
    # VFExpdifOp computes exp(lhs - rhs) and lowers only for f16/f32.
    "vexpdif": {
        "float": ("f16", "f32"),
        "signed": (),
        "unsigned": (),
    },
    # AVE_ShiftOp / AVE_ShiftVecScalarOp lower only the 8/16/32-bit
    # register forms.  The AVE TD accepts I64, but HIVMVShiftOpLowering has no
    # 64-bit path; CCE vector vshl/vshr also has no i64 overload.
    "vshl": {
        "float": (),
        "signed": ("i8", "i16", "i32"),
        "unsigned": ("i8", "i16", "i32"),
    },
    "vshr": {
        "float": (),
        "signed": ("i8", "i16", "i32"),
        "unsigned": ("i8", "i16", "i32"),
    },
    "vshls": {
        "float": (),
        "signed": ("i8", "i16", "i32"),
        "unsigned": ("i8", "i16", "i32"),
    },
    "vshrs": {
        "float": (),
        "signed": ("i8", "i16", "i32"),
        "unsigned": ("i8", "i16", "i32"),
    },
    # AVE_VFLRelusOp / AVE_VFPReluOp: f16/f32 only.
    "vlrelu": {
        "float": ("f16", "f32"),
        "signed": (),
        "unsigned": (),
    },
    "vprelu": {
        "float": ("f16", "f32"),
        "signed": (),
        "unsigned": (),
    },
    # VFMaxOp/VFMinOp cover f16/bf16 and 8/16/32-bit ints; i64/u64 go through
    # _mlir_ciface_vmax|vmin_int64_t|uint64_t.
    "vmax": {
        "float": ("f16", "bf16", "f32"),
        "signed": ("i8", "i16", "i32", "i64"),
        "unsigned": ("i8", "i16", "i32", "i64"),
    },
    "vmin": {
        "float": ("f16", "bf16", "f32"),
        "signed": ("i8", "i16", "i32", "i64"),
        "unsigned": ("i8", "i16", "i32", "i64"),
    },
    # VFMaxsOp/VMaxsUIOp: the AVE vector-scalar contract has no bf16/f8 scalar
    # slot; i64/u64 use the int64/uint64 library wrappers.
    "vmaxs": {
        "float": ("f16", "f32"),
        "signed": ("i8", "i16", "i32", "i64"),
        "unsigned": ("i8", "i16", "i32", "i64"),
    },
    # VFBroadcastScalarMaskOp lowers to the VdupsZ intrinsic, which is
    # 8/16/32-bit only.  The 64-bit library fallback
    # (_mlir_ciface_vdups_int64_t) calls vbr() and drops the predicate, so
    # i64/u64 are rejected instead of producing a silently unmasked broadcast.
    "vdup": {
        "float": ("f16", "bf16", "f32", "fp8e4nv"),
        "signed": ("i8", "i16", "i32"),
        "unsigned": ("i8", "i16", "i32"),
    },
    # VFBroadcastVectorOp (vector_broadcast) uses the same
    # _mlir_ciface_vdup_<elem> library entry as the scalar form, but the
    # 64-bit entry is a real 64-lane vector broadcast
    # (_mlir_ciface_vdup_int64_t -> ave_vdup_i64 -> vdup POS_LOWEST/HIGHEST)
    # that honours the predicate.  Unlike the VdupsZ scalar path, i64/u64 are
    # therefore representable; see the vdupv docstring for the verified
    # prefix/all-false/arbitrary simulator evidence.
    "vdupv": {
        "float": ("f16", "bf16", "f32", "fp8e4nv"),
        "signed": ("i8", "i16", "i32", "i64"),
        "unsigned": ("i8", "i16", "i32", "i64"),
    },
}


def _signed_type(tag, unsigned):
    return "u" + tag[1:] if unsigned and tag.startswith("i") else tag


_ELEMENT_CONTRACTS = {
    op:
    TypeContract(signatures=tuple(
        (tag, )
        for tag in spec["float"] + spec["signed"] + tuple(_signed_type(tag, True)
                                                          for tag in spec["unsigned"])))
    for op, spec in _OP_ELEM_SUPPORT.items()
}

# These exact pairs share CANN __VF_VCVTFI_SAT, whose static_assert permits
# R/A/F/C/Z only (__clang_cce_vector_intrinsics_legacy.h, CANN 9.3 B066).
# AVE forwards rnd directly to VcvtfiF322S32 / VcvtfiF162S16. Other pairs
# retain their existing checks until their complete contracts are established.
_ROUND_NAMES = dict(
    zip(("rint", "round", "floor", "ceil", "trunc", "odd"),
        ("ROUND.R", "ROUND.A", "ROUND.F", "ROUND.C", "ROUND.Z", "ROUND.O")))
_INT_ROUNDS = ("ROUND.R", "ROUND.A", "ROUND.F", "ROUND.C", "ROUND.Z")
_CONVERSION_CONTRACT = TypeContract(parameters=(
    ParameterRule(("f32", "i32"), "round", _INT_ROUNDS),
    ParameterRule(("f16", "i16"), "round", _INT_ROUNDS),
))


def check_element_type(op, tag, unsigned=False):
    """Apply an existing elementwise contract without losing signedness."""
    check_type_contract(op, (_signed_type(tag, unsigned), ), _ELEMENT_CONTRACTS[op])


def check_conversion(src, dst, src_unsigned, dst_unsigned, rounding):
    """Apply the established type-dependent restrictions for vcvt."""
    types = (_signed_type(src, src_unsigned), _signed_type(dst, dst_unsigned))
    check_type_contract("tla.simd.vcvt", types, _CONVERSION_CONTRACT, options={"round": _ROUND_NAMES[rounding]})
