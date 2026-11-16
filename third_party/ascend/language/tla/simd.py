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
"""
SIMD vector-register primitives for the engram_fused_weight decode chains.

Each op maps 1:1 onto an AVE dialect op (``ave.hir.*``) built through
``ascend_ir.cc`` bindings:

============================  ==========================================
API                           AVE op
============================  ==========================================
``vlds``                      ``ave.hir.vload``
``vsts``                      ``ave.hir.masked_store``
``vextf``                     ``ave.hir.vextf``
``vintlv``                    ``ave.hir.vintlv``
``vbitcast``                  ``vector.bitcast``
``vmul``                      ``ave.hir.vmul`` (mask = ``ave.hir.pge <ALL>``)
``vdiv``                      ``ave.hir.vdiv`` (``cast`` selects signedness)
``vmax`` / ``vmin``           ``ave.hir.vmax``/``vmin``, ``vumax``/``vumin`` if unsigned
``vmaxs``                     ``ave.hir.vmaxs`` / ``ave.hir.vumaxs``
``vexp``                      ``ave.hir.vexp``
``vdup`` / ``vdups``          ``ave.hir.broadcast`` (masked ``vdups``)
``vdupv``                     ``ave.hir.vector_broadcast`` (lowest/highest lane)
============================  ==========================================

Element-type support is per op (``_OP_ELEM_SUPPORT``) because the AVE
lowering -- not the name -- decides: ``vexp`` is f16/f32 only, scalar
``vdup``/``vdups`` are 8/16/32-bit only (the 64-bit scalar wrapper drops the
predicate), ``vdupv`` additionally has a real 64-lane i64/u64 form, and
``vmaxs`` has no bf16/f8 scalar slot.  ``mode`` accepts both
``MODE.ZEROING`` and ``MODE.MERGING``, and the normalized mode is forwarded to
the TLA C++ builder, which consumes it for real (``tla_builder.cc``
``applySimdMode``): ``ZEROING`` returns the bare masked AVE op, while
``MERGING`` materializes a same-typed ``+0`` SSA vector
(``arith.constant`` zero + ``ave.hir.scalar_broadcast``) and selects it for the
inactive lanes with ``ave.hir.vsel``.  The merge source therefore defaults to
``+0``, so both modes read inactive lanes as zero; ``MODE.MERGING`` does **not**
map onto CANN's native dst-preserving predicate and does not keep the old
destination / ``src0`` lanes.  Use an explicit
``vsel(mask, op(...), old_value)`` when inactive lanes must keep their previous
value.  The 16/32-bit unsigned integer division software sequence does not
itself guarantee zero inactive lanes, so ``ZEROING`` appends a masked select
when a caller mask is supplied; ``MERGING``'s builder-side select covers the
same case, so the two select paths never stack.
``precision`` accepts the hardware-SFU tier (``None``/``'ftz_true'``) and, for
**f32 vdiv only**, the precise tiers (``'exact'`` / ``'vdiv_0ulp_ftz_true'``),
which map onto ``ave.hir.vdivfhp``.  ``vexp``'s ``'ftz_false'`` software tier
still has no AVE lowering and raises ``NotImplementedError``.

The two supported decode chains (see ``TileKernels``
``quant/cast_back_asc.py`` / ``engram/engram_fused_weight_asc.py``):

- BF16: ``vlds(dist='UNPK_B16')`` -> ``vextf(f32, part='part_even')`` -> ``vmul``
- FP8 E4M3FN: ``vlds(dist='UNPK4_B8', elem=fp8e4nv)`` -> ``vextf(f32)`` -> ``vmul``
- store: ``vsts(..., dist='NORM_B32')``

Most full registers span 2048 bits; i64/u64 use 4096 bits. Conversion
result shapes follow their AVE operation contracts.  Dist/part/
pp/pge strings are validated by :mod:`simd_semantic`.
"""

from __future__ import annotations

import enum

import triton.language as tl
from .core import buffer, buffer_access, vector, vector_type
from triton.language.core import builtin, tensor, _unwrap_if_constexpr

from . import simd_semantic as ss

__all__ = [
    "MODE",
    "CMP",
    "POS",
    "vlds",
    "vsts",
    "vsstb",
    "vsub",
    "vor",
    "vmins",
    "vextf",
    "vintlv",
    "vdintlv",
    "vbitcast",
    "vmul",
    "vadd",
    "vadds",
    "vaddc",
    "vdiv",
    "vmax",
    "vmin",
    "vmaxs",
    "vexp",
    "vdup",
    "vdups",
    "vdupv",
    "vxor",
    "vmod",
    "vfma",
    "vsel",
    "vneg",
    "vsqrt",
    "vrsqrt",
    "vexp",
    "vreduce",
    "vcmax",
    "vcmin",
    "vcmp",
    "vbroadcast",
    "vfull",
    "vsplat",
    "vtruncf",
    "vreduce_sum",
    "vreduce_max",
    "vreduce_min",
    "vci",
    "pge",
    "membar",
]


def _lanes_for(tag):
    """Full-register lane count; i64/u64 span 4096 bits (2x VL_BITS)."""
    bits = _ELEM_SPECS[tag]
    if bits == 64:
        return 2 * ss.VL_BITS // 64
    return ss.lanes_for(bits)


# Precise element tags accepted by the AVE builder (ascend_ir.cc
# resolveSimdElementType).  Each tag maps to exactly one MLIR element type, so
# the frontend's declared dtype and the handle's IR type stay identical.
# ``fp8e4nv`` is the Triton spelling of the E4M3FN format (1-4-3, bias 7); the
# builder resolves it to Float8E4M3FNType.
_ELEM_SPECS = {
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "bf16": 16,
    "f16": 16,
    "f32": 32,
    "fp8e4nv": 8,
    "fp8e5": 8,
}

# String spellings accepted by vbitcast()/vdup(), resolved to the canonical
# dtype names of the bundled Triton (``fp16``/``fp32``/``bf16``/``fp8e4nv``,
# ``int8``..``int64``, ``uint8``..``uint64``).  Kept as strings (not dtype
# objects) so this module does not evaluate tl.<dtype> at import time
# (extension/__init__ imports simd while triton.language is still initializing).
_DTYPE_ALIASES = {
    "fp8e5": "fp8e5",
    # TLA / C-like short spellings
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "f16": "fp16",
    "f32": "fp32",
    "bf16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "bfloat16": "bf16",
    "float8e4nv": "fp8e4nv",
    "fp8e4m3fn": "fp8e4nv",
    "float8e4m3fn": "fp8e4nv",
    # canonical spellings (identity entries keep the accepted set explicit)
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "fp16": "fp16",
    "fp32": "fp32",
    "fp8e4nv": "fp8e4nv",
}


class MODE(str, enum.Enum):
    """Predicate mode for the SIMD register ops.

    Both modes are forwarded unchanged to the TLA C++ builder, which lowers
    them explicitly (``tla_builder.cc`` ``applySimdMode``):

    * ``ZEROING`` -> the bare masked AVE op, inactive lanes read as zero;
    * ``MERGING`` -> the same masked AVE op followed by
      ``vsel(mask, result, zero_vector)``, where ``zero_vector`` is the
      same-typed merge source initialized to ``+0``.

    AVE regbase value ops have no mutable destination, so ``MERGING`` is **not**
    CANN's native dst-preserving predicate: the merge source defaults to a
    ``+0`` SSA vector and the old destination / ``src0`` lanes are not
    preserved.  Compose ``vsel(mask, op(...), old_value)`` explicitly when
    inactive lanes must keep their previous value.
    """

    ZEROING = "MODE_ZEROING"
    MERGING = "MODE_MERGING"

    def __str__(self):
        return self.value


_MODE_ZEROING_ALIASES = ("MODE_ZEROING", "ZEROING")
_MODE_MERGING_ALIASES = ("MODE_MERGING", "MERGING")
# Builder-side spelling consumed by tla_builder.cc::applySimdMode.  The mode is
# an actual lowering choice (bare op vs op + zero-init + vsel), never an
# unconsumed attribute and never rewritten back to zeroing on the way down.
_MODE_BUILDER_NAMES = {MODE.ZEROING: "zeroing", MODE.MERGING: "merging"}


class POS:
    """Lane selector for :func:`vdupv`.

    ``LOWEST`` duplicates lane 0, ``HIGHEST`` duplicates the last lane of the
    register.  These are single-lane broadcasts, not half-register copies.
    """

    LOWEST = "POS_LOWEST"
    HIGHEST = "POS_HIGHEST"


_POS_LOWEST_ALIASES = ("POS_LOWEST", "LOWEST")
_POS_HIGHEST_ALIASES = ("POS_HIGHEST", "HIGHEST")

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
        "hint": "f16/f32/i16/i32/i64 and u16/u32/u64",
    },
    # VFExpOp lowers only for F16/F32 (no lib fallback: the op has no
    # getOpLibraryCallName override), so everything else is rejected up front.
    "vexp": {
        "float": ("f16", "f32"),
        "signed": (),
        "unsigned": (),
        "hint": "f16/f32",
    },
    # VFMaxOp/VFMinOp cover f16/bf16 and 8/16/32-bit ints; i64/u64 go through
    # _mlir_ciface_vmax|vmin_int64_t|uint64_t.
    "vmax": {
        "float": ("f16", "bf16", "f32"),
        "signed": ("i8", "i16", "i32", "i64"),
        "unsigned": ("i8", "i16", "i32", "i64"),
        "hint": "f16/bf16/f32/i8/i16/i32/i64 and u8/u16/u32/u64",
    },
    "vmin": {
        "float": ("f16", "bf16", "f32"),
        "signed": ("i8", "i16", "i32", "i64"),
        "unsigned": ("i8", "i16", "i32", "i64"),
        "hint": "f16/bf16/f32/i8/i16/i32/i64 and u8/u16/u32/u64",
    },
    # VFMaxsOp/VMaxsUIOp: the AVE vector-scalar contract has no bf16/f8 scalar
    # slot; i64/u64 use the int64/uint64 library wrappers.
    "vmaxs": {
        "float": ("f16", "f32"),
        "signed": ("i8", "i16", "i32", "i64"),
        "unsigned": ("i8", "i16", "i32", "i64"),
        "hint": "f16/f32/i8/i16/i32/i64 and u8/u16/u32/u64",
    },
    # VFBroadcastScalarMaskOp lowers to the VdupsZ intrinsic, which is
    # 8/16/32-bit only.  The 64-bit library fallback
    # (_mlir_ciface_vdups_int64_t) calls vbr() and drops the predicate, so
    # i64/u64 are rejected instead of producing a silently unmasked broadcast.
    "vdup": {
        "float": ("f16", "bf16", "f32", "fp8e4nv"),
        "signed": ("i8", "i16", "i32"),
        "unsigned": ("i8", "i16", "i32"),
        "hint": "i8/i16/i32/bf16/f16/f32/fp8e4nv and u8/u16/u32",
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
        "hint": "i8/i16/i32/i64/bf16/f16/f32/fp8e4nv and u8/u16/u32/u64",
    },
}


def _elem_tag(dtype_):
    """Return the precise element tag for a supported SIMD element dtype.

    Foreign E4M3FN spellings are normalized to ``fp8e4nv``. Unsupported
    element types are rejected rather than mapped to an incorrect bit width.
    Signedness remains frontend metadata; the builder uses signless integers.
    """
    name = str(_unwrap_if_constexpr(dtype_))
    tag = {
        "fp8e5": "fp8e5",
        "fp8e4nv": "fp8e4nv",
        "float8e4nv": "fp8e4nv",
        "fp8e4m3fn": "fp8e4nv",
        "float8e4m3fn": "fp8e4nv",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "f16",
        "float16": "f16",
        "fp32": "f32",
        "float32": "f32",
        "f32": "f32",
        "int8": "i8",
        "i8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "i8",
        "uint16": "i16",
        "uint32": "i32",
        "uint64": "i64",
    }.get(name)
    if tag is None:
        raise ValueError(f"unsupported simd element dtype {dtype_}; supported: "
                         f"i8/i16/i32/i64/bf16/f16/f32/fp8e4nv")
    return tag


def _require_vector(value, name):
    if not isinstance(value, vector):
        raise TypeError(f"{name} must be a tla.vector, got {type(value).__name__}")


def _check_mask(mask, lanes):
    _require_vector(mask, "mask")
    if mask.dtype != tl.int1 or mask.type.lanes != lanes:
        raise ValueError(f"mask must be vector<{lanes}xint1>, got {mask.type}")


class CMP(enum.Enum):
    """Compare type for :func:`vcmp`."""

    EQ = "EQ"
    NE = "NE"
    GT = "GT"
    GE = "GE"
    LT = "LT"
    LE = "LE"
    UGT = "UGT"
    UGE = "UGE"
    ULT = "ULT"
    ULE = "ULE"


def _resolve_mode(mode):
    """Normalize a MODE enum / string / constexpr into its canonical name."""
    mode = _unwrap_if_constexpr(mode)
    if isinstance(mode, MODE):
        mode = mode.value
    if isinstance(mode, str) and mode.upper().startswith("MODE_"):
        mode = mode[5:]
    return ss.check_mode(mode)


def _resolve_cmp(cmp_type):
    """Normalize a CMP enum / string / constexpr into its canonical name."""
    cmp_type = _unwrap_if_constexpr(cmp_type)
    if isinstance(cmp_type, CMP):
        cmp_type = cmp_type.value
    return ss.check_cmp(cmp_type)


def _mask_for(mask, lanes, _semantic):
    """Return ``mask`` or an all-true mask register with ``lanes`` lanes."""
    mask = _unwrap_if_constexpr(mask)
    if mask is None:
        return _all_mask(lanes, _semantic)
    _check_mask(mask, lanes)
    return mask


def _masked_binary(kind, lhs, rhs, mask, mode, _semantic):
    """Elementwise binary op under an explicit mask and predicate mode."""
    return _binary(kind, lhs, rhs, _semantic, mask=mask, mode=mode)


def _resolve_base_and_indices(ptr_or_buffer, offsets, _semantic):
    """Return a memref base and scalar index handles from a TLA buffer."""
    if isinstance(ptr_or_buffer, buffer):
        base = ptr_or_buffer.handle
    else:
        raise TypeError("simd load/store expects a tla.buffer, "
                        f"got {type(ptr_or_buffer).__name__}")
    indices = []
    for off in offsets:
        off = _unwrap_if_constexpr(off)
        if isinstance(off, tensor):
            if off.type.is_block() or not off.dtype.is_int():
                raise TypeError("simd load/store indices must be scalar integers")
        else:
            if type(off) is not int:
                raise TypeError("simd load/store indices must be scalar integers")
            off = _semantic.to_tensor(off)
        indices.append(off.handle)
    return base, indices


def _all_mask(lanes, _semantic):
    ty = vector_type(lanes, tl.int1)
    handle = _semantic.builder.create_simd_pge("ALL", lanes)
    return vector(handle, ty)


def _resolve_mask(mask, lanes, _semantic):
    """Return ``mask`` after checking it, or an all-true mask when omitted."""
    if mask is None:
        return _all_mask(lanes, _semantic)
    _check_mask(mask, lanes)
    return mask


def _resolve_pos(pos):
    """Return True for LOWEST and False for HIGHEST; reject anything else."""
    name = str(_unwrap_if_constexpr(pos)).upper()
    if name in _POS_LOWEST_ALIASES:
        return True
    if name in _POS_HIGHEST_ALIASES:
        return False
    raise ValueError(f"invalid pos {pos!r}: only POS_LOWEST/POS_HIGHEST are supported")


def _check_mode(mode):
    """Validate ``mode`` and return the normalized ``MODE.*`` constant.

    The two modes are distinct all the way down to the C++ builder: ZEROING
    emits the bare masked AVE op, MERGING emits the op plus an explicit
    zero-initialized merge source selected by ``ave.hir.vsel``.  Never rewrite
    MERGING to ZEROING here -- that would silently discard the caller's
    request before the backend can lower it.
    """
    mode = _unwrap_if_constexpr(mode)
    name = (mode.value if isinstance(mode, MODE) else str(mode)).upper()
    if name in _MODE_ZEROING_ALIASES:
        return MODE.ZEROING
    if name in _MODE_MERGING_ALIASES:
        return MODE.MERGING
    raise ValueError(f"invalid mode {mode!r}: supported: "
                     "MODE_ZEROING, MODE_MERGING")


# Precision selectors.  The hardware tier keeps the bare SFU instruction.
# vdiv's precise tier maps onto the real AVE high-precision division op
# (VFDivFHPOp / ave.hir.vdivfhp), whose pinned regbase lowering
# (bishengir/lib/Template/lib/RegBase/Vector/Vdiv.cpp: vdiv_float_hp) is the
# same 0ULP error-correction sequence TileLang exposes as
# vdiv_0ulp_ftz_true.  It is f32-only, exactly like TileLang's
# ``_precision_annotation`` (the C++ wrapper hardcodes float32).
# vexp's ftz_false tier would need the multi-op vexp_1ulp_ftz_false algorithm;
# it has no committed AVE lowering yet, so it is rejected explicitly instead of
# silently falling back to the FTZ hardware instruction.
_PRECISION_HW = ("FTZ_TRUE", )
_PRECISION_VDIV_EXACT = ("EXACT", "VDIV_0ULP_FTZ_TRUE")
_PRECISION_VEXP_FTZ_FALSE = ("FTZ_FALSE", "VEXP_1ULP_FTZ_FALSE")


def _normalize_precision(precision, op):
    """Validate a per-op precision selector.

    Returns ``None`` for the hardware SFU tier and ``'exact'`` for vdiv's
    precise tier.  vexp's subnormal-preserving tier has a concrete TileLang
    algorithm but no AVE IR yet, so it raises instead of degrading silently.
    """
    if precision is None:
        return None
    name = str(_unwrap_if_constexpr(precision)).upper()
    if name in _PRECISION_HW:
        return None
    if op == "vdiv" and name in _PRECISION_VDIV_EXACT:
        return "exact"
    if op == "vexp" and name in _PRECISION_VEXP_FTZ_FALSE:
        raise NotImplementedError(f"{op}: precision={precision!r} needs TileLang's "
                                  "vexp_1ulp_ftz_false algorithm (exp(x/2)^2 + a subnormal-boundary "
                                  "select); it has no AVE IR representation yet, so it is rejected "
                                  "instead of emitting the FTZ hardware instruction. supported: "
                                  "precision=None, 'ftz_true'")
    raise ValueError(f"{op}: invalid precision {precision!r}; "
                     f"supported: None, 'ftz_true'" + (", 'exact'/'vdiv_0ulp_ftz_true'" if op == "vdiv" else ""))


def _is_unsigned(dtype_):
    """Frontend signedness of an integer dtype (False for floats)."""
    is_unsigned = getattr(dtype_, "is_int_unsigned", None)
    return bool(is_unsigned()) if is_unsigned is not None else False


def _require_elem_support(op, tag, unsigned):
    """Reject element types the AVE lowering cannot express for ``op``."""
    table = _OP_ELEM_SUPPORT[op]
    if tag in ("f16", "bf16", "f32", "fp8e4nv"):
        supported = tag in table["float"]
    else:
        supported = tag in (table["unsigned"] if unsigned else table["signed"])
    if not supported:
        sign = " (unsigned)" if unsigned else ""
        raise ValueError(f"{op} does not support element type {tag}{sign}; "
                         f"supported: {table['hint']}")


def _resolve_element_dtype(dtype_):
    """Return ``(elem_tag, tl.dtype)`` for a dtype object or dtype name.

    Strings are resolved through :data:`_DTYPE_ALIASES`, so the TLA short
    spellings used elsewhere in this module (``i8``/``i32``/``f16``/``u32``)
    work alongside canonical Triton names (``int32``/``fp16``/``uint32``).
    Signedness is preserved: a name resolving to ``uint*`` keeps a
    ``tl.uint*`` dtype object, so ``_is_unsigned`` still selects the unsigned
    AVE op.  Unknown names raise ``ValueError`` instead of leaking the
    ``tl.dtype`` assertion.
    """
    value = _unwrap_if_constexpr(dtype_)
    if isinstance(value, tl.dtype):
        return _elem_tag(value), value
    name = _DTYPE_ALIASES.get(str(value))
    if name is None:
        valid = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(f"unsupported vdup dtype {value!r}; expected a "
                         f"tl.dtype or one of: {valid}")
    dtype_obj = tl.dtype(name)
    return _elem_tag(dtype_obj), dtype_obj


def _make_scalar(scalar, dtype, tag, op, _semantic):
    """Materialize a scalar operand whose type is exactly ``dtype``.

    Integers go through ``create_simd_scalar_int`` so that i64 values are not
    rounded through a Python double; ``create_simd_scalar`` (float path) is
    left untouched for vsplat/vfull compatibility.
    """
    value = _unwrap_if_constexpr(scalar)
    if isinstance(value, tensor):
        if value.type.is_block():
            raise TypeError(f"{op} scalar must be a scalar value, got {value.type}")
        if value.dtype != dtype:
            if value.dtype.is_int() and not value.dtype.is_bool() and dtype.is_int():
                return _semantic.cast(value, dtype).handle
            raise TypeError(f"{op} scalar dtype must match the vector element "
                            f"type {dtype}, got {value.dtype}")
        return value.handle
    if isinstance(value, bool):
        raise TypeError(f"{op} scalar must be a number, got bool")
    if isinstance(value, int):
        if tag.startswith("i"):
            bits = _ELEM_SPECS[tag]
            if _is_unsigned(dtype):
                in_range = 0 <= value < (1 << bits)
                bounds = f"[0, 2**{bits})"
            else:
                in_range = -(1 << (bits - 1)) <= value < (1 << (bits - 1))
                bounds = f"[-2**{bits - 1}, 2**{bits - 1})"
            if not in_range:
                raise ValueError(f"{op} scalar {value} does not fit {dtype} ({bounds})")
            # The exact binding takes an int64_t.  Unsigned 64-bit values in
            # [2**63, 2**64) are passed as the equivalent two's-complement
            # Python int so IntegerAttr gets the exact bits (no float path).
            if bits == 64 and _is_unsigned(dtype) and value >= (1 << 63):
                value -= 1 << 64
            return _semantic.builder.create_simd_scalar_int(tag, value)
        if dtype.is_floating():
            return _semantic.builder.create_simd_scalar(tag, float(value))
        raise TypeError(f"{op} scalar must be numeric, got {type(value).__name__}")
    if isinstance(value, float):
        if not dtype.is_floating():
            raise TypeError(f"{op} scalar {value!r} is a float but the vector "
                            f"element type is {dtype}; pass an int instead")
        return _semantic.builder.create_simd_scalar(tag, value)
    raise TypeError(f"{op} scalar must be a constexpr number or a scalar tensor, "
                    f"got {type(value).__name__}")


@builtin
def vlds(ptr, offsets=None, *, dist="NORM", elem=None, _semantic=None) -> vector:
    """
    Vector-register load from UB (``ave.hir.vload``).

    :param ptr: ``tla.buffer`` in UB, or a subscripted element range
        ``buf[i0, i1, ...]`` whose indices become the offsets
    :param offsets: index values (int / constexpr / tensor), one per memref
        dim; omit when ``ptr`` is a subscripted buffer
    :param dist: load distribution; defaults to ``NORM``
    :param elem: result element dtype; defaults to the buffer element type.
        Use ``tl.float8e4nv`` (E4M3FN) for the packed FP8 profile.
    :return: full vector register (2048 bits, or 4096 bits for i64/u64)
    """
    if isinstance(ptr, buffer_access):
        ptr, offsets = ptr.buffer, list(ptr.indices)
    elif offsets is None:
        raise TypeError("vlds needs an index: pass buf[i, ...] or explicit offsets")
    if not isinstance(ptr, buffer):
        raise TypeError("simd.vlds expects a tla.buffer")
    elem_ty = ptr.dtype if elem is None else elem
    dist = _unwrap_if_constexpr(dist)
    elem_tag = _elem_tag(elem_ty)
    lanes = _lanes_for(elem_tag)
    ss.check_dist(dist, ss.LOAD_DISTS, "load")
    base, indices = _resolve_base_and_indices(ptr, offsets, _semantic)
    handle = _semantic.builder.create_simd_vlds(base, indices, elem_tag, lanes, dist)
    return vector(handle, vector_type(lanes, elem_ty))


@builtin
def vsts(ptr, offsets_or_val, val=None, *, dist=None, mask=None, _semantic=None):
    """
    Vector-register masked store to UB (``ave.hir.masked_store``).

    :param ptr: ``tla.buffer`` in UB, or a subscripted element range
        ``buf[i0, i1, ...]`` whose indices become the offsets
    :param offsets_or_val: index values (int / constexpr / tensor), one per
        memref dim; or the vector register to store when ``ptr`` is a
        subscripted buffer
    :param val: vector register to store
    :param dist: store distribution; defaults to ``NORM_B32``
    :param mask: predicate vector; defaults to an all-true ``pge <ALL>`` mask
    """
    if isinstance(ptr, buffer_access):
        ptr, offsets = ptr.buffer, list(ptr.indices)
        val = offsets_or_val
    elif val is None:
        raise TypeError("vsts needs a value: vsts(buf[i, ...], val)")
    else:
        offsets = offsets_or_val
    if not isinstance(ptr, buffer):
        raise TypeError("simd.vsts expects a tla.buffer")
    _require_vector(val, "val")
    val_lanes = val.shape[0]
    dist = _unwrap_if_constexpr(dist)
    if dist is None:
        dist = "NORM_B32"
    ss.check_dist(dist, ss.STORE_DISTS, "vsts")
    if mask is None:
        mask = _all_mask(val_lanes, _semantic)
    _check_mask(mask, val_lanes)
    base, indices = _resolve_base_and_indices(ptr, offsets, _semantic)
    _semantic.builder.create_simd_vsts(base, indices, val.handle, mask.handle, dist)


@builtin
def vextf(src, target_dtype, mask=None, part="part_even", _semantic=None) -> vector:
    """
    Widening float conversion (``ave.hir.vextf``), e.g. bf16/f8e4m3fn -> f32.

    The result is a full F32 register: ``2048 // 32 == 64`` lanes.  The ``part``
    selector (``part_even``/``part_odd``) picks which half/quarter of the
    source register the intrinsic reads; it does not change the result width.

    :param src: source vector register
    :param target_dtype: destination element dtype (f32 for the decode chains)
    :param mask: predicate vector; defaults to an all-true mask of result width
    :param part: ``part_even`` / ``part_odd`` sub-register selector
    """
    _require_vector(src, "src")
    part = ss.check_part(_unwrap_if_constexpr(part))
    out_tag = _elem_tag(target_dtype)
    if out_tag != "f32":
        raise ValueError(f"vextf (cast) only widens to f32, got {target_dtype}")
    out_lanes = ss.lanes_for(_ELEM_SPECS[out_tag])
    if mask is None:
        mask = _all_mask(out_lanes, _semantic)
    _check_mask(mask, out_lanes)
    handle = _semantic.builder.create_simd_vcvt("vextf", src.handle, mask.handle, out_tag, part)
    return vector(handle, vector_type(out_lanes, target_dtype))


def _interleave(op, src1, src2, _semantic):
    """Shared body of vintlv/vdintlv; they differ only in the builder call."""
    _require_vector(src1, "src1")
    _require_vector(src2, "src2")
    if src1.type != src2.type:
        raise ValueError(f"{op} requires same-typed operands, "
                         f"got {src1.type} and {src2.type}")
    res1, res2 = getattr(_semantic.builder, f"create_simd_{op}")(src1.handle, src2.handle)
    ty = src1.type
    return vector(res1, ty), vector(res2, ty)


@builtin
def vintlv(src1, src2, _semantic=None):
    """
    Pairwise interleave of two vector registers (``ave.hir.vintlv``,
    ``Layout_Change::SPARSE``), returning the two result registers.
    """
    return _interleave("vintlv", src1, src2, _semantic)


@builtin
def vdintlv(src1, src2, _semantic=None):
    """
    Pairwise deinterleave of two vector registers (``ave.hir.vdintlv``,
    ``Layout_Change::SPARSE``), the inverse of :func:`vintlv`.
    """
    return _interleave("vdintlv", src1, src2, _semantic)


@builtin
def vbitcast(src, dtype_str, _semantic=None) -> vector:
    """
    Bit reinterpretation of a vector register (``vector.bitcast``).

    :param dtype_str: destination element type, e.g. ``'f32'`` or ``'bf16'``
    :return: vector with ``src_lanes * src_bits / dst_bits`` lanes
    """
    _require_vector(src, "src")
    dtype_str = _unwrap_if_constexpr(dtype_str)
    target_name = _DTYPE_ALIASES.get(dtype_str, dtype_str)
    target = tl.dtype(target_name)
    src_tag = _elem_tag(src.dtype)
    src_bits = _ELEM_SPECS[src_tag]
    out_tag = _elem_tag(target)
    out_bits = _ELEM_SPECS[out_tag]
    total_bits = src.shape[0] * src_bits
    if total_bits % out_bits != 0:
        raise ValueError(f"bitcast {src.shape[0]}x{src_bits} bits does not divide {out_bits}")
    out_lanes = total_bits // out_bits
    ss.check_total_bits(out_lanes, out_bits, "bitcast result")
    handle = _semantic.builder.create_simd_bitcast(src.handle, out_tag, out_lanes)
    return vector(handle, vector_type(out_lanes, target))


def _binary(name, lhs, rhs, _semantic, mask=None, mode=MODE.ZEROING):
    """Shared body of the elementwise vector binary ops (``ave.hir.v*``)."""
    _require_vector(lhs, "lhs")
    _require_vector(rhs, "rhs")
    if lhs.type != rhs.type:
        raise ValueError(f"{name} requires same-typed operands, "
                         f"got {lhs.type} and {rhs.type}")
    mode = _check_mode(mode)
    lanes = lhs.shape[0]
    mask = _resolve_mask(mask, lanes, _semantic)
    handle = _semantic.builder.create_simd_binary(name, lhs.handle, rhs.handle, mask.handle, _MODE_BUILDER_NAMES[mode])
    return vector(handle, lhs.type)


@builtin
def vmul(lhs, rhs, _semantic=None) -> vector:
    """
    Elementwise vector multiply (``ave.hir.vmul``) under an all-true
    ``ave.hir.pge <ALL>`` mask.
    """
    return _binary("mul", lhs, rhs, _semantic)


@builtin
def vadd(src0: vector, src1: vector, mask: vector = None, mode: tl.constexpr = MODE.ZEROING, _semantic=None) -> vector:
    """
    Elementwise vector add (``ave.hir.vadd``).

    :param src0: first vector register
    :param src1: second vector register (same type as ``src0``)
    :param mask: predicate register; defaults to an all-true mask
    :param mode: ``MODE.ZEROING`` (default) or ``MODE.MERGING``.  MERGING
        materializes a same-typed ``+0`` merge source and selects it for the
        inactive lanes; it does not preserve ``src0``.
    """
    return _masked_binary("add", src0, src1, mask, mode, _semantic)


def _scalar_handle(src, scalar, _semantic):
    """Build or validate the scalar operand of a vector-scalar op."""
    if isinstance(scalar, vector):
        if scalar.shape[0] != 1:
            raise ValueError(f"scalar register must have one lane, got {scalar.shape[0]}")
        if scalar.dtype != src.dtype:
            raise TypeError(f"scalar dtype {scalar.dtype} does not match src dtype "
                            f"{src.dtype}")
        return scalar.handle
    if isinstance(scalar, tensor):
        if len(scalar.shape) != 0:
            raise ValueError(f"scalar must be rank-0, got shape {list(scalar.shape)}")
        if str(scalar.dtype) != str(src.dtype):
            raise TypeError(f"scalar dtype {scalar.dtype} does not match src dtype "
                            f"{src.dtype}")
        return scalar.handle
    scalar = _unwrap_if_constexpr(scalar)
    dtype = str(src.dtype)
    if dtype.startswith(("int", "uint")):
        if not isinstance(scalar, int) or isinstance(scalar, bool):
            raise TypeError(f"scalar must be an integer for {dtype}, "
                            f"got {type(scalar).__name__}")
        bits = _ELEM_SPECS[_elem_tag(src.dtype)]
        if dtype.startswith("uint"):
            lo, hi = 0, (1 << bits) - 1
        else:
            lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        if not lo <= scalar <= hi:
            raise ValueError(f"scalar {scalar} does not fit in {dtype}")
    elif not isinstance(scalar, (int, float)) or isinstance(scalar, bool):
        raise TypeError(f"scalar must be numeric, got {type(scalar).__name__}")
    return _semantic.builder.create_simd_scalar(_elem_tag(src.dtype), float(scalar))


@builtin
def vadds(src: vector, scalar, mask: vector = None, mode: tl.constexpr = MODE.ZEROING, _semantic=None) -> vector:
    """
    Elementwise vector-scalar add (``ave.hir.vadds``).

    :param src: source vector register
    :param scalar: Python scalar or rank-0 tensor of the register's dtype
    :param mask: predicate register; defaults to an all-true mask
    :param mode: ``MODE.ZEROING`` (default) or ``MODE.MERGING``
    """
    _require_vector(src, "src")
    lanes = src.type.lanes
    scalar_handle = _scalar_handle(src, scalar, _semantic)
    mode = _check_mode(mode)
    mask = _resolve_mask(mask, lanes, _semantic)
    handle = _semantic.builder.create_simd_binary_scalar("adds", src.handle, scalar_handle, mask.handle,
                                                         _MODE_BUILDER_NAMES[mode])
    return vector(handle, src.type)


@builtin
def vaddc(src0: vector, src1: vector, mask: vector = None, _semantic=None) -> tuple[vector, vector]:
    """
    Vector add with carry-out (``vaddc``) for i32/u32 registers.

    Returns ``(sum, carry)`` where ``sum = src0 + src1`` (modulo 2**32) and
    ``carry`` is an i1 mask holding the two's-complement carry-out of each
    lane, derived as ``sum <u src0``.  The AVE dialect has no dedicated
    ``vaddc`` op, so the result is always zeroing.
    """
    dtype = str(src0.dtype)
    if dtype not in ("int32", "uint32"):
        raise ValueError(f"vaddc only supports int32/uint32, got {dtype}")
    if src1.dtype != src0.dtype:
        raise TypeError(f"vaddc operands must have the same dtype, "
                        f"got {src0.dtype} and {src1.dtype}")
    lanes = src0.type.lanes
    mask = _mask_for(mask, lanes, _semantic)
    total = _masked_binary("add", src0, src1, mask, MODE.ZEROING, _semantic)
    carry = vcmp(total, src0, mask=mask, op=CMP.ULT, _semantic=_semantic)
    return total, carry


@builtin
def vdiv(src0, src1, mask=None, mode=MODE.ZEROING, precision=None, _semantic=None) -> vector:
    """
    Elementwise vector divide (``ave.hir.vdiv``).

    Unsigned integer operands select the unsigned library form through the
    op's ``cast`` attribute (``cast = #hivm.cast<cast_unsigned>``); signed and
    floating point operands keep the default signed cast.

    :param src0: dividend vector register
    :param src1: divisor vector register, same type as ``src0``
    :param mask: predicate vector; defaults to an all-true ``pge <ALL>`` mask
    :param mode: ``MODE.ZEROING`` emits the bare masked AVE op;
        ``MODE.MERGING`` emits the same op plus an ``ave.hir.vsel`` that
        merges a same-typed ``+0`` SSA vector into the inactive lanes.
        MERGING is not CANN's native dst-preserving predicate and does not keep
        the old dst/``src0`` lanes (see :class:`MODE`)
    :param precision: ``None``/``'ftz_true'`` keep the hardware SFU
        instruction.  ``'exact'``/``'vdiv_0ulp_ftz_true'`` select the real
        high-precision AVE op (``ave.hir.vdivfhp``, VFDivFHPOp), whose pinned
        regbase lowering is TileLang's 0ULP ``vdiv_0ulp_ftz_true`` sequence;
        it is f32-only, matching TileLang, and f16 raises ``ValueError``.  The
        caller mask is forwarded unchanged.  Note this differs from TileLang,
        where ``precision=None`` follows the global fast-math flag; TLA has no
        such flag and always emits the hardware instruction.
    :return: vector register, same type as ``src0``
    Inactive lanes are ``+0`` for both modes (there is no dst to preserve):
    ZEROING relies on the masked AVE op, MERGING explicitly selects a ``+0``
    merge source in the C++ builder.  For unsigned 16/32-bit operands with a
    caller-supplied mask, ZEROING additionally appends an explicit masked
    select because the underlying software division sequence does not zero
    inactive lanes by itself; MERGING reuses its builder-side merge select
    instead of stacking a second one.
    Supported element types: f16/f32/i16/i32/i64 (u16/u32/u64).
    """
    _require_vector(src0, "src0")
    _require_vector(src1, "src1")
    if src0.type != src1.type:
        raise ValueError(f"vdiv requires same-typed operands, "
                         f"got {src0.type} and {src1.type}")
    mode = _check_mode(mode)
    mode_name = _MODE_BUILDER_NAMES[mode]
    precise = _normalize_precision(precision, "vdiv")
    tag = _elem_tag(src0.dtype)
    unsigned = _is_unsigned(src0.dtype)
    _require_elem_support("vdiv", tag, unsigned)
    lanes = src0.shape[0]
    explicit_mask = mask is not None
    mask = _resolve_mask(mask, lanes, _semantic)
    if precise == "exact":
        # VFDivFHPOp has a float-only library wrapper, the same restriction as
        # TileLang's fp32-only precise path.  Reject other dtypes before the
        # builder so the failure is a clear ValueError, not a missing symbol.
        if tag != "f32":
            raise ValueError(f"vdiv: precision={precision!r} requires float32 operands "
                             f"(got {src0.dtype}); non-fp32 always uses the hardware instruction")
        handle = _semantic.builder.create_simd_binary("divfhp", src0.handle, src1.handle, mask.handle, mode_name)
        return vector(handle, src0.type)
    kind = "divu" if unsigned else "div"
    handle = _semantic.builder.create_simd_binary(kind, src0.handle, src1.handle, mask.handle, mode_name)
    res = vector(handle, src0.type)
    # The 16/32-bit unsigned division software sequence computes its quotient
    # under PAT_ALL and returns it unchanged for MODE_ZEROING, so inactive lanes
    # are not guaranteed to be zero.  Make the ZEROING inactive lanes explicit
    # when the caller supplied a mask (no select for the default all-true mask).
    # MERGING is not handled here: the C++ builder already appends its
    # zero-init + vsel merge for that mode, and adding a second select here
    # would duplicate the same predicate/zero pair.
    if mode == MODE.ZEROING and explicit_mask and unsigned and tag in ("i16", "i32"):
        zero = vfull(0, src0.dtype, lanes, _semantic=_semantic)
        res = vsel(mask, res, zero, _semantic=_semantic)
    return res


def _minmax(op, signed_kind, unsigned_kind, src0, src1, mask, mode, _semantic):
    """Shared body of vmax/vmin: float/signed -> VFMax/VFMin, unsigned -> U op."""
    _require_vector(src0, "src0")
    _require_vector(src1, "src1")
    if src0.type != src1.type:
        raise ValueError(f"{op} requires same-typed operands, "
                         f"got {src0.type} and {src1.type}")
    mode = _check_mode(mode)
    tag = _elem_tag(src0.dtype)
    unsigned = _is_unsigned(src0.dtype)
    _require_elem_support(op, tag, unsigned)
    lanes = src0.shape[0]
    mask = _resolve_mask(mask, lanes, _semantic)
    kind = unsigned_kind if unsigned else signed_kind
    handle = _semantic.builder.create_simd_binary(kind, src0.handle, src1.handle, mask.handle,
                                                  _MODE_BUILDER_NAMES[mode])
    return vector(handle, src0.type)


@builtin
def vmax(src0, src1, mask=None, mode=MODE.ZEROING, _semantic=None) -> vector:
    """
    Elementwise vector maximum (``ave.hir.vmax``).

    Floating point and signed integers use ``ave.hir.vmax``; unsigned integers
    use ``ave.hir.vumax`` so the comparison is not silently signed.

    :param mask: predicate vector; defaults to an all-true mask
    :param mode: ``MODE.ZEROING`` emits the bare masked AVE op;
        ``MODE.MERGING`` emits the same op plus an ``ave.hir.vsel`` that
        merges a same-typed ``+0`` SSA vector into the inactive lanes.
        MERGING is not CANN's native dst-preserving predicate and does not keep
        the old dst/``src0`` lanes (see :class:`MODE`)
    Inactive lanes are ``+0`` for both modes: ZEROING relies on the masked AVE
    op, MERGING explicitly selects a ``+0`` merge source in the C++ builder.
    Supported element types: f16/bf16/f32/i8/i16/i32/i64 (u8/u16/u32/u64).
    """
    return _minmax("vmax", "max", "umax", src0, src1, mask, mode, _semantic)


@builtin
def vmin(src0, src1, mask=None, mode=MODE.ZEROING, _semantic=None) -> vector:
    """
    Elementwise vector minimum (``ave.hir.vmin``).

    Floating point and signed integers use ``ave.hir.vmin``; unsigned integers
    use ``ave.hir.vumin``.

    :param mask: predicate vector; defaults to an all-true mask
    :param mode: ``MODE.ZEROING`` emits the bare masked AVE op;
        ``MODE.MERGING`` emits the same op plus an ``ave.hir.vsel`` that
        merges a same-typed ``+0`` SSA vector into the inactive lanes.
        MERGING is not CANN's native dst-preserving predicate and does not keep
        the old dst/``src0`` lanes (see :class:`MODE`)
    Inactive lanes are ``+0`` for both modes: ZEROING relies on the masked AVE
    op, MERGING explicitly selects a ``+0`` merge source in the C++ builder.
    Supported element types: f16/bf16/f32/i8/i16/i32/i64 (u8/u16/u32/u64).
    """
    return _minmax("vmin", "min", "umin", src0, src1, mask, mode, _semantic)


@builtin
def vmaxs(src0, scalar, mask=None, mode=MODE.ZEROING, _semantic=None) -> vector:
    """
    Vector-vs-scalar maximum (``ave.hir.vmaxs`` / ``ave.hir.vumaxs``).

    The scalar operand must have exactly the vector's element type; it is
    either a constexpr number or a same-typed scalar tensor.  Constexpr
    integers are materialized with an exact integer constant (no Python double
    round-trip), so i64 values above 2**53 stay exact.  Unsigned 64-bit
    constexprs are passed as the equivalent two's-complement i64 bits, so the
    whole ``[0, 2**64)`` range is accepted (and out-of-range values raise
    ``ValueError``).

    :param scalar: constexpr number or scalar tensor, element type of ``src0``
    :param mask: predicate vector; defaults to an all-true mask
    :param mode: ``MODE.ZEROING`` emits the bare masked AVE op;
        ``MODE.MERGING`` emits the same op plus an ``ave.hir.vsel`` that
        merges a same-typed ``+0`` SSA vector into the inactive lanes.
        MERGING is not CANN's native dst-preserving predicate and does not keep
        the old dst/``src0`` lanes (see :class:`MODE`)
    Inactive lanes are ``+0`` for both modes: ZEROING relies on the masked AVE
    op, MERGING explicitly selects a ``+0`` merge source in the C++ builder.
    Supported element types: f16/f32/i8/i16/i32/i64 (u8/u16/u32/u64); the AVE
    vector-scalar contract has no bf16/f8 scalar slot.
    """
    _require_vector(src0, "src0")
    mode = _check_mode(mode)
    tag = _elem_tag(src0.dtype)
    unsigned = _is_unsigned(src0.dtype)
    _require_elem_support("vmaxs", tag, unsigned)
    lanes = src0.shape[0]
    mask = _resolve_mask(mask, lanes, _semantic)
    scalar_handle = _make_scalar(scalar, src0.dtype, tag, "vmaxs", _semantic)
    kind = "umaxs" if unsigned else "maxs"
    handle = _semantic.builder.create_simd_binary_scalar(kind, src0.handle, scalar_handle, mask.handle,
                                                         _MODE_BUILDER_NAMES[mode])
    return vector(handle, src0.type)


@builtin
def vexp(src, mask=None, mode=MODE.ZEROING, precision=None, _semantic=None) -> vector:
    """
    Elementwise natural exponential (``ave.hir.vexp``).

    :param mask: predicate vector; defaults to an all-true mask
    :param mode: ``MODE.ZEROING`` emits the bare masked AVE op;
        ``MODE.MERGING`` emits the same op plus an ``ave.hir.vsel`` that
        merges a same-typed ``+0`` SSA vector into the inactive lanes.
        MERGING is not CANN's native dst-preserving predicate and does not keep
        the old dst/``src0`` lanes (see :class:`MODE`)
    :param precision: ``None``/``'ftz_true'`` keep the hardware SFU
        instruction.  TileLang's ``'ftz_false'`` (``'vexp_1ulp_ftz_false'``)
        subnormal-preserving wrapper is ``exp(x/2)**2`` with a boundary select;
        TLA has the constituent ops but no committed lowering for that
        sequence, so this raises ``NotImplementedError`` instead of silently
        emitting the FTZ instruction.
    Inactive lanes are ``+0`` for both modes: ZEROING relies on the masked AVE
    op, MERGING explicitly selects a ``+0`` merge source in the C++ builder.
    Supported element types: f16/f32 (VFExpOp has no other lowering).
    """
    _require_vector(src, "src")
    mode = _check_mode(mode)
    _normalize_precision(precision, "vexp")
    tag = _elem_tag(src.dtype)
    _require_elem_support("vexp", tag, _is_unsigned(src.dtype))
    lanes = src.shape[0]
    mask = _resolve_mask(mask, lanes, _semantic)
    handle = _semantic.builder.create_simd_unary("exp", src.handle, mask.handle, _MODE_BUILDER_NAMES[mode])
    return vector(handle, src.type)


@builtin
def vdup(scalar, dtype, mask=None, mode=MODE.ZEROING, _semantic=None) -> vector:
    """
    Masked scalar broadcast: broadcast ``scalar`` to a full register
    (``ave.hir.broadcast``, i.e. the masked ``vdups`` form -- not the unmasked
    ``ave.hir.scalar_broadcast`` that :func:`vsplat` emits).

    Register width is ``2048 // elem_bits`` lanes for the supported
    8/16/32-bit elements (a full i64 register would be 4096 bits / 64 lanes,
    but i64 is rejected below).

    :param scalar: constexpr number or scalar tensor of type ``dtype``
    :param dtype: target element dtype, ``tl.dtype`` object or a dtype name
        accepted by :func:`_resolve_element_dtype` (canonical Triton names and
        the TLA ``i8``/``i32``/``f16``/``u32`` short spellings)
    :param mask: predicate vector; defaults to an all-true mask of ``lanes``
    :param mode: ``MODE.ZEROING`` emits the bare masked AVE op;
        ``MODE.MERGING`` emits the same op plus an ``ave.hir.vsel`` that
        merges a same-typed ``+0`` SSA vector into the inactive lanes.
        MERGING is not CANN's native dst-preserving predicate and does not keep
        the old dst/``src0`` lanes (see :class:`MODE`)
    Inactive lanes are ``+0`` for both modes: ZEROING relies on the masked AVE
    op, MERGING explicitly selects a ``+0`` merge source in the C++ builder.
    Supported element types: i8/i16/i32/bf16/f16/f32/fp8e4nv (u8/u16/u32).
    i64/u64 are rejected: the VdupsZ intrinsic is 8/16/32-bit only, and the
    64-bit library fallback (``_mlir_ciface_vdups_int64_t``) drops the
    predicate and would silently ignore a mask.  Use :func:`vsplat` when an
    unmasked i64/u64 broadcast is needed.
    """
    mode = _check_mode(mode)
    tag, dtype_obj = _resolve_element_dtype(dtype)
    unsigned = _is_unsigned(dtype_obj)
    _require_elem_support("vdup", tag, unsigned)
    lanes = _lanes_for(tag)
    mask = _resolve_mask(mask, lanes, _semantic)
    scalar_handle = _make_scalar(scalar, dtype_obj, tag, "vdup", _semantic)
    handle = _semantic.builder.create_simd_vdup(scalar_handle, tag, lanes, mask.handle, _MODE_BUILDER_NAMES[mode])
    return vector(handle, vector_type(lanes, dtype_obj))


@builtin
def vdups(scalar, dtype, mask=None, mode=MODE.ZEROING, _semantic=None) -> vector:
    """Masked scalar broadcast (``ave.hir.broadcast`` -> ``vdups.z``).

    Alias of :func:`vdup`: the AVE/CCE intrinsic is named ``vdups``, while the
    TileLang-facing spelling has historically been ``vdup``.  Element support,
    lane count, mask handling and mode semantics are identical to :func:`vdup`.
    """
    return vdup(scalar, dtype, mask=mask, mode=mode, _semantic=_semantic)


@builtin
def vdupv(src, mask=None, pos=POS.LOWEST, mode=MODE.ZEROING, _semantic=None) -> vector:
    """Broadcast one lane of ``src`` to the full register.

    Lowers to ``ave.hir.vector_broadcast``: ``pos=POS.LOWEST`` selects lane 0
    (``vdup.z``), ``pos=POS.HIGHEST`` selects the last lane (``vdupm.z``).
    This is a single-lane broadcast, not a half-register copy.

    :param src: source vector register
    :param mask: predicate vector; defaults to an all-true mask of ``src`` lanes
    :param pos: ``POS.LOWEST`` / ``POS.HIGHEST``
    :param mode: ``MODE.ZEROING`` emits the bare masked broadcast;
        ``MODE.MERGING`` emits it plus an ``ave.hir.vsel`` that merges a
        same-typed ``+0`` SSA vector into the inactive lanes.  MERGING is not
        CANN's native dst-preserving predicate (see :class:`MODE`)
    Supported element types: the same set as :func:`vdup` plus i64/u64.
    Unlike the masked scalar ``vdups`` path (whose 64-bit wrapper drops the
    predicate), ``vector_broadcast`` has a real 64-lane library entry
    (``_mlir_ciface_vdup_int64_t`` -> ``ave_vdup_i64``) that forwards the
    predicate and the lowest/highest selector.  i64 and u64 therefore share
    the same 64-lane ``vector<64xi64>`` / ``vector<64xi1>`` ABI; prefix,
    all-false and arbitrary masks were verified with the local simulator.
    """
    _require_vector(src, "src")
    mode = _check_mode(mode)
    tag = _elem_tag(src.dtype)
    _require_elem_support("vdupv", tag, _is_unsigned(src.dtype))
    lanes = src.shape[0]
    mask = _resolve_mask(mask, lanes, _semantic)
    low = _resolve_pos(pos)
    handle = _semantic.builder.create_simd_vbroadcast(src.handle, mask.handle, low, _MODE_BUILDER_NAMES[mode])
    return vector(handle, src.type)


@builtin
def vxor(lhs, rhs, mask=None, mode=MODE.ZEROING, _semantic=None) -> vector:
    """
    Elementwise bitwise XOR (``ave.hir.vxor``) under an all-true mask.
    """
    return _masked_binary("xor", lhs, rhs, mask, mode, _semantic)


@builtin
def vsub(lhs, rhs, mask=None, mode=MODE.ZEROING, _semantic=None) -> vector:
    """Elementwise vector subtract under an optional predicate."""
    return _masked_binary("sub", lhs, rhs, mask, mode, _semantic)


@builtin
def vor(lhs, rhs, mask=None, mode=MODE.ZEROING, _semantic=None) -> vector:
    """Elementwise bitwise OR under an optional predicate."""
    return _masked_binary("or", lhs, rhs, mask, mode, _semantic)


@builtin
def vmins(src, scalar, mask=None, mode=MODE.ZEROING, _semantic=None) -> vector:
    """Vector-scalar minimum, preserving src on inactive MERGING lanes."""
    _require_vector(src, "src")
    tag = _elem_tag(src.dtype)
    if tag not in ("i8", "i16", "i32", "f16", "f32"):
        raise ValueError(f"vmins does not support {src.dtype} registers")
    mode = _check_mode(mode)
    mask = _resolve_mask(mask, src.type.lanes, _semantic)
    scalar_handle = _make_scalar(scalar, src.dtype, tag, "vmins", _semantic)
    handle = _semantic.builder.create_simd_binary_scalar("umins" if _is_unsigned(src.dtype) else "mins", src.handle,
                                                         scalar_handle, mask.handle, _MODE_BUILDER_NAMES[mode])
    return vector(handle, src.type)


@builtin
def vsstb(src, dst, stride, mask=None, update=False, _semantic=None):
    """Store a vector to a 2-D UB buffer in 32-byte blocks at element stride."""
    if isinstance(dst, buffer_access):
        dst, offsets = dst.buffer, list(dst.indices)
    elif isinstance(dst, buffer):
        offsets = [0] * len(dst.shape)
    else:
        raise TypeError("vsstb expects a tla.buffer or subscripted buffer")
    if dst.space != "ub" or len(dst.shape) != 2 or len(offsets) != 2:
        raise ValueError("vsstb requires a two-dimensional UB buffer and two indices")
    update = _unwrap_if_constexpr(update)
    if type(update) is not bool:
        raise ValueError("vsstb update must be a compile-time bool")
    if update:
        raise NotImplementedError("vsstb update=True is not supported")
    _require_vector(src, "src")
    tag = _elem_tag(src.dtype)
    if src.dtype != dst.dtype:
        raise ValueError("vsstb source and destination element dtypes must match")
    stride = ss.check_vsstb_stride(_unwrap_if_constexpr(stride), _ELEM_SPECS[tag])
    ss.check_total_bits(src.type.lanes, _ELEM_SPECS[tag], "vsstb")
    mask = _mask_for(mask, src.type.lanes, _semantic)
    base, indices = _resolve_base_and_indices(dst, offsets, _semantic)
    _semantic.builder.create_simd_vsstb(base, indices, stride, src.handle, mask.handle)


@builtin
def vmod(lhs, rhs, _semantic=None) -> vector:
    """
    Elementwise integer modulo (``ave.hir.vmod``) under an all-true mask.
    """
    return _binary("mod", lhs, rhs, _semantic)


@builtin
def vfma(acc, lhs, rhs, _semantic=None) -> vector:
    """
    Fused multiply-add (``ave.hir.vmula``): ``acc + lhs * rhs``.
    """
    _require_vector(acc, "acc")
    _require_vector(lhs, "lhs")
    _require_vector(rhs, "rhs")
    if not (acc.type == lhs.type == rhs.type):
        raise ValueError("fma requires same-typed operands, "
                         f"got {acc.type}, {lhs.type}, {rhs.type}")
    lanes = acc.shape[0]
    mask = _all_mask(lanes, _semantic)
    handle = _semantic.builder.create_simd_vfma(acc.handle, lhs.handle, rhs.handle, mask.handle)
    return vector(handle, acc.type)


@builtin
def vsel(mask, true_val, false_val, _semantic=None) -> vector:
    """
    Bitwise select (``ave.hir.vsel``): ``mask ? true_val : false_val``.
    """
    _require_vector(true_val, "true_val")
    _require_vector(false_val, "false_val")
    if true_val.type != false_val.type:
        raise ValueError("select requires same-typed value operands, "
                         f"got {true_val.type} and {false_val.type}")
    _check_mask(mask, true_val.type.lanes)
    handle = _semantic.builder.create_simd_vsel(mask.handle, true_val.handle, false_val.handle)
    return vector(handle, true_val.type)


def _unary(name, src, _semantic):
    _require_vector(src, "src")
    lanes = src.shape[0]
    mask = _all_mask(lanes, _semantic)
    handle = _semantic.builder.create_simd_unary(name, src.handle, mask.handle)
    return vector(handle, src.type)


@builtin
def vneg(src, _semantic=None) -> vector:
    """Elementwise negate (``ave.hir.vneg``)."""
    return _unary("neg", src, _semantic)


@builtin
def vsqrt(src, _semantic=None) -> vector:
    """Elementwise square root (``ave.hir.vsqrt``)."""
    return _unary("sqrt", src, _semantic)


@builtin
def vrsqrt(src, _semantic=None) -> vector:
    """Elementwise reciprocal square root (``ave.hir.vrsqrt``)."""
    return _unary("rsqrt", src, _semantic)


@builtin
def vreduce(src, kind, mask=None, _semantic=None) -> vector:
    """
    Horizontal reduction (``ave.hir.reduction``) of a vector register.

    ``kind`` is ``add`` / ``min`` / ``max`` / ``umin`` / ``umax``; the
    combining result is placed at the first lane (AVE reduction semantics).
    """
    _require_vector(src, "src")
    kind = _unwrap_if_constexpr(kind)
    kind = str(kind).lower()
    if kind not in ("add", "min", "max", "umin", "umax"):
        raise ValueError(f"reduce kind must be add/min/max/umin/umax, got {kind!r}")
    lanes = src.type.lanes
    mask = _mask_for(mask, lanes, _semantic)
    handle = _semantic.builder.create_simd_vreduce(src.handle, mask.handle, kind)
    return vector(handle, src.type)


def _reduce_horiz(op, src, kind, mask, mode, _semantic):
    """Masked horizontal reduction shared by vcmax/vcmin."""
    mode = _resolve_mode(mode)
    if mode != "ZEROING":
        raise NotImplementedError(f"{op} only supports MODE.ZEROING; MERGING needs a destination "
                                  "register and is not exposed by the AVE dialect")
    return vreduce(src, kind, mask=mask, _semantic=_semantic)


@builtin
def vcmax(src: vector, mask: vector = None, mode: tl.constexpr = MODE.ZEROING, _semantic=None) -> vector:
    """
    Horizontal max reduction (``vcmax`` -> ``ave.hir.reduction``).

    The maximum is placed in lane 0 of the result register.  Unsigned integer
    inputs use the ``umax`` combining kind.  Only ``MODE.ZEROING`` is
    supported.
    """
    kind = "umax" if _is_unsigned(src.dtype) else "max"
    return _reduce_horiz("vcmax", src, kind, mask, mode, _semantic)


@builtin
def vcmin(src: vector, mask: vector = None, mode: tl.constexpr = MODE.ZEROING, _semantic=None) -> vector:
    """
    Horizontal min reduction (``vcmin`` -> ``ave.hir.reduction``).

    The minimum is placed in lane 0 of the result register.  Unsigned integer
    inputs use the ``umin`` combining kind.  Only ``MODE.ZEROING`` is
    supported.
    """
    kind = "umin" if _is_unsigned(src.dtype) else "min"
    return _reduce_horiz("vcmin", src, kind, mask, mode, _semantic)


_UNSIGNED_CMP = {
    "GT": "UGT",
    "GE": "UGE",
    "LT": "ULT",
    "LE": "ULE",
}


@builtin
def vcmp(src0: vector, src1: vector, mask: vector = None, op: tl.constexpr = CMP.EQ, _semantic=None) -> vector:
    """
    Elementwise compare (``ave.hir.vcmp``) returning an i1 mask.

    :param src0: first vector register
    :param src1: second vector register (same type as ``src0``)
    :param mask: predicate register; defaults to an all-true mask
    :param op: ``CMP`` value (EQ/NE/GT/GE/LT/LE plus the explicit unsigned
        UGT/UGE/ULT/ULE variants).  For unsigned integer operands the ordered
        comparisons are mapped to their unsigned counterparts automatically.
    """
    # Backwards compatibility: the old signature took the compare type as the
    # third positional argument, e.g. vcmp(a, b, "ne").
    if isinstance(_unwrap_if_constexpr(mask), (str, CMP)):
        op, mask = mask, None
    op = _resolve_cmp(op)
    _require_vector(src0, "src0")
    _require_vector(src1, "src1")
    if src0.type != src1.type:
        raise ValueError("vcmp requires same-typed operands, "
                         f"got {src0.type} and {src1.type}")
    if op in ("UGT", "UGE", "ULT", "ULE"):
        if not str(src0.dtype).startswith(("int", "uint")):
            raise TypeError(f"{op} requires integer operands, got {src0.dtype}")
    elif _is_unsigned(src0.dtype):
        op = _UNSIGNED_CMP.get(op, op)
    lanes = src0.type.lanes
    mask = _mask_for(mask, lanes, _semantic)
    handle = _semantic.builder.create_simd_vcmp(op, src0.handle, src1.handle, mask.handle)
    return vector(handle, vector_type(lanes, tl.int1))


@builtin
def vbroadcast(src, low=True, _semantic=None) -> vector:
    """
    Broadcast a vector register's half (``ave.hir.vector_broadcast``).

    ``low`` selects the low half to broadcast to the full register; used to
    splat a reduced value (e.g. the lane-0 sum) across all lanes.
    """
    _require_vector(src, "src")
    lanes = src.shape[0]
    mask = _all_mask(lanes, _semantic)
    handle = _semantic.builder.create_simd_vbroadcast(src.handle, mask.handle, bool(low))
    return vector(handle, src.type)


@builtin
def vfull(value, dtype, lanes, _semantic=None) -> vector:
    """
    Constant vector register: broadcast a scalar ``value`` to ``lanes`` lanes.

    Used for the -inf sentinel in topk, zero vectors, etc.
    """
    value = _unwrap_if_constexpr(value)
    lanes = _unwrap_if_constexpr(lanes)
    tag = _elem_tag(dtype)
    ty = vector_type(lanes, dtype)
    scalar = _semantic.builder.create_simd_scalar(tag, float(value))
    handle = _semantic.builder.create_simd_vsplat(scalar, tag, lanes)
    return vector(handle, ty)


@builtin
def vtruncf(src, target_dtype, mask=None, part="part_even", _semantic=None) -> vector:
    """
    Narrowing float conversion (``ave.hir.vtruncf``), e.g. f32 -> bf16.

    Keeps the total bit count; the number of lanes grows as the element type narrows.

    :param src: source vector register (f32)
    :param target_dtype: destination element dtype (bf16/f16)
    :param mask: predicate vector; defaults to an all-true mask
    :param part: ``part_even`` / ``part_odd`` sub-register selector
    """
    _require_vector(src, "src")
    part = ss.check_part(_unwrap_if_constexpr(part))
    out_tag = _elem_tag(target_dtype)
    if out_tag not in ("bf16", "f16", "fp8e4nv"):
        raise ValueError(f"vtruncf only narrows to bf16/f16/fp8e4nv, got {target_dtype}")
    src_tag = _elem_tag(src.dtype)
    src_bits = _ELEM_SPECS[src_tag]
    out_bits = _ELEM_SPECS[out_tag]
    out_lanes = src.shape[0] * src_bits // out_bits
    if mask is None:
        mask = _all_mask(src.shape[0], _semantic)
    _check_mask(mask, src.type.lanes)
    handle = _semantic.builder.create_simd_vcvt("truncf", src.handle, mask.handle, out_tag, part)
    return vector(handle, vector_type(out_lanes, target_dtype))


@builtin
def vsplat(value, dtype, _semantic=None) -> vector:
    """
    Constant vector register: broadcast a scalar ``value`` to a full register
    of ``dtype`` lanes (``ave.hir.scalar_broadcast``).

    Thin wrapper over :func:`vfull` that derives the lane count from ``dtype``
    (2048 // elem_bits; i64 spans 4096 bits).

    :param value: scalar value (Python int/float or constexpr)
    :param dtype: element dtype; lane count is derived from it
    """
    tag = _elem_tag(dtype)
    return vfull(value, dtype, _lanes_for(tag), _semantic=_semantic)


@builtin
def vreduce_sum(src, _semantic=None) -> vector:
    """
    Horizontal reduction to sum (``ave.hir.reduction <add>``).

    Alias of ``vreduce(src, 'add')``; the combining result is placed at lane 0.
    """
    return vreduce(src, "add", _semantic=_semantic)


def _reduce_across(regs, kind, _semantic):
    """
    Reduce a single register or a tuple/list of registers to the elementwise
    max/min, then horizontally reduce and broadcast lane 0 to the full
    register (``ave.hir.vmax``/``vmin`` + ``reduction`` + ``vector_broadcast``).

    Mirrors the TileKernels ``S.vdupv(S.vcmax(...))`` chain: the result is the
    global max/min (over all registers and lanes) replicated across every lane,
    so it can be fed directly into :func:`vcmp` with the original registers.
    """
    if isinstance(regs, (list, tuple)):
        regs = list(regs)
    elif isinstance(regs, tl.tuple):
        # triton visit_List materializes [a, b] as tl.tuple; unwrap it.
        regs = list(regs.values)
    else:
        regs = [regs]
    if not regs:
        raise ValueError(f"{kind}-across requires at least one register")
    for reg in regs:
        _require_vector(reg, "reduction operand")
        if reg.type != regs[0].type:
            raise ValueError(f"{kind}-across requires same-typed registers")
    acc = regs[0]
    for r in regs[1:]:
        lanes = acc.shape[0]
        mask = _all_mask(lanes, _semantic)
        kind_op = ("max" if kind == "max" else "min")
        handle = _semantic.builder.create_simd_binary(kind_op, acc.handle, r.handle, mask.handle)
        acc = vector(handle, acc.type)
    reduced = vreduce(acc, kind, _semantic=_semantic)
    return vbroadcast(reduced, low=True, _semantic=_semantic)


@builtin
def vreduce_max(regs, _semantic=None) -> vector:
    """
    Reduce a register or list of registers to the global max across all lanes
    and registers, broadcast to every lane.

    Combines elementwise ``ave.hir.vmax`` folds across registers, a horizontal
    ``ave.hir.reduction <max>``, and ``ave.hir.vector_broadcast`` (low half).
    """
    return _reduce_across(regs, "max", _semantic)


@builtin
def vreduce_min(regs, _semantic=None) -> vector:
    """
    Reduce a register or list of registers to the global min across all lanes
    and registers, broadcast to every lane.  Symmetric to
    :func:`vreduce_max` with ``ave.hir.vmin``.
    """
    return _reduce_across(regs, "min", _semantic)


@builtin
def vci(base, _semantic=None) -> vector:
    """
    Lane-index register: lane ``i`` holds ``base + i`` (``ave.hir.vci``).

    The AVE VCI lowering supports i8/i16/i32/f16/f32 element widths; this
    interface fixes the element type to i32 with 64 lanes (2048 bits).  The
    ``base`` is either a Python int / constexpr (lowered to an ``arith.constant
    i32``) or a scalar i32 tensor (e.g. ``tl.program_id(0) * 64``).

    :param base: starting index for lane 0
    """
    base = _unwrap_if_constexpr(base)
    if isinstance(base, tensor):
        if base.type.is_block() or base.dtype != tl.int32:
            raise TypeError("vci base must be a scalar int32")
        base_handle = base.handle
    else:
        if type(base) is not int or not -(2**31) <= base < 2**31:
            raise ValueError("vci base must be a compile-time int32 or scalar int32 tensor")
        base_handle = _semantic.builder.create_simd_scalar("i32", float(base))
    lanes = _lanes_for("i32")
    handle = _semantic.builder.create_simd_vci(base_handle, "i32", lanes, False)
    return vector(handle, vector_type(lanes, tl.int32))


@builtin
def pge(n, lanes, _semantic=None) -> vector:
    """
    Prefix predicate: the first ``n`` lanes are true, the rest false
    (``ave.hir.pge <VLn>``).

    Used for group-boundary selects (e.g. cast_back's 32-channel scale groups
    over a 64-lane register: ``pge(32, 64)``).

    :param n: number of leading true lanes; must match a supported VL pattern
        (1, 2, 3, 4, 8, 16, 32, 64, 128)
    :param lanes: number of i1 predicate lanes
    """
    n = _unwrap_if_constexpr(n)
    lanes = _unwrap_if_constexpr(lanes)
    ty = vector_type(lanes, tl.int1)
    pattern = ss.check_pge_pattern(f"VL{n}")
    handle = _semantic.builder.create_simd_pge(pattern, lanes)
    return vector(handle, ty)


@builtin
def membar(scope, _semantic=None):
    """
    Memory barrier inside a VF (``ave.hir.membar``).

    ``scope`` names the AVE membar kind, e.g. ``VST_VLD`` (vector-store ->
    vector-load, used before reading values written by masked stores) or
    ``VV_ALL``.  Accepted scopes: VV_ALL, VST_VLD, VLD_VST, VST_VST, VS_ALL,
    VST_LD, VLD_ST, VST_ST, SV_ALL, ST_VLD, LD_VST, ST_VST, SS_ALL, ST_LD,
    LD_ST, ST_ST.
    """
    scope = str(_unwrap_if_constexpr(scope)).upper()
    _semantic.builder.create_simd_vmembar(scope)
