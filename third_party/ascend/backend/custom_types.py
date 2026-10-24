# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Portable conversions for FP8 encodings without native Ascend storage."""

from triton.language import core
from triton._C.libtriton import ir


def _is_fnuz(dtype):
    return dtype.is_fp8e4b8() or dtype.is_fp8e5b16()


def _convert_fnuz(arg, dst_ty, rounding, semantic):
    """Use byte storage and exact integer RTNE/RTZ encoding for FNUZ.

    FNUZ has one zero and one NaN (0x80). Every other encoding is finite.
    RTNE overflow produces NaN; RTZ finite overflow selects the largest
    finite value. Infinite and NaN inputs produce the unique NaN in either
    mode. FP16/BF16 inputs widen exactly before the single rounding step.
    """
    src = arg.type.scalar
    dst = dst_ty.scalar
    if _is_fnuz(src):
        mantissa, bias = (3, 8) if src.is_fp8e4b8() else (2, 16)
        # Fraction and exponent are nonnegative and fit in signed i32;
        # the sign is only a bit mask. Signed field arithmetic avoids
        # unsupported unsigned casts on A3.
        bits = semantic.cast(semantic.bitcast(arg, core.uint8), core.int32)
        fraction = semantic.and_(bits, (1 << mantissa) - 1)
        exponent = semantic.lshr(semantic.and_(bits, 0x7f), mantissa)
        sign = semantic.shl(semantic.and_(bits, 0x80), 24)
        normal = semantic.or_(
            sign,
            semantic.or_(semantic.shl(semantic.add(exponent, 127 - bias, sanitize_overflow=False), 23),
                         semantic.shl(fraction, 23 - mantissa)))
        # Subnormal magnitudes are small exact powers-of-two multiples.
        small = semantic.mul(semantic.cast(fraction, core.float32), 2.0**(1 - bias - mantissa), sanitize_overflow=False)
        small_bits = semantic.or_(semantic.bitcast(small, core.int32), sign)
        decoded = semantic.where(semantic.equal(exponent, 0), small_bits, normal)
        decoded = semantic.where(semantic.equal(bits, 0x80), 0x7fc00000, decoded)
        return semantic.cast(semantic.bitcast(decoded, core.float32), dst_ty)

    assert _is_fnuz(dst)
    if not (src.is_floating() or src.is_int()):
        raise ValueError(f"Unsupported FNUZ source type {src}")
    if src.is_fp64():
        raise ValueError("Ascend software FNUZ conversion supports up to float32 inputs")
    arg = semantic.cast(arg, core.float32)
    mantissa, bias = (3, 8) if dst.is_fp8e4b8() else (2, 16)
    bits = semantic.bitcast(arg, core.int32)
    magnitude = semantic.and_(bits, 0x7fffffff)
    invalid_input = semantic.greater_equal(magnitude, 0x7f800000)
    # NaN payloads are irrelevant to FNUZ's unique NaN. Bound them before
    # rounding so even the unselected arithmetic stays within signed i32.
    magnitude = semantic.minimum(magnitude, 0x7f800000, core.PropagateNan.NONE)
    sign = semantic.and_(semantic.lshr(bits, 24), 0x80)
    exponent = semantic.lshr(magnitude, 23)
    shift = 23 - mantissa
    if rounding in (None, ir.ROUNDING_MODE.RTNE):
        correction = semantic.add((1 << (shift - 1)) - 1, semantic.and_(semantic.lshr(magnitude, shift), 1),
                                  sanitize_overflow=False)
        rounded = semantic.add(magnitude, correction, sanitize_overflow=False)
    elif rounding == ir.ROUNDING_MODE.RTZ:
        rounded = magnitude
    else:
        raise ValueError(f"Unsupported FNUZ rounding mode {rounding}")
    normal = semantic.sub(semantic.lshr(rounded, shift), (127 - bias) << mantissa, sanitize_overflow=False)
    significand = semantic.or_(semantic.and_(magnitude, 0x7fffff), 0x800000)
    sub_shift = semantic.sub(151 - bias - mantissa, exponent, sanitize_overflow=False)
    sub_shift = semantic.minimum(sub_shift, 24, core.PropagateNan.NONE)
    sub_shift = semantic.maximum(sub_shift, 1, core.PropagateNan.NONE)
    if rounding in (None, ir.ROUNDING_MODE.RTNE):
        halfway = semantic.shl(1, semantic.sub(sub_shift, 1, sanitize_overflow=False))
        correction = semantic.add(semantic.sub(halfway, 1, sanitize_overflow=False),
                                  semantic.and_(semantic.lshr(significand, sub_shift), 1), sanitize_overflow=False)
        significand = semantic.add(significand, correction, sanitize_overflow=False)
    subnormal = semantic.lshr(significand, sub_shift)
    subnormal = semantic.where(semantic.greater_equal(exponent, 127 - bias - mantissa), subnormal, 0)
    encoded = semantic.where(semantic.greater_equal(exponent, 128 - bias), normal, subnormal)
    if rounding == ir.ROUNDING_MODE.RTZ:
        encoded = semantic.minimum(encoded, 0x7f, core.PropagateNan.NONE)
    # Both signs of a value rounding to zero become the single positive zero.
    signed = semantic.where(semantic.equal(encoded, 0), 0, semantic.or_(encoded, sign))
    invalid = semantic.or_(invalid_input, semantic.greater_equal(encoded, 0x80))
    signed = semantic.where(invalid, 0x80, signed)
    return semantic.bitcast(semantic.cast(signed, core.uint8), dst_ty)


@core.builtin
def convert_custom_float8(arg, dst_ty, fp_downcast_rounding=None, _semantic=None):
    """Implement the legacy ``fp8e4b15`` conversion without target assembly.

    Match ``third_party/nvidia/language/cuda/utils.py``: decoding embeds the
    sign, four exponent bits and three fraction bits into FP16 (bias 15).
    Every byte has a finite value, including 0x7f/0xff, which decode to
    +/-1.875. Encoding instead saturates at +/-1.75, as the upstream software
    converter does, and preserves the input sign, including signed zero/NaN.

    The legacy encoder rounds halfway cases away from zero, regardless of
    ``fp_downcast_rounding``; FP32 inputs first truncate to FP16. This is not
    an IEEE round-to-nearest-even encoder, despite the generic cast API's
    default rounding argument. Keeping this behavior matches existing
    Triton kernels using this software format.
    """
    src_scalar = arg.type.scalar
    dst_scalar = dst_ty.scalar

    if _is_fnuz(src_scalar) or _is_fnuz(dst_scalar):
        return _convert_fnuz(arg, dst_ty, fp_downcast_rounding, _semantic)

    if src_scalar.is_fp8e4b15():
        bits = _semantic.cast(_semantic.bitcast(arg, core.uint8), core.uint16)
        magnitude = _semantic.shl(_semantic.and_(bits, 0x7f), 7)
        sign = _semantic.shl(_semantic.and_(bits, 0x80), 8)
        half = _semantic.bitcast(_semantic.or_(magnitude, sign), core.float16)
        # Keep the requested scalar element type and the original block shape.
        return _semantic.cast(half, dst_ty)

    if dst_scalar.is_fp8e4b15() and src_scalar.is_bf16():
        # Widen the representation without canonicalizing a signed NaN.
        bits = _semantic.cast(_semantic.bitcast(arg, core.int16), core.int32)
        arg = _semantic.bitcast(_semantic.shl(bits, 16), core.float32)
        src_scalar = arg.type.scalar

    if dst_scalar.is_fp8e4b15() and src_scalar.is_int():
        # Only 0 and +/-1 lie within the encoder's saturation range. Encode
        # them directly: converting a wider integer to float is unnecessary
        # and can be unsupported or lose its sign on A3.
        one = _semantic.equal(arg, 1)
        signed = src_scalar.is_int_signed() and not src_scalar.is_bool()
        if signed:
            one = _semantic.or_(one, _semantic.equal(arg, -1))
        encoded = _semantic.where(one, 0x78, 0x7e)
        encoded = _semantic.where(_semantic.equal(arg, 0), 0, encoded)
        if signed:
            encoded = _semantic.or_(encoded, _semantic.where(_semantic.less_than(arg, 0), 0x80, 0))
        return _semantic.bitcast(_semantic.cast(encoded, core.uint8), dst_ty)

    if not dst_scalar.is_fp8e4b15() or not (src_scalar.is_fp16() or src_scalar.is_fp32()):
        raise ValueError("Ascend fp8e4b15 encoding supports integer, float16, bfloat16 and float32 inputs; "
                         f"got {src_scalar} -> {dst_scalar}")

    if src_scalar.is_fp16():
        bits = _semantic.bitcast(arg, core.uint16)
        sign = _semantic.lshr(_semantic.and_(bits, 0x8000), 8)
        magnitude = _semantic.and_(bits, 0x7fff)
        # Nonnegative FP16 bit patterns are ordered as unsigned integers.
        # Infinity and every NaN payload saturate, like upstream min.f16x2
        # without the .NaN modifier.
        magnitude = _semantic.minimum(magnitude, 0x3f00, core.PropagateNan.NONE)
        magnitude = _semantic.lshr(_semantic.add(magnitude, 0x40, sanitize_overflow=False), 7)
    else:
        # Every FP8 rounding midpoint is exactly representable in FP16.
        # Direct rounding is therefore equivalent to the upstream FP32 ->
        # FP16 RTZ -> FP8 ties-away sequence. Integer operations also avoid
        # a tensor-only HFusion cast when the argument is a scalar.
        bits = _semantic.bitcast(arg, core.int32)
        sign = _semantic.and_(_semantic.lshr(bits, 24), 0x80)
        magnitude = _semantic.and_(bits, 0x7fffffff)
        magnitude = _semantic.minimum(magnitude, 0x3fe00000, core.PropagateNan.NONE)
        exponent = _semantic.lshr(magnitude, 23)

        rounded = _semantic.lshr(_semantic.add(magnitude, 0x80000, sanitize_overflow=False), 20)
        normal = _semantic.sub(rounded, (127 - 15) << 3, sanitize_overflow=False)

        # Subnormals are multiples of 2**-17. FP32 exponents 109..112
        # contain values that round into FP8 subnormals (or the first normal).
        significand = _semantic.or_(_semantic.and_(magnitude, 0x7fffff), 0x800000)
        shift = _semantic.sub(133, exponent, sanitize_overflow=False)
        # Keep every shift in range even on lanes that select another branch.
        shift = _semantic.minimum(shift, 24, core.PropagateNan.NONE)
        halfway = _semantic.shl(1, _semantic.sub(shift, 1, sanitize_overflow=False))
        subnormal = _semantic.lshr(_semantic.add(significand, halfway, sanitize_overflow=False), shift)
        subnormal = _semantic.where(_semantic.greater_equal(exponent, 109), subnormal, 0)
        magnitude = _semantic.where(_semantic.greater_equal(exponent, 113), normal, subnormal)

    encoded = _semantic.cast(_semantic.or_(magnitude, sign), core.uint8)
    return _semantic.bitcast(encoded, dst_ty)
