"""Explicit DMA routes and configuration objects passed through to HIVM.

Construct params inline, pass them as a host constexpr argument, or assign a
local configuration with ``p: tl.constexpr = tla.LoadGM2UBParams(...)``. Scalar
operand fields may contain runtime scalar tensors; attribute fields are static.
Shape, dtype and hardware restrictions are checked by the backend verifier.
"""
from dataclasses import dataclass
from enum import Enum, IntEnum

from triton.language.core import builtin, float32, _unwrap_if_constexpr
from .core import buffer


class PadMode(Enum):
    PadNull = "PadNull"
    PadFirstElem = "PadFirstElem"
    PadValue = "PadValue"


class EvictionPolicy(Enum):
    EvictNormal = "EvictNormal"
    EvictFirst = "EvictFirst"
    EvictLast = "EvictLast"


class AtomicKind(Enum):
    NONE = "none"
    ADD = "add"
    MAX = "max"
    MIN = "min"
    AND = "and"
    OR = "or"
    XOR = "xor"
    UMAX = "umax"
    UMIN = "umin"


class UnitFlagMode(IntEnum):
    DISABLED = 0
    RESERVED = 1
    ENABLED_WITHOUT_UPDATE = 2
    ENABLED_WITH_UPDATE = 3


class FixpipeDMAMode(Enum):
    NZ2ND = "nz2nd"
    NZ2DN = "nz2dn"
    NZ2NZ = "normal"


class FixpipeDualDstMode(Enum):
    NO_DUAL = "NO_DUAL"
    ROW_SPLIT = "ROW_SPLIT"
    COLUMN_SPLIT = "COLUMN_SPLIT"


class FixpipeSubBlock(Enum):
    SUB_BLOCK_0 = "sub_block_0"
    SUB_BLOCK_1 = "sub_block_1"


class FixpipePreQuantMode(Enum):
    NO_QUANT = "NO_QUANT"
    F322F16 = "F322F16"
    S322I8 = "S322I8"
    QF322F32_PRE = "QF322F32_PRE"
    F322BF16 = "F322BF16"


class FixpipePreReluMode(Enum):
    NO_RELU = "NO_RELU"
    NORMAL_RELU = "NORMAL_RELU"
    LEAKY_RELU = "LEAKY_RELU"
    P_RELU = "P_RELU"


def _buffers(src, dst):
    if not isinstance(src, buffer) or not isinstance(dst, buffer):
        raise TypeError("TLA DMA expects two tla.buffer operands")


def _enum_attr(value, enum, mnemonic, semantic):
    value = _unwrap_if_constexpr(value)
    if value is None:
        return None
    if not isinstance(value, enum):
        raise TypeError(f"expected {enum.__name__}, got {value!r}")
    return semantic.builder.parse_attr(f"#hivm.{mnemonic}<{value.value}>")


def _integer(value, name, bits=64):
    value = _unwrap_if_constexpr(value)
    if type(value) is not int or not -(2**(bits - 1)) <= value < 2**(bits - 1):
        raise ValueError(f"{name} must be a compile-time i{bits} integer")
    return value


def _boolean(value, name):
    value = _unwrap_if_constexpr(value)
    if type(value) is not bool:
        raise TypeError(f"{name} must be a compile-time bool")
    return value


def _scalar(value, semantic, dtype=None):
    value = _unwrap_if_constexpr(value)
    if value is None:
        return None
    result = semantic.to_tensor(value)
    if result.shape:
        raise TypeError("DMA scalar operands must be scalar values")
    if dtype is not None:
        result = semantic.cast(result, dtype)
    elif not result.dtype.is_int():
        raise TypeError("DMA index operands must be integers")
    return result.handle


def _unit_flag_args(control, group_id):
    """Unwrap static inputs; native op builders encode the HIVM attributes."""
    control = _unwrap_if_constexpr(control)
    if control is not None:
        if isinstance(control, UnitFlagMode):
            control = control.value
        control = _integer(control, "unit_flag_ctrl", bits=32)
    group_id = _unwrap_if_constexpr(group_id)
    if group_id is not None:
        group_id = _integer(group_id, "unit_flag_group_id")
    return control, group_id


@dataclass(frozen=True)
class LoadGM2UBParams:
    pad_mode: PadMode = None
    pad_value: object = None
    left_padding_num: object = None
    right_padding_num: object = None
    eviction_policy: EvictionPolicy = None


@dataclass(frozen=True)
class StoreUB2GMParams:
    """Native StoreOp configuration; named for the UB -> GM direction."""
    atomic_kind: AtomicKind = None


@dataclass(frozen=True)
class LoadGM2L1Params:
    """GM ND -> L1 NZ. The two reverse-layout flags await specification review."""
    l2_cache_ctrl: int = None
    enable_nz2nd: bool = False
    enable_nz2dn: bool = False


@dataclass(frozen=True)
class CopyUB2L1Params:
    pad_mode: PadMode = None
    pad_value: object = None
    collapse_reassociation: tuple = None


@dataclass(frozen=True)
class CopyL12UBParams:
    """L12UBOp has no configuration operands or attributes."""


@dataclass(frozen=True)
class CopyL12BTParams:
    """n is the native index operand; None uses the last source extent."""
    n: object = None


@dataclass(frozen=True)
class CopyL12FBParams:
    """Reserved pending a backend L1 -> FixBUF operation."""


@dataclass(frozen=True)
class CopyL1ToL0Params:
    """Explicit native K-partition fields; actual_batch has no proven mapping."""
    is_transpose: bool = False
    actual_batch: int = None
    k_part_idx: int = None
    k_part: int = None
    k_part_ceil: int = None
    k_part_loop: int = None
    k_part_actual: int = None
    m: int = None


@dataclass(frozen=True)
class FixpipeL0C2GMParams:
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2NZ
    pre_quant: FixpipePreQuantMode = FixpipePreQuantMode.NO_QUANT
    pre_relu: FixpipePreReluMode = FixpipePreReluMode.NO_RELU
    quant_scale: object = None
    unit_flag_ctrl: UnitFlagMode = None
    unit_flag_group_id: int = None


@dataclass(frozen=True)
class FixpipeL0C2L1Params:
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2NZ
    pre_quant: FixpipePreQuantMode = FixpipePreQuantMode.NO_QUANT
    pre_relu: FixpipePreReluMode = FixpipePreReluMode.NO_RELU
    channel_split: bool = False
    c0_pad_en: bool = True
    unit_flag_ctrl: UnitFlagMode = None
    unit_flag_group_id: int = None
    quant_scale: object = None


@dataclass(frozen=True)
class FixpipeL0C2UBParams:
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2NZ
    dual_dst_mode: FixpipeDualDstMode = FixpipeDualDstMode.NO_DUAL
    sub_block_idx: FixpipeSubBlock = FixpipeSubBlock.SUB_BLOCK_0
    pre_quant: FixpipePreQuantMode = FixpipePreQuantMode.NO_QUANT
    pre_relu: FixpipePreReluMode = FixpipePreReluMode.NO_RELU
    quant_scale: object = None
    unit_flag_ctrl: UnitFlagMode = None
    unit_flag_group_id: int = None


def _params(value, cls):
    value = _unwrap_if_constexpr(value)
    if value is None:
        return cls()
    if type(value) is not cls:
        raise TypeError(f"expected {cls.__name__}, got {type(value).__name__}")
    return value


def _route(src, dst, spaces):
    _buffers(src, dst)
    if (src.space, dst.space) not in spaces:
        raise ValueError(f"DMA interface expects {spaces}, got {(src.space, dst.space)}")


@builtin
def load_gm_to_ub(src, dst, params=None, _semantic=None):
    """GM -> UB; params scalars become LoadOp operands, enums become attributes."""
    _route(src, dst, (("gm", "ub"), ))
    p = _params(params, LoadGM2UBParams)
    attrs = {
        "pad_mode": _enum_attr(p.pad_mode, PadMode, "padmode", _semantic),
        "eviction_policy": _enum_attr(p.eviction_policy, EvictionPolicy, "eviction_policy", _semantic),
    }
    _semantic.builder.create_tla_load(src.handle, dst.handle, attrs,
                                      pad_value=_scalar(p.pad_value, _semantic, src.dtype),
                                      left_padding_num=_scalar(p.left_padding_num, _semantic),
                                      right_padding_num=_scalar(p.right_padding_num, _semantic))


@builtin
def store_ub_to_gm(src, dst, params=None, _semantic=None):
    _route(src, dst, (("ub", "gm"), ))
    p = _params(params, StoreUB2GMParams)
    attrs = {"atomic_kind": _enum_attr(p.atomic_kind, AtomicKind, "atomic_kind", _semantic)}
    _semantic.builder.create_tla_store(src.handle, dst.handle, attrs)


@builtin
def load_gm_to_l1(src, dst, params=None, _semantic=None):
    """GM ND -> L1 NZ via ND2NZOp, including its raw i32 L2 cache field."""
    _route(src, dst, (("gm", "l1"), ))
    p = _params(params, LoadGM2L1Params)
    if _boolean(p.enable_nz2nd, "enable_nz2nd") or _boolean(p.enable_nz2dn, "enable_nz2dn"):
        raise NotImplementedError("GM->L1 ND2NZOp has no enable_nz2nd/enable_nz2dn fields; "
                                  "reverse-layout flags require specification review")
    cache = _unwrap_if_constexpr(p.l2_cache_ctrl)
    attrs = {}
    if cache is not None:
        attrs["l2_cache_mode"] = _semantic.builder.get_int32_attr(_integer(cache, "l2_cache_ctrl", bits=32))
    _semantic.builder.create_tla_nd2nz(src.handle, dst.handle, attrs)


@builtin
def copy_ub_to_l1(src, dst, params=None, _semantic=None):
    _route(src, dst, (("ub", "l1"), ))
    p = _params(params, CopyUB2L1Params)
    attrs = {"pad_mode": _enum_attr(p.pad_mode, PadMode, "padmode", _semantic)}
    groups = _unwrap_if_constexpr(p.collapse_reassociation)
    if groups is not None:
        indices = [[_integer(axis, "collapse_reassociation")
                    for axis in _unwrap_if_constexpr(group)]
                   for group in groups]
        attrs["collapse_reassociation"] = _semantic.builder.parse_attr(str(indices))
    _semantic.builder.create_tla_copy(src.handle, dst.handle, attrs,
                                      pad_value=_scalar(p.pad_value, _semantic, src.dtype))


@builtin
def copy_l1_to_ub(src, dst, params=None, _semantic=None):
    """L1 NZ -> UB ND; L12UBOp includes the layout conversion."""
    _route(src, dst, (("l1", "ub"), ))
    _params(params, CopyL12UBParams)
    _semantic.builder.create_tla_l12ub(src.handle, dst.handle)


@builtin
def copy_l1_to_bt(src, dst, params=None, _semantic=None):
    """Emit L12BTOp; standalone backend lowering remains a separate capability."""
    _route(src, dst, (("l1", "bt"), ))
    p = _params(params, CopyL12BTParams)
    n = src.shape[-1] if _unwrap_if_constexpr(p.n) is None else p.n
    _semantic.builder.create_tla_l12bt(src.handle, dst.handle, _scalar(n, _semantic))


@builtin
def copy_l1_to_fb(src, dst, params=None, _semantic=None):
    _route(src, dst, (("l1", "fb"), ))
    _params(params, CopyL12FBParams)
    raise NotImplementedError("PR #4403 has no L1->FixBUF DMA op; mapping requires backend review")


@builtin
def copy_l1_to_l0(src, dst, params=None, _semantic=None):
    _route(src, dst, (("l1", "l0a"), ("l1", "l0b")))
    p = _params(params, CopyL1ToL0Params)
    if _unwrap_if_constexpr(p.actual_batch) is not None:
        raise NotImplementedError("actual_batch has no field in L12L0Op; use the native k_part_* and m fields "
                                  "pending specification review")
    parts = {}
    for name in ("k_part_idx", "k_part", "k_part_ceil", "k_part_loop", "k_part_actual", "m"):
        value = _unwrap_if_constexpr(getattr(p, name))
        if value is None:
            raise ValueError(f"CopyL1ToL0Params requires native backend field {name}")
        parts[name] = _integer(value, name)
    _semantic.builder.create_tla_l12l0(src.handle, dst.handle, is_transposed=_boolean(p.is_transpose, "is_transpose"),
                                       **parts)


def _fixpipe(src, dst, p, semantic):
    attrs = {
        "dma_mode": _enum_attr(p.dma_mode, FixpipeDMAMode, "dma_mode", semantic),
        "pre_quant": _enum_attr(p.pre_quant, FixpipePreQuantMode, "fixpipe_pre_quant_mode", semantic),
        "pre_relu": _enum_attr(p.pre_relu, FixpipePreReluMode, "fixpipe_pre_relu_mode", semantic),
    }
    if isinstance(p, FixpipeL0C2UBParams):
        dual = _unwrap_if_constexpr(p.dual_dst_mode)
        # Native single-target lowering selects sub_block_idx only when the
        # optional dual-mode attribute is absent; NO_DUAL is its default.
        if dual != FixpipeDualDstMode.NO_DUAL:
            attrs["dual_dst_mode"] = _enum_attr(dual, FixpipeDualDstMode, "fixpipe_dual_dst_mode", semantic)
        attrs["sub_block_idx"] = _enum_attr(p.sub_block_idx, FixpipeSubBlock, "fixpipe_sub_block", semantic)
    if isinstance(p, FixpipeL0C2L1Params):
        attrs["channel_split"] = semantic.builder.get_bool_attr(_boolean(p.channel_split, "channel_split"))
        attrs["c0_pad_en"] = semantic.builder.get_bool_attr(_boolean(p.c0_pad_en, "c0_pad_en"))
    unit_flag_ctrl, unit_flag_group_id = _unit_flag_args(p.unit_flag_ctrl, p.unit_flag_group_id)
    semantic.builder.create_tla_fixpipe(src.handle, dst.handle, attrs,
                                        quant_scale=_scalar(p.quant_scale, semantic, float32),
                                        unit_flag_ctrl=unit_flag_ctrl, unit_flag_group_id=unit_flag_group_id)


@builtin
def store_l0c_to_gm(src, dst, params=None, _semantic=None):
    _route(src, dst, (("l0c", "gm"), ))
    _fixpipe(src, dst, _params(params, FixpipeL0C2GMParams), _semantic)


@builtin
def copy_l0c_to_ub(src, dst, params=None, _semantic=None):
    _route(src, dst, (("l0c", "ub"), ))
    _fixpipe(src, dst, _params(params, FixpipeL0C2UBParams), _semantic)


@builtin
def copy_l0c_to_l1(src, dst, params=None, _semantic=None):
    _route(src, dst, (("l0c", "l1"), ))
    _fixpipe(src, dst, _params(params, FixpipeL0C2L1Params), _semantic)


@builtin
def set_ctrl(enable, idx, _semantic=None):
    """Native SetCtrlOp; argument order remains (enable, idx)."""
    _semantic.builder.create_tla_set_ctrl(_boolean(enable, "enable"), _integer(idx, "idx"))
