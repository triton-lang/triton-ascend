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
    """Configuration for :func:`load_gm_to_ub`.

    :param pad_mode: Compile-time ``PadNull``, ``PadFirstElem`` or ``PadValue``
        from ``tla.PadMode``. ``None`` leaves the attribute unset.
    :type pad_mode: tla.PadMode or None
    :param pad_value: Scalar fill value, converted to the source element type.
        A runtime tensor must have scalar shape. ``None`` omits the operand.
    :type pad_value: int, float, scalar tl.tensor or None
    :param left_padding_num: Number of left padding elements, passed as an
        index operand. ``None`` omits the operand.
    :type left_padding_num: int, scalar integer tl.tensor or None
    :param right_padding_num: Number of right padding elements, passed as an
        index operand. ``None`` omits the operand.
    :type right_padding_num: int, scalar integer tl.tensor or None
    :param eviction_policy: Compile-time L2 eviction policy: ``EvictNormal``,
        ``EvictFirst`` or ``EvictLast``. ``None`` leaves the attribute unset.
    :type eviction_policy: tla.EvictionPolicy or None

    All fields default to ``None``. Padding values/counts may be runtime
    scalars; enum fields must be compile-time values. The backend checks
    the padding-mode/operand combination and alignment.
    """
    pad_mode: PadMode = None
    pad_value: object = None
    left_padding_num: object = None
    right_padding_num: object = None
    eviction_policy: EvictionPolicy = None


@dataclass(frozen=True)
class StoreUB2GMParams:
    """Configuration for :func:`store_ub_to_gm`.

    :param atomic_kind: Compile-time atomic operation. Defaults to ``None``,
        which omits the attribute. ``ADD``, ``MAX`` and ``MIN`` are the native
        StoreOp atomic operations documented by the backend. Other enum
        members do not imply support for this route, dtype or target.
    :type atomic_kind: tla.AtomicKind or None
    """
    atomic_kind: AtomicKind = None


@dataclass(frozen=True)
class LoadGM2L1Params:
    """Configuration for :func:`load_gm_to_l1`.

    :param l2_cache_ctrl: Raw compile-time MTE2 cache-control field, encoded as
        a signed 32-bit ``l2_cache_mode`` attribute. ``None`` (default) omits
        it, selecting the backend default. This is not an eviction-policy enum.
    :type l2_cache_ctrl: int or None
    :param enable_nz2nd: Reserved reverse-layout flag. Must be ``False``
        (default); ``True`` raises ``NotImplementedError``.
    :type enable_nz2nd: bool
    :param enable_nz2dn: Reserved reverse-layout flag. Must be ``False``
        (default); ``True`` raises ``NotImplementedError``.
    :type enable_nz2dn: bool
    """
    l2_cache_ctrl: int = None
    enable_nz2nd: bool = False
    enable_nz2dn: bool = False


@dataclass(frozen=True)
class CopyUB2L1Params:
    """Configuration for :func:`copy_ub_to_l1`.

    :param pad_mode: Compile-time ``tla.PadMode`` member. ``None`` omits the
        padding-mode attribute.
    :type pad_mode: tla.PadMode or None
    :param pad_value: Scalar padding value converted to the source element
        type. ``None`` omits the operand.
    :type pad_value: int, float, scalar tl.tensor or None
    :param collapse_reassociation: Compile-time groups of source dimension
        indices to collapse, for example ``((0, 1), (2, 3))`` for a rank-four
        source. Each group contains integer axis indices; the backend checks
        validity. ``None`` omits the attribute.
    :type collapse_reassociation: tuple[tuple[int, ...], ...] or None

    All fields default to ``None``.
    """
    pad_mode: PadMode = None
    pad_value: object = None
    collapse_reassociation: tuple = None


@dataclass(frozen=True)
class CopyL12UBParams:
    """Configuration for :func:`copy_l1_to_ub`.

    This immutable configuration has no fields. Pass ``tla.CopyL12UBParams()``
    or omit the operation's ``params`` argument.
    """


@dataclass(frozen=True)
class CopyL12BTParams:
    """Configuration for :func:`copy_l1_to_bt`.

    :param n: Native L12BT ``n`` index operand. ``None`` (default) uses
        ``src.shape[-1]``. A runtime tensor must be a scalar integer;
        backend layout and extent restrictions still apply.
    :type n: int, scalar integer tl.tensor or None
    """
    n: object = None


@dataclass(frozen=True)
class CopyL12FBParams:
    """Reserved configuration for :func:`copy_l1_to_fb`.

    This immutable configuration has no fields. Constructing it is supported,
    but the copy operation itself always raises ``NotImplementedError``.
    """


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
    """Configuration for :func:`store_l0c_to_gm`.

    :param dma_mode: Compile-time layout conversion. ``NZ2NZ`` (default)
        preserves NZ, ``NZ2ND`` produces ND, and ``NZ2DN`` produces transposed ND.
    :type dma_mode: tla.FixpipeDMAMode
    :param pre_quant: Compile-time conversion mode. Defaults to ``NO_QUANT``.
        Members include ``F322F16``, ``F322BF16``, ``S322I8`` and
        ``QF322F32_PRE``. The backend validates the dtype/mode combination.
    :type pre_quant: tla.FixpipePreQuantMode
    :param pre_relu: Compile-time activation mode. Defaults to ``NO_RELU``.
        Other enum members are ``NORMAL_RELU``, ``LEAKY_RELU`` and ``P_RELU``;
        support depends on the backend and target.
    :type pre_relu: tla.FixpipePreReluMode
    :param quant_scale: Optional scalar scale, converted to ``tl.float32``.
        Required combinations with ``pre_quant`` are backend-dependent.
        ``None`` (default) omits the operand.
    :type quant_scale: int, float, scalar tl.tensor or None
    :param unit_flag_ctrl: Compile-time UnitFlag mode. ``None`` (default)
        leaves it unset. ``DISABLED`` (0), ``ENABLED_WITHOUT_UPDATE`` (2) and
        ``ENABLED_WITH_UPDATE`` (3) encode explicit modes. Value 1 is reserved.
    :type unit_flag_ctrl: tla.UnitFlagMode, int or None
    :param unit_flag_group_id: Compile-time signed 64-bit group ID, or ``None``
        (default) to omit it. Match the corresponding MMAD protocol;
        valid hardware IDs are backend-dependent.
    :type unit_flag_group_id: int or None

    .. note::
        ``None`` is different from explicitly disabling UnitFlag. To preserve
        manual modes, launch with ``unit_flag=False`` to disable automatic
        UnitFlag inference. Enum membership alone does not guarantee support
        for every dtype or target.
    """
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2NZ
    pre_quant: FixpipePreQuantMode = FixpipePreQuantMode.NO_QUANT
    pre_relu: FixpipePreReluMode = FixpipePreReluMode.NO_RELU
    quant_scale: object = None
    unit_flag_ctrl: UnitFlagMode = None
    unit_flag_group_id: int = None


@dataclass(frozen=True)
class FixpipeL0C2L1Params:
    """Configuration for :func:`copy_l0c_to_l1`.

    :param dma_mode: Compile-time layout mode. Defaults to ``NZ2NZ``.
        The backend checks which modes are supported for an L1 destination.
    :type dma_mode: tla.FixpipeDMAMode
    :param pre_quant: Compile-time conversion mode. Defaults to ``NO_QUANT``.
        The destination dtype must match the selected conversion.
    :type pre_quant: tla.FixpipePreQuantMode
    :param pre_relu: Compile-time activation mode. Defaults to ``NO_RELU``.
    :type pre_relu: tla.FixpipePreReluMode
    :param channel_split: Compile-time native channel-split flag for the L1
        destination. Defaults to ``False``.
    :type channel_split: bool
    :param c0_pad_en: Compile-time native C0-padding flag. Defaults to ``True``.
    :type c0_pad_en: bool
    :param unit_flag_ctrl: Compile-time UnitFlag mode. ``None`` (default)
        omits it. Modes 0, 2 and 3 mean disabled, enabled without update and
        enabled with update, respectively. Value 1 is reserved.
    :type unit_flag_ctrl: tla.UnitFlagMode, int or None
    :param unit_flag_group_id: Compile-time signed 64-bit group ID, or ``None``
        (default) to omit it. Match the corresponding MMAD protocol.
    :type unit_flag_group_id: int or None
    :param quant_scale: Optional scalar scale, converted to ``tl.float32``.
        ``None`` (default) omits the operand.
    :type quant_scale: int, float, scalar tl.tensor or None

    See :class:`FixpipeL0C2GMParams` for enum meanings and the manual UnitFlag
    launch option. Supported mode/dtype combinations and group IDs depend
    on the backend's L1-destination implementation.
    """
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
    """Configuration for :func:`copy_l0c_to_ub`.

    :param dma_mode: Compile-time layout mode. Defaults to ``NZ2NZ``.
        ``NZ2ND`` produces ND; ``NZ2DN`` produces transposed ND.
    :type dma_mode: tla.FixpipeDMAMode
    :param dual_dst_mode: Compile-time UB destination selection: ``NO_DUAL``
        (default), ``ROW_SPLIT`` or ``COLUMN_SPLIT`` across two AIV sub-blocks.
    :type dual_dst_mode: tla.FixpipeDualDstMode
    :param sub_block_idx: AIV sub-block selected in single-destination mode.
        Defaults to ``SUB_BLOCK_0``; ``SUB_BLOCK_1`` selects the other sub-block.
        For a dual split, pass ``None`` to omit this attribute and let the split
        mode determine the two destinations.
    :type sub_block_idx: tla.FixpipeSubBlock or None
    :param pre_quant: Compile-time conversion mode. Defaults to ``NO_QUANT``.
        The destination dtype must match the selected conversion.
    :type pre_quant: tla.FixpipePreQuantMode
    :param pre_relu: Compile-time activation mode. Defaults to ``NO_RELU``.
    :type pre_relu: tla.FixpipePreReluMode
    :param quant_scale: Optional scalar scale, converted to ``tl.float32``.
        ``None`` (default) omits the operand.
    :type quant_scale: int, float, scalar tl.tensor or None
    :param unit_flag_ctrl: Compile-time UnitFlag mode. ``None`` (default)
        omits it. Modes 0, 2 and 3 mean disabled, enabled without update and
        enabled with update, respectively. Value 1 is reserved.
    :type unit_flag_ctrl: tla.UnitFlagMode, int or None
    :param unit_flag_group_id: Compile-time signed 64-bit group ID, or ``None``
        (default) to omit it. Match the corresponding MMAD protocol.
    :type unit_flag_group_id: int or None

    See :class:`FixpipeL0C2GMParams` for enum meanings and the manual UnitFlag
    launch option. Supported mode/dtype combinations and group IDs depend
    on the backend's UB-destination implementation.
    """
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
    """Loads global memory into UB using ``hivm.hir.load``.

    :param src: GM source with ``space="gm"``, typically created with ``tla.view``.
    :type src: tla.buffer
    :param dst: UB destination with ``space="ub"`` and the same element type
        as ``src``. Its shape must accommodate the requested data and padding.
    :type dst: tla.buffer
    :param params: Configuration object. ``None`` (default) constructs
        ``tla.LoadGM2UBParams()``. See :class:`LoadGM2UBParams` for field formats.
    :type params: tla.LoadGM2UBParams or None
    :return: None. The destination buffer is updated in place.
    :rtype: None

    .. note::
        Pass buffer views, not raw pointers or buffer element accesses.
        Padding counts are scalar operands; the backend checks supported
        padding, alignment and dtype combinations. This interface does not
        take the per-element mask of ``tl.load``.

    Example inside a ``@tla.jit`` kernel::

        tile = tla.alloc((16, 32), tl.float16)
        tla.load_gm_to_ub(tla.view(X, (16, 32)), tile)
    """
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
    """Stores UB data to global memory using ``hivm.hir.store``.

    :param src: UB source with ``space="ub"``.
    :type src: tla.buffer
    :param dst: GM destination with ``space="gm"``, typically created with
        ``tla.view``. Its element type must match ``src``.
    :type dst: tla.buffer
    :param params: Configuration object. ``None`` (default) constructs
        ``tla.StoreUB2GMParams()``. See :class:`StoreUB2GMParams` for atomic modes.
    :type params: tla.StoreUB2GMParams or None
    :return: None. The destination buffer is updated in place.
    :rtype: None

    .. note::
        Source and destination views must describe a backend-compatible tile.
        Atomic operation and dtype support depend on the target backend;
        ``atomic_kind=None`` leaves the atomic attribute unset.
    """
    _route(src, dst, (("ub", "gm"), ))
    p = _params(params, StoreUB2GMParams)
    attrs = {"atomic_kind": _enum_attr(p.atomic_kind, AtomicKind, "atomic_kind", _semantic)}
    _semantic.builder.create_tla_store(src.handle, dst.handle, attrs)


@builtin
def load_gm_to_l1(src, dst, params=None, _semantic=None):
    """Loads GM data into L1 with ND-to-NZ conversion using ``hivm.hir.nd2nz``.

    :param src: GM source with ``space="gm"`` in ND layout.
    :type src: tla.buffer
    :param dst: L1 destination with ``space="l1"`` in the corresponding NZ
        layout. For an aligned fp16 ``(M, N)`` tile, a physical NZ shape is
        ``(N // 16, M // 16, 16, 16)``.
    :type dst: tla.buffer
    :param params: Configuration object. ``None`` (default) constructs
        ``tla.LoadGM2L1Params()``. See :class:`LoadGM2L1Params` for cache control.
    :type params: tla.LoadGM2L1Params or None
    :return: None. The destination buffer is updated in place.
    :rtype: None

    .. note::
        Emit this Cube transfer in ``tla.scope.cube()``. The two views must
        describe the same logical tile in their respective layouts. The
        backend checks dtype and layout alignment. Reverse-layout flags
        ``enable_nz2nd`` and ``enable_nz2dn`` must remain ``False``.
    """
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
    """Copies UB data into L1 using ``hivm.hir.copy``.

    :param src: UB source with ``space="ub"``.
    :type src: tla.buffer
    :param dst: L1 destination with ``space="l1"`` and a backend-compatible
        shape and element type.
    :type dst: tla.buffer
    :param params: Configuration object. ``None`` (default) constructs
        ``tla.CopyUB2L1Params()``. See :class:`CopyUB2L1Params` for padding
        and dimension-reassociation formats.
    :type params: tla.CopyUB2L1Params or None
    :return: None. The destination buffer is updated in place.
    :rtype: None

    .. note::
        This route does not implicitly perform the GM-to-L1 ND2NZ conversion.
        Prepare compatible physical layouts. When a Cube scope consumes data
        produced by a Vector scope, explicitly synchronize the handoff.
    """
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
    """Copies L1 NZ data into UB ND layout using ``hivm.hir.l12ub``.

    :param src: L1 source with ``space="l1"`` in NZ layout.
    :type src: tla.buffer
    :param dst: UB destination with ``space="ub"`` in the corresponding ND layout.
    :type dst: tla.buffer
    :param params: Empty configuration object. ``None`` (default) is equivalent
        to ``tla.CopyL12UBParams()``; this native operation has no options.
    :type params: tla.CopyL12UBParams or None
    :return: None. The destination buffer is updated in place.
    :rtype: None

    .. note::
        Emit this Cube transfer in ``tla.scope.cube()``. Before a Vector scope
        consumes the destination, synchronize the handoff and select the AIV
        sub-block that owns the written UB. The backend checks layout and dtype.
    """
    _route(src, dst, (("l1", "ub"), ))
    _params(params, CopyL12UBParams)
    _semantic.builder.create_tla_l12ub(src.handle, dst.handle)


@builtin
def copy_l1_to_bt(src, dst, params=None, _semantic=None):
    """Copies L1 data into the bias buffer using ``hivm.hir.l12bt``.

    :param src: L1 source with ``space="l1"``.
    :type src: tla.buffer
    :param dst: Bias-buffer destination with ``space="bt"``.
    :type dst: tla.buffer
    :param params: Configuration object. ``None`` (default) constructs
        ``tla.CopyL12BTParams()``. Its ``n`` operand defaults to ``src.shape[-1]``.
        See :class:`CopyL12BTParams` for the scalar format.
    :type params: tla.CopyL12BTParams or None
    :return: None. The emitted operation writes to ``dst``.
    :rtype: None

    .. note::
        This frontend emits the native operation; standalone compilation
        requires L12BT lowering support in the selected backend. A recognized
        ``"bt"`` address space does not imply support for allocating it with
        ``tla.alloc``. Backend layout and dtype restrictions still apply.
    """
    _route(src, dst, (("l1", "bt"), ))
    p = _params(params, CopyL12BTParams)
    n = src.shape[-1] if _unwrap_if_constexpr(p.n) is None else p.n
    _semantic.builder.create_tla_l12bt(src.handle, dst.handle, _scalar(n, _semantic))


@builtin
def copy_l1_to_fb(src, dst, params=None, _semantic=None):
    """Describes the reserved L1-to-FixBUF transfer interface.

    :param src: L1 source with ``space="l1"``.
    :type src: tla.buffer
    :param dst: FixBUF destination with ``space="fb"``.
    :type dst: tla.buffer
    :param params: Empty configuration object. ``None`` (default) is equivalent
        to ``tla.CopyL12FBParams()``.
    :type params: tla.CopyL12FBParams or None
    :raises NotImplementedError: After validating the inputs, because no
        backend operation has been mapped to this route.

    .. note::
        This interface does not currently emit a copy or return a result.
        Constructing ``CopyL12FBParams`` does not make the route available.
    """
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
    """Stores the L0C accumulator to GM using ``hivm.hir.fixpipe``.

    :param src: L0C source with ``space="l0c"``, typically the fractal
        accumulator produced by ``tla.cube.mmad``.
    :type src: tla.buffer
    :param dst: GM destination with ``space="gm"``. Its shape and element
        type must match the selected layout and quantization mode.
    :type dst: tla.buffer
    :param params: Configuration object. ``None`` (default) constructs
        ``tla.FixpipeL0C2GMParams()``. See :class:`FixpipeL0C2GMParams` for
        conversion, activation, scale and UnitFlag fields.
    :type params: tla.FixpipeL0C2GMParams or None
    :return: None. The destination buffer is updated in place.
    :rtype: None

    .. note::
        Emit in ``tla.scope.cube()``. Select ``NZ2ND`` for a logical row-major
        GM output, or ``NZ2DN`` for transposed ND. The default ``NZ2NZ`` keeps
        the fractal layout. Dtype conversion is controlled by ``pre_quant``;
        the backend checks the destination layout, dtype and alignment.
    """
    _route(src, dst, (("l0c", "gm"), ))
    _fixpipe(src, dst, _params(params, FixpipeL0C2GMParams), _semantic)


@builtin
def copy_l0c_to_ub(src, dst, params=None, _semantic=None):
    """Copies the L0C accumulator to UB using ``hivm.hir.fixpipe``.

    :param src: L0C source with ``space="l0c"`` in fractal layout.
    :type src: tla.buffer
    :param dst: UB destination with ``space="ub"``. Its shape and element
        type must match the selected layout and quantization mode.
    :type dst: tla.buffer
    :param params: Configuration object. ``None`` (default) constructs
        ``tla.FixpipeL0C2UBParams()``. See :class:`FixpipeL0C2UBParams` for
        conversion and AIV destination-selection fields.
    :type params: tla.FixpipeL0C2UBParams or None
    :return: None. The destination buffer is updated in place.
    :rtype: None

    .. note::
        Emit in ``tla.scope.cube()`` and synchronize before a Vector consumer.
        With ``NO_DUAL``, only the selected AIV sub-block owns the output;
        with a split mode, each AIV must use its own output portion. Writing
        back unconditionally from both sub-blocks can overwrite valid data.
        Mode, dtype and alignment restrictions are checked by the backend.
    """
    _route(src, dst, (("l0c", "ub"), ))
    _fixpipe(src, dst, _params(params, FixpipeL0C2UBParams), _semantic)


@builtin
def copy_l0c_to_l1(src, dst, params=None, _semantic=None):
    """Copies the L0C accumulator to L1 using ``hivm.hir.fixpipe``.

    :param src: L0C source with ``space="l0c"`` in fractal layout.
    :type src: tla.buffer
    :param dst: L1 destination with ``space="l1"``. Its shape and element
        type must match the selected layout and quantization mode.
    :type dst: tla.buffer
    :param params: Configuration object. ``None`` (default) constructs
        ``tla.FixpipeL0C2L1Params()``. See :class:`FixpipeL0C2L1Params` for
        conversion, channel-split and C0-padding fields.
    :type params: tla.FixpipeL0C2L1Params or None
    :return: None. The destination buffer is updated in place.
    :rtype: None

    .. note::
        Emit in ``tla.scope.cube()``. The backend checks mode, dtype and
        alignment. If the result is subsequently copied to UB, synchronize
        with the Vector consumer and respect UB sub-block ownership.
    """
    _route(src, dst, (("l0c", "l1"), ))
    _fixpipe(src, dst, _params(params, FixpipeL0C2L1Params), _semantic)


@builtin
def set_ctrl(enable, idx, _semantic=None):
    """Sets or clears a hardware CTRL bit using ``hivm.hir.set_ctrl``.

    :param enable: Compile-time ``True`` to set the bit or ``False`` to clear it.
    :type enable: bool
    :param idx: Compile-time bit index, representable as a signed 64-bit integer.
        The bit must be supported by the target hardware and backend.
    :type idx: int
    :return: None.
    :rtype: None

    .. note::
        The argument order is ``(enable, idx)``. Pass a numeric bit index,
        not a named mode enum. The previous state is not automatically restored.
        For example, ``tla.set_ctrl(False, 48)`` emits a clear of CTRL bit 48.
    """
    _semantic.builder.create_tla_set_ctrl(_boolean(enable, "enable"), _integer(idx, "idx"))
