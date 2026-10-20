# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Stub for triton._C.libtriton.ir used in doc-build mock mode.

Provides proper enum classes for constants that appear in function signatures
and default parameter values, so that Sphinx autodoc renders clean names
instead of MagicMock repr strings.
"""
import enum


class PROPAGATE_NAN(enum.Enum):
    NONE = 0
    ALL = 1


class ROUNDING_MODE(enum.Enum):
    RTNE = 0
    RTZ = 1
    RTZ_DYNAMIC = 2


class CACHE_MODIFIER(enum.Enum):
    NONE = 0
    CG = 1
    CA = 2
    WB = 3
    WT = 4


class MEMORY_ORDER(enum.Enum):
    RELAXED = 0
    ACQUIRE = 1
    RELEASE = 2
    ACQ_REL = 3


class MEMORY_SCOPE(enum.Enum):
    GPU = 0
    CTA = 1
    SYSTEM = 2


class ATOMIC_OP(enum.Enum):
    ADD = 0
    FADD = 1
    MIN = 2
    MAX = 3
    AND = 4
    OR = 5
    XOR = 6
    XCHG = 7
    CAS = 8
    EXCH = 9
