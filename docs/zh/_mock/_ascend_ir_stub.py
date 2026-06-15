# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Stub for triton._C.libtriton.ascend.ir used in doc-build mock mode.

The pybind11 extension exposes C++ enums as Python enums where enum members
in ``__dict__`` are instances of the enum class itself.  Python's
``enum.Enum`` replicates this behaviour, so the downstream code that uses
``isinstance(v, ascend_ir.AddressSpace)`` works correctly.
"""
import enum
from unittest.mock import MagicMock


class CoreType(enum.Enum):
    VECTOR = 0
    CUBE = 1
    CUBE_OR_VECTOR = 2
    CUBE_AND_VECTOR = 3


class PIPE(enum.Enum):
    PIPE_S = 0
    PIPE_V = 1
    PIPE_M = 2
    PIPE_MTE1 = 3
    PIPE_MTE2 = 4
    PIPE_MTE3 = 5
    PIPE_ALL = 6
    PIPE_FIX = 7


class MODE(enum.Enum):
    SIMD = 0
    SIMT = 1
    MIX = 2


class IteratorType(enum.Enum):
    Parallel = 0
    Broadcast = 1
    Transpose = 2
    Reduction = 3
    Interleave = 4
    Deinterleave = 5
    Inverse = 6
    Pad = 7
    Concat = 8
    Gather = 9
    Cumulative = 10
    Opaque = 11


class AddressSpace(enum.Enum):
    UB = 0
    L1 = 1
    L0A = 2
    L0B = 3
    L0C = 4
    OUT = 5
    GM = 6
    WORKSPACE = 7


class FixpipeDMAMode(enum.Enum):
    NZ2DN = 0
    NZ2ND = 1
    NZ2NZ = 2


class FixpipeDualDstMode(enum.Enum):
    NO_DUAL = 0
    COLUMN_SPLIT = 1
    ROW_SPLIT = 2


class FixpipePreQuantMode(enum.Enum):
    NO_QUANT = 0
    F322BF16 = 1
    F322F16 = 2
    S322I8 = 3


class FixpipePreReluMode(enum.Enum):
    LEAKY_RELU = 0
    NO_RELU = 1
    NORMAL_RELU = 2
    P_RELU = 3


# MLIR affine types are only used as docstring-level references in the RST;
# keep them as MagicMock so attribute access never raises.
affine_expr = MagicMock(name="affine_expr")
affine_constant_expr = MagicMock(name="affine_constant_expr")
affine_dim_expr = MagicMock(name="affine_dim_expr")
affine_symbol_expr = MagicMock(name="affine_symbol_expr")
affine_binary_op_expr = MagicMock(name="affine_binary_op_expr")
affine_map = MagicMock(name="affine_map")

# Builder used in ascend backend patches – only called at runtime, not import
ascendnpu_ir_builder = MagicMock(name="ascendnpu_ir_builder")
