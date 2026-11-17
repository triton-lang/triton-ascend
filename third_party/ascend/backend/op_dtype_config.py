# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Machine-readable operator dtype configuration for the Ascend backend.
#
# Source of truth: docs/zh/python-api/_ascend_constraints.py. Each documented
# operator carries an ASCII ``dtype_support`` table (one column per scalar
# dtype, one row per Ascend platform); the rules below transcribe that table
# for every operator whose support differs across platforms. Fully supported
# operators have no rule.
#
# ---------------------------------------------------------------------------
# Rule schema
# ---------------------------------------------------------------------------
#   "<rule id>": {
#       # Fully qualified names matched by the *entry layer* (direct
#       # ``tl.<name>(...)`` / @jit composite calls). Only direct
#       # ``triton.language`` operators are guarded.
#       "names": ["triton.language.permute"],
#
#       # Tensor dunder builtins matched by the *builtin-dunder layer*
#       # (``a + b``, ``a >> b``, ``a & b`` ...); reflected dunders included.
#       "dunder": ["__add__", "__radd__"],
#       "dunder_label": "+",
#
#       # Positions (0-based) of tensor-like operands to inspect. When
#       # omitted, every positional operand is inspected (non-tensors and
#       # pointers are skipped automatically).
#       "tensor_args": [0, 1],
#
#       # Positions that are pointers: the pointee element dtype is read.
#       "pointer_args": [0],
#
#       # The dtype restriction per architecture bucket:
#       #   "a2a3" -> Ascend A2/A3 ; "a5" -> Ascend 950
#       #   * "unsupported": blacklist -- the interception decision uses this.
#       #     For every blacklist rule it is the UNIFORM baseline:
#       #     a2 -> [fp64, fp8e4m3, fp8e5m2] ; a5 -> [fp64].
#       #   * "supported" (without "unsupported"): whitelist -- used by the
#       #     operators that carry a frontend @_check_dtype whitelist in
#       #     math.py (exp/exp2/log/log2/cos/sin/sqrt/sqrt_rn/rsqrt/div_rn/
#       #     erf/floor/ceil/umulhi). Every dtype NOT listed is rejected.
#       # Every blacklist rule also spells out an explicit "supported" list:
#       # the transcribed per-operator support table from
#       # docs/zh/python-api/_ascend_constraints.py (complement of that
#       # table's x marks over ALL_SCALAR_DTYPES, in universe order). It
#       # drives the error message; interception itself always uses the
#       # uniform "unsupported" blacklist. The "dot" rule is the only
#       # exception that keeps its own non-uniform blacklist (parameter
#       # level int8 constraint).
#       "unsupported": {"a2a3": [...], "a5": [...]},   # blacklist mode
#       "supported":   {"a2a3": [...], "a5": [...]},   # whitelist mode
#   }
#
# Dtype tokens (matched against ``dtype.name``):
#   * regular names: int8/int16/int32/int64/uint8/uint16/uint32/uint64/
#     fp16/bf16/fp32/fp64/bool
#   * "fp8e4m3" : the E4M3 column, matching fp8e4nv/fp8e4b8/fp8e4b15 at runtime
#   * "fp8e5m2" : the E5M2 column, matching fp8e5/fp8e5b16 at runtime
#   (``bool`` is represented internally as ``int1``; the matching layer
#   normalizes both. The concrete fp8 variant names live in FP8*_DTYPES.)

FP8E4_DTYPES = ["fp8e4nv", "fp8e4b8", "fp8e4b15"]
FP8E5_DTYPES = ["fp8e5", "fp8e5b16"]

# The 15 scalar-dtype columns of the documented ``dtype_support`` tables, in
# table order. Each rule's "unsupported" list is a subset of this universe; the
# accepted dtypes for a rule are the complement of that list.
ALL_SCALAR_DTYPES = [
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "fp16",
    "fp32",
    "fp64",
    "bf16",
    "fp8e4m3",
    "fp8e5m2",
    "bool",
]

OP_DTYPE_RULES = {
    "abs": {
        "names": ["triton.language.abs"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "add": {
        "names": ["triton.language.add"],
        "dunder": ["__add__", "__radd__"],
        "dunder_label": '+',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "advance": {
        "names": ["triton.language.advance"],
        "pointer_args": [0],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2"
            ],
        },
    },
    "and": {
        "dunder": ["__and__", "__rand__"],
        "dunder_label": '&',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
        },
    },
    "arange": {
        "names": ["triton.language.arange"],
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32"],
            "a5": ["uint8", "int8", "uint16", "int16", "int32"],
        },
    },
    "argmax": {
        "names": ["triton.language.argmax"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "argmin": {
        "names": ["triton.language.argmin"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "associative_scan": {
        "names": ["triton.language.associative_scan"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bool"],
        },
    },
    "assume": {
        "names": ["triton.language.assume"],
        "supported": {
            "a2a3": ["bool"],
            "a5": ["bool"],
        },
    },
    "atomic_add": {
        "names": ["triton.language.atomic_add"],
        "tensor_args": [0, 1],
        "pointer_args": [0],
        "supported": {
            "a2a3": ["int32", "int64", "fp16", "fp32", "bf16"],
            "a5": ["uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16"],
        },
    },
    "atomic_and": {
        "names": ["triton.language.atomic_and"],
        "tensor_args": [0, 1],
        "pointer_args": [0],
        "supported": {
            "a2a3": ["int32", "int64"],
            "a5": ["uint32", "int32", "uint64", "int64"],
        },
    },
    "atomic_cas": {
        "names": ["triton.language.atomic_cas"],
        "tensor_args": [0, 1, 2],
        "pointer_args": [0],
        "supported": {
            "a2a3": ["int16", "int32", "int64", "fp16", "fp32", "bf16"],
            "a5": ["uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16"],
        },
    },
    "atomic_max": {
        "names": ["triton.language.atomic_max"],
        "tensor_args": [0, 1],
        "pointer_args": [0],
        "supported": {
            "a2a3": ["int32", "int64", "fp32"],
            "a5": ["uint32", "int32", "uint64", "int64", "fp32"],
        },
    },
    "atomic_min": {
        "names": ["triton.language.atomic_min"],
        "tensor_args": [0, 1],
        "pointer_args": [0],
        "supported": {
            "a2a3": ["int32", "int64", "fp32"],
            "a5": ["uint32", "int32", "uint64", "int64", "fp32"],
        },
    },
    "atomic_or": {
        "names": ["triton.language.atomic_or"],
        "tensor_args": [0, 1],
        "pointer_args": [0],
        "supported": {
            "a2a3": ["int32", "int64"],
            "a5": ["uint32", "int32", "uint64", "int64"],
        },
    },
    "atomic_xchg": {
        "names": ["triton.language.atomic_xchg"],
        "tensor_args": [0, 1],
        "pointer_args": [0],
        "supported": {
            "a2a3": ["int32", "int64", "fp32"],
            "a5": ["uint32", "int32", "uint64", "int64", "fp32"],
        },
    },
    "atomic_xor": {
        "names": ["triton.language.atomic_xor"],
        "tensor_args": [0, 1],
        "pointer_args": [0],
        "supported": {
            "a2a3": ["int32", "int64"],
            "a5": ["uint32", "int32", "uint64", "int64"],
        },
    },
    "bitonic_merge": {
        "names": ["triton.language.bitonic_merge"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2", "bool"],
        },
    },
    "broadcast": {
        "names": ["triton.language.broadcast"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "broadcast_to": {
        "names": ["triton.language.broadcast_to"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "cast": {
        "names": ["triton.language.cast"],
        "tensor_args": [1],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "cat": {
        "names": ["triton.language.cat"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "cdiv": {
        "names": ["triton.language.cdiv"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"],
        },
    },
    "ceil": {
        "names": ["triton.language.ceil"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "clamp": {
        "names": ["triton.language.clamp"],
        "supported": {
            "a2a3": ["fp16", "fp32", "bf16"],
            "a5": ["fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    },
    "condition": {
        "names": ["triton.language.condition"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "cos": {
        "names": ["triton.language.cos"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "cumprod": {
        "names": ["triton.language.cumprod"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "cumsum": {
        "names": ["triton.language.cumsum"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "device_assert": {
        "names": ["triton.language.device_assert"],
        "tensor_args": [0],
        "supported": {
            "a2a3": ["bool"],
            "a5": ["bool"],
        },
    },
    "device_print": {
        "names": ["triton.language.device_print"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "div": {
        "dunder": ["__truediv__", "__rtruediv__"],
        "dunder_label": '/',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "div_rn": {
        "names": ["triton.language.div_rn"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "dot": {
        "names": ["triton.language.dot"],
        "tensor_args": [0, 1],
        "unsupported": {
            "a2a3":
            ["uint8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp64", "fp8e4m3", "fp8e5m2", "bool"],
            "a5": ["uint8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp64", "bool"],
        },
        "supported": {
            "a2a3": ["int4", "int8", "fp16", "fp32", "bf16"],
            "a5": ["int8", "fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    },
    "dot_scaled": {
        "names": ["triton.language.dot_scaled"],
        "tensor_args": [0, 3],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["fp16", "bf16"],
            "a5": ["fp16", "bf16", "fp8e4m3", "fp8e5m2", "fp4e2m1"],
        },
    },
    "eq": {
        "dunder": ["__eq__", "__req__"],
        "dunder_label": '==',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "erf": {
        "names": ["triton.language.erf"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "exp": {
        "names": ["triton.language.exp"],
        "supported": {
            "a2a3": ["fp16", "fp32"],
            "a5": ["fp16", "fp32"],
        },
    },
    "exp2": {
        "names": ["triton.language.exp2"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "expand_dims": {
        "names": ["triton.language.expand_dims"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "fdiv": {
        "names": ["triton.language.fdiv"],
        "supported": {
            "a2a3": ["fp16", "fp32", "bf16"],
            "a5": ["fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    },
    "flip": {
        "names": ["triton.language.flip"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "floor": {
        "names": ["triton.language.floor"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "floordiv": {
        "dunder": ["__floordiv__", "__rfloordiv__"],
        "dunder_label": '//',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"],
        },
    },
    "fma": {
        "names": ["triton.language.fma"],
        "supported": {
            "a2a3": ["fp16", "fp32", "bf16"],
            "a5": ["fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    },
    "full": {
        "names": ["triton.language.full"],
        "tensor_args": [1, 2],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "gather": {
        "names": ["triton.language.gather"],
        "tensor_args": [0],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["fp16", "fp32", "bf16", "bool"],
            "a5": ["fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2", "bool"],
        },
    },
    "ge": {
        "dunder": ["__ge__", "__rge__"],
        "dunder_label": '>=',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "gt": {
        "dunder": ["__gt__", "__rgt__"],
        "dunder_label": '>',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "histogram": {
        "names": ["triton.language.histogram"],
        "tensor_args": [0],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"],
        },
    },
    "inline_asm_elementwise": {
        "names": ["triton.language.inline_asm_elementwise"],
        "tensor_args": [3],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp32"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp32"],
        },
    },
    "interleave": {
        "names": ["triton.language.interleave"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "invert": {
        "dunder": ["__invert__"],
        "dunder_label": '~',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "bool"],
        },
    },
    "join": {
        "names": ["triton.language.join"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "le": {
        "dunder": ["__le__", "__rle__"],
        "dunder_label": '<=',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "load": {
        "names": ["triton.language.load"],
        "tensor_args": [0, 2],
        "pointer_args": [0],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "load_tensor_descriptor": {
        "names": ["triton.language.load_tensor_descriptor"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2"
            ],
        },
    },
    "log": {
        "names": ["triton.language.log"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "log2": {
        "names": ["triton.language.log2"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "logical_and": {
        "names": ["triton.language.tensor.logical_and"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
        },
    },
    "logical_or": {
        "names": ["triton.language.tensor.logical_or"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
        },
    },
    "lshift": {
        "dunder": ["__lshift__", "__rlshift__"],
        "dunder_label": '<<',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "bool"],
        },
    },
    "lt": {
        "dunder": ["__lt__", "__rlt__"],
        "dunder_label": '<',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "make_block_ptr": {
        "names": ["triton.language.make_block_ptr"],
        "pointer_args": [0],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "make_tensor_descriptor": {
        "names": ["triton.language.make_tensor_descriptor"],
        "pointer_args": [0],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2"
            ],
        },
    },
    "max": {
        "names": ["triton.language.max"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "max_constancy": {
        "names": ["triton.language.max_constancy"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "max_contiguous": {
        "names": ["triton.language.max_contiguous"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "maximum": {
        "names": ["triton.language.maximum"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "int16", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2",
                "bool"
            ],
        },
    },
    "min": {
        "names": ["triton.language.min"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "minimum": {
        "names": ["triton.language.minimum"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "int16", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2",
                "bool"
            ],
        },
    },
    "mod": {
        "dunder": ["__mod__", "__rmod__"],
        "dunder_label": '%',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "mul": {
        "names": ["triton.language.mul"],
        "dunder": ["__mul__", "__rmul__"],
        "dunder_label": '*',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "multiple_of": {
        "names": ["triton.language.multiple_of"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "ne": {
        "dunder": ["__ne__", "__rne__"],
        "dunder_label": '!=',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "neg": {
        "dunder": ["__neg__"],
        "dunder_label": 'unary -',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["int8", "int16", "int32", "int64", "fp16", "fp32", "bf16"],
            "a5": ["int8", "int16", "int32", "int64", "fp16", "fp32", "bf16"],
        },
    },
    "not": {
        "dunder": ["__not__"],
        "dunder_label": 'not',
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["bool"],
        },
    },
    "or": {
        "dunder": ["__or__", "__ror__"],
        "dunder_label": '|',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "bool"],
        },
    },
    "permute": {
        "names": ["triton.language.permute"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "rand": {
        "names": ["triton.language.rand"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
        },
    },
    "randint": {
        "names": ["triton.language.randint"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
        },
    },
    "randint4x": {
        "names": ["triton.language.randint4x"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
        },
    },
    "randn": {
        "names": ["triton.language.randn"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
        },
    },
    "range": {
        "names": ["triton.language.range"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"],
        },
    },
    "ravel": {
        "names": ["triton.language.ravel"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "reduce": {
        "names": ["triton.language.reduce"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "reduce_or": {
        "names": ["triton.language.reduce_or"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"],
        },
    },
    "reshape": {
        "names": ["triton.language.reshape"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "rshift": {
        "dunder": ["__rshift__", "__rrshift__"],
        "dunder_label": '>>',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "bool"],
        },
    },
    "rsqrt": {
        "names": ["triton.language.rsqrt"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "sigmoid": {
        "names": ["triton.language.sigmoid"],
        "supported": {
            "a2a3": ["fp16", "fp32"],
            "a5": ["fp16", "fp32"],
        },
    },
    "sin": {
        "names": ["triton.language.sin"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "softmax": {
        "names": ["triton.language.softmax"],
        "supported": {
            "a2a3": ["fp16", "fp32", "bf16"],
            "a5": ["fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    },
    "sort": {
        "names": ["triton.language.sort"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["int8", "int16", "fp16", "fp32", "bf16"],
            "a5": ["int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2", "bool"],
        },
    },
    "split": {
        "names": ["triton.language.split"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "sqrt": {
        "names": ["triton.language.sqrt"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "sqrt_rn": {
        "names": ["triton.language.sqrt_rn"],
        "supported": {
            "a2a3": ["fp32"],
            "a5": ["fp32"],
        },
    },
    "static_assert": {
        "names": ["triton.language.static_assert"],
        "supported": {
            "a2a3": ["bool"],
            "a5": ["bool"],
        },
    },
    "static_print": {
        "names": ["triton.language.static_print"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": ["int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "static_range": {
        "names": ["triton.language.static_range"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"],
        },
    },
    "store": {
        "names": ["triton.language.store"],
        "tensor_args": [0, 1],
        "pointer_args": [0],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "store_tensor_descriptor": {
        "names": ["triton.language.store_tensor_descriptor"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2"
            ],
        },
    },
    "sub": {
        "names": ["triton.language.sub"],
        "dunder": ["__sub__", "__rsub__"],
        "dunder_label": '-',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "sum": {
        "names": ["triton.language.sum"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": ["int8", "int16", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2", "bool"],
        },
    },
    "swizzle2d": {
        "names": ["triton.language.swizzle2d"],
        "supported": {
            "a2a3": ["int32", "int64"],
            "a5": ["int32", "int64"],
        },
    },
    "topk": {
        "names": ["triton.language.topk"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["int8", "int16", "fp16", "fp32", "bf16"],
            "a5": ["int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2", "bool"],
        },
    },
    "trans": {
        "names": ["triton.language.trans"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "umulhi": {
        "names": ["triton.language.umulhi"],
        "supported": {
            "a2a3": ["int32"],
            "a5": ["int32"],
        },
    },
    "view": {
        "names": ["triton.language.view"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
                "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16",
                "fp8e4m3", "fp8e5m2", "bool"
            ],
        },
    },
    "where": {
        "names": ["triton.language.where"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5":
            ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16", "bool"],
        },
    },
    "xor": {
        "dunder": ["__xor__", "__rxor__"],
        "dunder_label": '^',
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "bool"],
        },
    },
    "xor_sum": {
        "names": ["triton.language.xor_sum"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["int8", "int16", "int32", "int64", "bool"],
            "a5": ["int8", "int16", "int32", "uint64", "int64", "bool"],
        },
    },
    "zeros": {
        "names": ["triton.language.zeros"],
        "tensor_args": [1],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16"],
        },
    },
    "zeros_like": {
        "names": ["triton.language.zeros_like"],
        "unsupported": {
            "a2a3": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2a3": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "fp16", "fp32", "bf16"],
        },
    },
}


def _build_lookup_indexes():
    by_name = {}
    for rule_id, rule in OP_DTYPE_RULES.items():
        for name in rule.get("names", []):
            by_name[name] = rule_id
    return by_name


RULES_BY_NAME = _build_lookup_indexes()
