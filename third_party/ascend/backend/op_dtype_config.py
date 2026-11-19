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
#       #   "a2" -> Ascend A2/A3 ; "a5" -> Ascend 950
#       #   * "unsupported": blacklist -- the interception decision uses this:
#       #     a dtype listed here is rejected at the front end.
#       #   * "supported" (without "unsupported"): whitelist -- used by the
#       #     operators that carry a frontend @_check_dtype whitelist in
#       #     math.py (exp/exp2/log/log2/cos/sin/sqrt/sqrt_rn/rsqrt/div_rn/
#       #     erf/floor/ceil/umulhi): every dtype NOT listed is rejected.
#       # Every blacklist rule also spells out an explicit "supported" list:
#       # the complement of that bucket's "unsupported" over
#       # ALL_SCALAR_DTYPES, in universe order. It drives the error message
#       # (and overrides the runtime derivation when deliberately narrower,
#       # as for "dot_scaled"). The interception decision itself always uses
#       # the "unsupported" blacklist.
#       # A missing bucket means "no dtype restriction on that architecture"
#       # (every dtype accepted).
#       "unsupported": {"a2": [...], "a5": [...]},   # blacklist mode
#       "supported":   {"a2": [...], "a5": [...]},   # whitelist mode
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
        'names': ['triton.language.abs'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
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
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "add": {
        'names': ['triton.language.add'],
        'dunder': ['__add__', '__radd__'],
        'dunder_label': '+',
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"],
            "a5": [
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
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool"
            ],
        }
    }, "advance": {
        'names': ['triton.language.advance'],
        'pointer_args': [0],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
            "a5": ["fp64", "bool"],
        },
    }, "and": {
        'dunder': ['__and__', '__rand__'],
        'dunder_label': '&',
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"],
        },
        "supported": {
            "a2": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "bool",
            ],
        },
    }, "arange": {
        'names': ['triton.language.arange'],
        "supported": {
            "a2": ["uint8", "int8", "int16", "int32"],
            "a5": ["uint8", "int8", "uint16", "int16", "int32"],
        },
    }, "argmax": {
        'names': ['triton.language.argmax'],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2"],
        },
    }, "argmin": {
        'names': ['triton.language.argmin'],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2"],
        },
    }, "associative_scan": {
        'names': ['triton.language.associative_scan'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64", "bf16", "fp8e4m3", "fp8e5m2"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
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
                "bool",
            ],
        },
    }, "assume": {
        'names': ['triton.language.assume'],
        "supported": {
            "a2": ["bool"],
            "a5": ["bool"],
        },
    }, "atomic_add": {
        'names': ['triton.language.atomic_add'],
        'tensor_args': [0, 1],
        'pointer_args': [0],
        "supported": {
            "a2": ["int32", "int64", "fp16", "fp32", "bf16"],
            "a5": [
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp16",
                "fp32",
                "bf16",
            ],
        },
    }, "atomic_and": {
        'names': ['triton.language.atomic_and'],
        'tensor_args': [0, 1],
        'pointer_args': [0],
        "supported": {
            "a2": ["int32", "int64"],
            "a5": ["uint32", "int32", "uint64", "int64"],
        },
    }, "atomic_cas": {
        'names': ['triton.language.atomic_cas'],
        'tensor_args': [0, 1, 2],
        'pointer_args': [0],
        "supported": {
            "a2": ["int16", "int32", "int64", "fp16", "fp32", "bf16"],
            "a5": [
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp16",
                "fp32",
                "bf16",
            ],
        },
    }, "atomic_max": {
        'names': ['triton.language.atomic_max'],
        'tensor_args': [0, 1],
        'pointer_args': [0],
        "supported": {
            "a2": ["int32", "int64", "fp32"],
            "a5": ["uint32", "int32", "uint64", "int64", "fp32"],
        },
    }, "atomic_min": {
        'names': ['triton.language.atomic_min'],
        'tensor_args': [0, 1],
        'pointer_args': [0],
        "supported": {
            "a2": ["int32", "int64", "fp32"],
            "a5": ["uint32", "int32", "uint64", "int64", "fp32"],
        },
    }, "atomic_or": {
        'names': ['triton.language.atomic_or'],
        'tensor_args': [0, 1],
        'pointer_args': [0],
        "supported": {
            "a2": ["int32", "int64"],
            "a5": ["uint32", "int32", "uint64", "int64"],
        },
    }, "atomic_xchg": {
        'names': ['triton.language.atomic_xchg'],
        'tensor_args': [0, 1],
        'pointer_args': [0],
        "supported": {
            "a2": ["int32", "int64", "fp32"],
            "a5": ["uint32", "int32", "uint64", "int64", "fp32"],
        },
    }, "atomic_xor": {
        'names': ['triton.language.atomic_xor'],
        'tensor_args': [0, 1],
        'pointer_args': [0],
        "supported": {
            "a2": ["int32", "int64"],
            "a5": ["uint32", "int32", "uint64", "int64"],
        },
    }, "bitonic_merge": {
        'names': ['triton.language.bitonic_merge'],
        "unsupported": {
            "a2": ["uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "broadcast": {
        'names': ['triton.language.broadcast'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "broadcast_to": {
        'names': ['triton.language.broadcast_to'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "cast": {
        'names': ['triton.language.cast'],
        'tensor_args': [1],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "cat": {
        'names': ['triton.language.cat'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "cdiv": {
        'names': ['triton.language.cdiv'],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "ceil": {
        'names': ['triton.language.ceil'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "clamp": {
        'names': ['triton.language.clamp'],
        "supported": {
            "a2": ["fp16", "fp32", "bf16"],
            "a5": ["fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    }, "condition": {
        'names': ['triton.language.condition'],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
        },
    }, "cos": {
        'names': ['triton.language.cos'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "cumprod": {
        'names': ['triton.language.cumprod'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
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
                "bf16",
                "bool",
            ],
        },
    }, "cumsum": {
        'names': ['triton.language.cumsum'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
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
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "device_assert": {
        'names': ['triton.language.device_assert'],
        'tensor_args': [0],
        "supported": {
            "a2": ["bool"],
            "a5": ["bool"],
        },
    }, "device_print": {
        'names': ['triton.language.device_print'],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "div": {
        'dunder': ['__truediv__', '__rtruediv__'],
        'dunder_label': '/',
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "div_rn": {
        'names': ['triton.language.div_rn'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "dot": {
        'names': ['triton.language.dot'],
        'tensor_args': [0, 1],
        "unsupported": {
            "a2": [
                "uint8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
            "a5": [
                "uint8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp64",
                "bool",
            ],
        },
        "supported": {
            "a2": ["int4", "int8", "fp16", "fp32", "bf16"],
            "a5": ["int8", "fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    }, "dot_scaled": {
        'names': ['triton.language.dot_scaled'],
        'tensor_args': [0, 3],
        "unsupported": {
            "a2": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp32",
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
            "a5": [
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp32",
                "fp64",
                "bool",
            ],
        },
        "supported": {
            "a2": ["fp16", "bf16"],
            "a5": ["fp16", "bf16", "fp8e4m3", "fp8e5m2", "fp4e2m1"],
        },
    }, "eq": {
        'dunder': ['__eq__', '__req__'],
        'dunder_label': '==',
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "erf": {
        'names': ['triton.language.erf'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "exp": {
        'names': ['triton.language.exp'],
        "supported": {
            "a2": ["fp16", "fp32"],
            "a5": ["fp16", "fp32"],
        },
    }, "exp2": {
        'names': ['triton.language.exp2'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "expand_dims": {
        'names': ['triton.language.expand_dims'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "fdiv": {
        'names': ['triton.language.fdiv'],
        "supported": {
            "a2": ["fp16", "fp32", "bf16"],
            "a5": ["fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    }, "flip": {
        'names': ['triton.language.flip'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2"],
        },
    }, "floor": {
        'names': ['triton.language.floor'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "floordiv": {
        'dunder': ['__floordiv__', '__rfloordiv__'],
        'dunder_label': '//',
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "fma": {
        'names': ['triton.language.fma'],
        "supported": {
            "a2": ["fp16", "fp32", "bf16"],
            "a5": ["fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    }, "full": {
        'names': ['triton.language.full'],
        'tensor_args': [1, 2],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2"],
        },
    }, "gather": {
        'names': ['triton.language.gather'],
        'tensor_args': [0],
        "unsupported": {
            "a2": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp64",
            ],
        },
    }, "ge": {
        'dunder': ['__ge__', '__rge__'],
        'dunder_label': '>=',
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "gt": {
        'dunder': ['__gt__', '__rgt__'],
        'dunder_label': '>',
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
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
                "bf16",
                "bool",
            ],
        },
    }, "histogram": {
        'names': ['triton.language.histogram'],
        'tensor_args': [0],
        "unsupported": {
            "a2": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
            ],
        },
    }, "inline_asm_elementwise": {
        'names': ['triton.language.inline_asm_elementwise'],
        'tensor_args': [3],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp16",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
            "a5": [
                "fp16",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
        "supported": {
            "a2": ["uint8", "int8", "int16", "int32", "int64", "fp32"],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "int64",
                "fp32",
            ],
        },
    }, "interleave": {
        'names': ['triton.language.interleave'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "invert": {
        'dunder': ['__invert__'],
        'dunder_label': '~',
        "unsupported": {
            "a2": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
        },
        "supported": {
            "a2": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "bool"],
        },
    }, "join": {
        'names': ['triton.language.join'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "le": {
        'dunder': ['__le__', '__rle__'],
        'dunder_label': '<=',
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "load": {
        'names': ['triton.language.load'],
        'tensor_args': [0, 2],
        'pointer_args': [0],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "load_tensor_descriptor": {
        'names': ['triton.language.load_tensor_descriptor'],
        "unsupported": {
            "a2": [
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
            "a5": ["fp64", "bool"],
        },
    }, "log": {
        'names': ['triton.language.log'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "log2": {
        'names': ['triton.language.log2'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "logical_and": {
        'names': ['triton.language.tensor.logical_and'],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    }, "logical_or": {
        'names': ['triton.language.tensor.logical_or'],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    }, "lshift": {
        'dunder': ['__lshift__', '__rlshift__'],
        'dunder_label': '<<',
        "unsupported": {
            "a2": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
        },
        "supported": {
            "a2": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "bool"],
        },
    }, "lt": {
        'dunder': ['__lt__', '__rlt__'],
        'dunder_label': '<',
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "make_block_ptr": {
        'names': ['triton.language.make_block_ptr'],
        'pointer_args': [0],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "make_tensor_descriptor": {
        'names': ['triton.language.make_tensor_descriptor'],
        'pointer_args': [0],
        "unsupported": {
            "a2": [
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
            "a5": ["fp64", "bool"],
        },
    }, "max": {
        'names': ['triton.language.max'],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2"],
        },
    }, "max_constancy": {
        'names': ['triton.language.max_constancy'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
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
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "max_contiguous": {
        'names': ['triton.language.max_contiguous'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
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
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "maximum": {
        'names': ['triton.language.maximum'],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "uint64",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "min": {
        'names': ['triton.language.min'],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2"],
        },
    }, "minimum": {
        'names': ['triton.language.minimum'],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
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
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "mod": {
        'dunder': ['__mod__', '__rmod__'],
        'dunder_label': '%',
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "mul": {
        'names': ['triton.language.mul'],
        'dunder': ['__mul__', '__rmul__'],
        'dunder_label': '*',
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "multiple_of": {
        'names': ['triton.language.multiple_of'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
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
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "ne": {
        'dunder': ['__ne__', '__rne__'],
        'dunder_label': '!=',
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "neg": {
        'dunder': ['__neg__'],
        'dunder_label': 'unary -',
        "unsupported": {
            "a2": [
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
            "a5": [
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "not": {
        'dunder': ['__not__'],
        'dunder_label': 'not',
        "supported": {
            "a2": ["bool"],
            "a5": ["bool"],
        },
    }, "or": {
        'dunder': ['__or__', '__ror__'],
        'dunder_label': '|',
        "unsupported": {
            "a2": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
        },
        "supported": {
            "a2": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "bool"],
        },
    }, "permute": {
        'names': ['triton.language.permute'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "rand": {
        'names': ['triton.language.rand'],
        "unsupported": {
            "a2": ["fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    }, "randint": {
        'names': ['triton.language.randint'],
        "unsupported": {
            "a2": ["fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    }, "randint4x": {
        'names': ['triton.language.randint4x'],
        "unsupported": {
            "a2": ["fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    }, "randn": {
        'names': ['triton.language.randn'],
        "unsupported": {
            "a2": ["fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    }, "range": {
        'names': ['triton.language.range'],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "ravel": {
        'names': ['triton.language.ravel'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "reduce": {
        'names': ['triton.language.reduce'],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2"],
        },
    }, "reduce_or": {
        'names': ['triton.language.reduce_or'],
        "unsupported": {
            "a2": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
        },
        "supported": {
            "a2": ["uint8", "int8", "int16", "int32"],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
            ],
        },
    }, "reshape": {
        'names': ['triton.language.reshape'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "rshift": {
        'dunder': ['__rshift__', '__rrshift__'],
        'dunder_label': '>>',
        "unsupported": {
            "a2": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
        },
        "supported": {
            "a2": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "bool"],
        },
    }, "rsqrt": {
        'names': ['triton.language.rsqrt'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "sigmoid": {
        'names': ['triton.language.sigmoid'],
        "supported": {
            "a2": ["fp16", "fp32"],
            "a5": ["fp16", "fp32"],
        },
    }, "sin": {
        'names': ['triton.language.sin'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "softmax": {
        'names': ['triton.language.softmax'],
        "supported": {
            "a2": ["fp16", "fp32", "bf16"],
            "a5": ["fp16", "fp32", "bf16", "fp8e4m3", "fp8e5m2"],
        },
    }, "sort": {
        'names': ['triton.language.sort'],
        "unsupported": {
            "a2": [
                "uint8",
                "uint16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": ["int8", "int16", "fp16", "fp32", "bf16"],
            "a5": [
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "split": {
        'names': ['triton.language.split'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "sqrt": {
        'names': ['triton.language.sqrt'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "sqrt_rn": {
        'names': ['triton.language.sqrt_rn'],
        "supported": {
            "a2": ["fp32"],
            "a5": ["fp32"],
        },
    }, "static_assert": {
        'names': ['triton.language.static_assert'],
        "supported": {
            "a2": ["bool"],
            "a5": ["bool"],
        },
    }, "static_print": {
        'names': ['triton.language.static_print'],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": [
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
            ],
        },
        "supported": {
            "a2": [
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
        },
    }, "static_range": {
        'names': ['triton.language.static_range'],
        "unsupported": {
            "a2": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "store": {
        'names': ['triton.language.store'],
        'tensor_args': [0, 1],
        'pointer_args': [0],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "store_tensor_descriptor": {
        'names': ['triton.language.store_tensor_descriptor'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2", "bool"],
            "a5": ["fp64", "bool"],
        },
    }, "sub": {
        'names': ['triton.language.sub'],
        'dunder': ['__sub__', '__rsub__'],
        'dunder_label': '-',
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "sum": {
        'names': ['triton.language.sum'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "bool",
            ],
            "a5": [
                "int8",
                "int16",
                "int32",
                "uint64",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "swizzle2d": {
        'names': ['triton.language.swizzle2d'],
        "supported": {
            "a2": ["int32", "int64"],
            "a5": ["int32", "int64"],
        },
    }, "topk": {
        'names': ['triton.language.topk'],
        "unsupported": {
            "a2": [
                "fp64",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": ["fp64"],
        },
        "supported": {
            "a2": ["int8", "int16", "fp16", "fp32", "bf16"],
            "a5": [
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
                "bool",
            ],
        },
    }, "trans": {
        'names': ['triton.language.trans'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "umulhi": {
        'names': ['triton.language.umulhi'],
        "supported": {
            "a2": ["int32"],
            "a5": ["int32"],
        },
    }, "view": {
        'names': ['triton.language.view'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64"],
        },
    }, "where": {
        'names': ['triton.language.where'],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4m3", "fp8e5m2"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2"],
        },
    }, "xor": {
        'dunder': ['__xor__', '__rxor__'],
        'dunder_label': '^',
        "unsupported": {
            "a2": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
        },
        "supported": {
            "a2": ["uint8", "int8", "int16", "int32", "int64", "bool"],
            "a5": ["uint8", "int8", "int16", "int32", "int64", "bool"],
        },
    }, "xor_sum": {
        'names': ['triton.language.xor_sum'],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
            "a5": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4m3",
                "fp8e5m2",
            ],
        },
        "supported": {
            "a2": ["int8", "int16", "int32", "int64", "bool"],
            "a5": ["int8", "int16", "int32", "uint64", "int64", "bool"],
        },
    }, "zeros": {
        'names': ['triton.language.zeros'],
        'tensor_args': [1],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2", "bool"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2", "bool"],
        },
    }, "zeros_like": {
        'names': ['triton.language.zeros_like'],
        "unsupported": {
            "a2": ["fp64", "fp8e4m3", "fp8e5m2", "bool"],
            "a5": ["fp64", "fp8e4m3", "fp8e5m2", "bool"],
        },
    }
}


def _build_lookup_indexes():
    by_name = {}
    for rule_id, rule in OP_DTYPE_RULES.items():
        for name in rule.get("names", []):
            by_name[name] = rule_id
    return by_name


RULES_BY_NAME = _build_lookup_indexes()


def _build_lookup_indexes():
    by_name = {}
    for rule_id, rule in OP_DTYPE_RULES.items():
        for name in rule.get("names", []):
            by_name[name] = rule_id
    return by_name


RULES_BY_NAME = _build_lookup_indexes()
