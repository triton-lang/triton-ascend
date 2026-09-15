# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Machine-readable operator dtype support configuration for the Ascend backend.
#
# Source of truth: docs/zh/python-api/_ascend_constraints.py (the Ascend
# Python-API documentation). Only the "DataType" constraints are transcribed
# here; non-dtype constraints (cache modifiers, synchronization scopes, shape
# requirements, ...) stay in the documentation and are intentionally excluded.
#
# ---------------------------------------------------------------------------
# Rule schema
# ---------------------------------------------------------------------------
#   "<rule id>": {
#       # Fully qualified names used to match named operators reached through
#       # ``tl.<name>`` builtins / @jit composite ops and cann extension ops.
#       "names": ["triton.language.add", ...],
#
#       # TritonSemantic method names that funnel the same operation. This is
#       # how operator overloads (e.g. ``a + b``) are covered: they call
#       # semantic methods directly and never go through ``call_Function``.
#       "semantic": ["add", ...],
#
#       # Positions (0-based) of tensor-like operands to inspect. For
#       # ``names`` these are positions in the public builtin/JIT arguments;
#       # for ``semantic`` these are positions in the semantic method
#       # arguments (``self`` excluded). When omitted, every positional
#       # argument is inspected (non-tensor arguments are ignored).
#       "tensor_args": [0, 1],
#
#       # Positions (0-based) that hold a pointer whose *pointee* dtype is the
#       # value to check (e.g. the pointer of ``load`` / extension ``sort``).
#       # Pointer operands at any other inspected position are skipped, because
#       # arithmetic leaves are reused for address computation
#       # (``ptr + offsets``) and must not be read as scalar elementwise ops.
#       "pointer_args": [0],
#
#       # Unsupported scalar dtypes per architecture bucket:
#       #   "a3" -> Ascend a2/A3
#       #   "a5" -> Ascend 910_95 / 950
#       # A missing bucket means "no dtype restriction on that architecture".
#       "unsupported": {"a3": [...], "a5": [...]},
#   }
#
# Dtype tokens (matched against ``dtype.name``):
#   * regular names: int8/int16/int32/int64/uint8/uint16/uint32/uint64/
#     fp16/bf16/fp32/fp64/int1 (bool)
#   * "fp8"   : every fp8 variant
#   * "fp8e4" : the E4M3 variants (fp8e4nv/fp8e4b8/fp8e4b15)
#   * "fp8e5" : the E5M2 variants (fp8e5/fp8e5b16)

FP8E4_DTYPES = ["fp8e4nv", "fp8e4b8", "fp8e4b15"]
FP8E5_DTYPES = ["fp8e5", "fp8e5b16"]
FP8_DTYPES = FP8E4_DTYPES + FP8E5_DTYPES

# Hardware-limited dtype set shared by the layout / shape ops on A2/A3.
_LAYOUT_A3 = ["fp64", "fp8e4", "fp8e5", "uint16", "uint32", "uint64"]
_UINT16_32_64 = ["uint16", "uint32", "uint64"]
_FP64 = ["fp64"]

OP_DTYPE_RULES = {
    # ---------------------------------------------------------------- unary
    "abs": {
        "names": ["triton.language.abs"],
        "unsupported": {
            "a3": ["uint16", "uint32", "uint64", "fp64"],
            "a5": ["fp64"],
        },
    },
    "neg": {
        "names": ["triton.language.neg"],
        "semantic": ["minus"],
        "unsupported": {
            "a3": ["bool", "fp64", "uint16", "uint32", "uint64"],
            "a5": ["bool", "fp64", "uint16", "uint32", "uint64"],
        },
    },
    "exp": {
        "names": ["triton.language.exp"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "exp2": {
        "names": ["triton.language.exp2"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "log": {
        "names": ["triton.language.log"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "log2": {
        "names": ["triton.language.log2"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "cos": {
        "names": ["triton.language.cos"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "sin": {
        "names": ["triton.language.sin"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "sqrt": {
        "names": ["triton.language.sqrt"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "rsqrt": {
        "names": ["triton.language.rsqrt"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "sigmoid": {
        "names": ["triton.language.sigmoid"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "erf": {
        "names": ["triton.language.erf"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "ceil": {
        "names": ["triton.language.ceil"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "floor": {
        "names": ["triton.language.floor"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "fma": {
        "names": ["triton.language.fma"],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "umulhi": {
        "names": ["triton.language.umulhi"],
        "unsupported": {"a3": ["int64"], "a5": ["int64"]},
    },

    # --------------------------------------------------------------- binary
    "add": {
        "names": ["triton.language.add"],
        "semantic": ["add"],
        "tensor_args": [0, 1],
        "unsupported": {"a3": ["fp8", "fp64"], "a5": ["fp8", "fp64"]},
    },
    "sub": {
        "names": ["triton.language.sub"],
        "semantic": ["sub"],
        "tensor_args": [0, 1],
        "unsupported": {"a3": ["fp8", "fp64"], "a5": ["fp8", "fp64"]},
    },
    # ``tl.div`` is documented; the implementation is the ``/`` operator and
    # its semantic method ``truediv``.
    "div": {
        "names": ["triton.language.div"],
        "semantic": ["truediv"],
        "tensor_args": [0, 1],
        "unsupported": {
            "a3": ["uint8", "uint16", "uint32", "uint64", "fp8", "fp64"],
            "a5": ["uint8", "uint16", "uint32", "uint64", "fp8", "fp64"],
        },
    },
    "floordiv": {
        "names": ["triton.language.floordiv"],
        "semantic": ["floordiv"],
        "tensor_args": [0, 1],
        "unsupported": {"a3": _UINT16_32_64},
    },
    "fdiv": {
        "names": ["triton.language.fdiv"],
        "semantic": ["fdiv"],
        "tensor_args": [0, 1],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "mod": {
        "names": ["triton.language.mod"],
        "semantic": ["mod"],
        "tensor_args": [0, 1],
        "unsupported": {
            "a3": ["uint16", "uint32", "uint64", "fp8", "fp64"],
            "a5": ["fp8", "fp64"],
        },
    },
    "cdiv": {
        "names": ["triton.language.cdiv"],
        "tensor_args": [0, 1],
        "unsupported": {
            "a3": ["uint16", "uint32", "uint64", "bool"],
            "a5": ["bool"],
        },
    },
    "maximum": {
        "names": ["triton.language.maximum"],
        "semantic": ["maximum"],
        "tensor_args": [0, 1],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "minimum": {
        "names": ["triton.language.minimum"],
        "semantic": ["minimum"],
        "tensor_args": [0, 1],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "clamp": {
        "names": ["triton.language.clamp"],
        "semantic": ["clamp"],
        "tensor_args": [0, 1, 2],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "where": {
        "names": ["triton.language.where"],
        "semantic": ["where"],
        # The predicate (index 0, int1) is not value-typed and is excluded.
        "tensor_args": [1, 2],
        "unsupported": {
            "a3": ["fp64", "uint16", "uint32", "uint64", "uint8"],
            "a5": ["fp64", "uint16", "uint32", "uint64", "uint8"],
        },
    },

    # ------------------------------------------------------------- comparison
    "equal": {
        "names": ["triton.language.equal"],
        "semantic": ["equal"],
        "tensor_args": [0, 1],
        "unsupported": {
            "a3": ["uint16", "uint32", "uint64", "fp64"],
            "a5": ["fp64"],
        },
    },
    "greater_than": {
        "names": ["triton.language.greater_than"],
        "semantic": ["greater_than"],
        "tensor_args": [0, 1],
        "unsupported": {
            "a3": ["fp64", "uint16", "uint32", "uint64"],
            "a5": ["fp64", "uint16", "uint32", "uint64"],
        },
    },
    "greater_equal": {
        "names": ["triton.language.greater_equal"],
        "semantic": ["greater_equal"],
        "tensor_args": [0, 1],
        "unsupported": {
            "a3": ["uint16", "uint32", "uint64", "fp64"],
            "a5": ["fp64"],
        },
    },
    "less_than": {
        "names": ["triton.language.less_than"],
        "semantic": ["less_than"],
        "tensor_args": [0, 1],
        "unsupported": {
            "a3": ["uint16", "uint32", "uint64", "fp64"],
            "a5": ["fp64"],
        },
    },
    "less_equal": {
        "names": ["triton.language.less_equal"],
        "semantic": ["less_equal"],
        "tensor_args": [0, 1],
        "unsupported": {
            "a3": ["uint16", "uint32", "uint64", "fp64"],
            "a5": ["fp64"],
        },
    },

    # -------------------------------------------------------------- reductions
    "max": {
        "names": ["triton.language.max"],
        "tensor_args": [0],
        "unsupported": {
            "a3": ["fp64", "uint16", "uint32", "uint64"],
            "a5": ["fp64", "fp8e4", "fp8e5"],
        },
    },
    "min": {
        "names": ["triton.language.min"],
        "tensor_args": [0],
        "unsupported": {
            "a3": ["fp64", "uint16", "uint32", "uint64"],
            "a5": ["fp64", "fp8e4", "fp8e5"],
        },
    },
    "argmax": {
        "names": ["triton.language.argmax"],
        "tensor_args": [0],
        "unsupported": {
            "a3": ["fp64", "uint16", "uint32", "uint64"],
            "a5": ["fp64", "fp8e4", "fp8e5"],
        },
    },
    "argmin": {
        "names": ["triton.language.argmin"],
        "tensor_args": [0],
        "unsupported": {
            "a3": ["fp64", "uint16", "uint32", "uint64"],
            "a5": ["fp64", "fp8e4", "fp8e5"],
        },
    },
    "sum": {
        "names": ["triton.language.sum"],
        "tensor_args": [0],
        "unsupported": {"a3": _UINT16_32_64},
    },
    "cumsum": {
        "names": ["triton.language.cumsum"],
        "tensor_args": [0],
        "unsupported": {"a3": _UINT16_32_64},
    },
    "cumprod": {
        "names": ["triton.language.cumprod"],
        "tensor_args": [0],
        "unsupported": {"a3": _UINT16_32_64},
    },
    "xor_sum": {
        "names": ["triton.language.xor_sum"],
        "tensor_args": [0],
        "unsupported": {
            "a3": _UINT16_32_64,
            "a5": ["fp8e4", "fp8e5"],
        },
    },
    "reduce": {
        "names": ["triton.language.reduce"],
        "tensor_args": [0],
        "unsupported": {
            "a3": _UINT16_32_64,
            "a5": ["fp8e4", "fp8e5"],
        },
    },
    "associative_scan": {
        "names": ["triton.language.associative_scan"],
        "tensor_args": [0, 1],
        "unsupported": {"a3": _UINT16_32_64},
    },

    # ------------------------------------------------------- layout / shape ops
    "permute": {
        "names": ["triton.language.permute"],
        "semantic": ["permute"],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "broadcast": {
        "names": ["triton.language.broadcast"],
        "semantic": ["broadcast_impl_value"],
        "tensor_args": [0, 1],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "broadcast_to": {
        "names": ["triton.language.broadcast_to"],
        "tensor_args": [0],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "cast": {
        "names": ["triton.language.cast"],
        "semantic": ["cast"],
        # Destination dtype is the second argument.
        "tensor_args": [1],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "expand_dims": {
        "names": ["triton.language.expand_dims"],
        "semantic": ["expand_dims"],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "reshape": {
        "names": ["triton.language.reshape"],
        "semantic": ["reshape"],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "view": {
        "names": ["triton.language.view"],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "ravel": {
        "names": ["triton.language.ravel"],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "trans": {
        "names": ["triton.language.trans"],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "interleave": {
        "names": ["triton.language.interleave"],
        "tensor_args": [0, 1],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "join": {
        "names": ["triton.language.join"],
        "semantic": ["join"],
        "tensor_args": [0, 1],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "split": {
        "names": ["triton.language.split"],
        "semantic": ["split"],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "cat": {
        "names": ["triton.language.cat"],
        "semantic": ["cat"],
        "tensor_args": [0, 1],
        "unsupported": {"a3": list(_UINT16_32_64)},
    },
    "flip": {
        "names": ["triton.language.flip"],
        "unsupported": {
            "a3": ["fp64", "uint16", "uint32", "uint64", "uint8"],
            "a5": ["fp64", "uint16", "uint32", "uint64", "uint8"],
        },
    },

    # ------------------------------------------------------------ constructors
    "full": {
        "names": ["triton.language.full"],
        "semantic": ["full"],
        # full(shape, value, dtype): the destination dtype is argument 2.
        "tensor_args": [2],
        "unsupported": {"a3": list(_UINT16_32_64)},
    },
    "zeros": {
        "names": ["triton.language.zeros"],
        # zeros(shape, dtype): the destination dtype is argument 1.
        "tensor_args": [1],
        "unsupported": {"a3": list(_UINT16_32_64)},
    },
    "zeros_like": {
        "names": ["triton.language.zeros_like"],
        "tensor_args": [0],
        "unsupported": {"a3": list(_UINT16_32_64)},
    },

    # ------------------------------------------------- pointers / load / store
    "load": {
        "names": ["triton.language.load"],
        "semantic": ["load"],
        # The loaded element dtype is derived from the pointer (argument 0).
        "tensor_args": [0],
        "pointer_args": [0],
        "unsupported": {
            "a3": ["uint16", "uint32", "uint64", "fp64"],
            "a5": ["fp64"],
        },
    },
    "store": {
        "names": ["triton.language.store"],
        "semantic": ["store"],
        "tensor_args": [1],
        "unsupported": {
            "a3": ["uint16", "uint32", "uint64", "fp64"],
            "a5": ["fp64"],
        },
    },
    "gather": {
        "names": ["triton.language.gather"],
        "semantic": ["gather"],
        "tensor_args": [0],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },

    # ---------------------------------------------------------------- atomics
    "atomic_add": {
        "names": ["triton.language.atomic_add"],
        "semantic": ["atomic_add"],
        "tensor_args": [1],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "atomic_cas": {
        "names": ["triton.language.atomic_cas"],
        "semantic": ["atomic_cas"],
        "tensor_args": [1, 2],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "atomic_xchg": {
        "names": ["triton.language.atomic_xchg"],
        "semantic": ["atomic_xchg"],
        "tensor_args": [1],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },

    # ------------------------------------------------------------------- misc
    "softmax": {
        "names": ["triton.language.softmax"],
        "tensor_args": [0],
        "unsupported": {"a3": _FP64, "a5": _FP64},
    },
    "topk": {
        "names": ["triton.language.topk"],
        "tensor_args": [0],
        "unsupported": {
            "a3": ["bool", "fp64", "int32", "int64", "uint8"],
            "a5": ["bool", "fp64", "int32", "int64", "uint8"],
        },
    },
    "device_print": {
        "names": ["triton.language.device_print"],
        "unsupported": {
            "a3": ["uint16", "uint32", "uint64", "fp64"],
            "a5": ["fp64"],
        },
    },
    "static_print": {
        "names": ["triton.language.static_print"],
        "unsupported": {
            "a3": ["uint16", "uint32", "uint64", "fp64"],
            "a5": ["fp64"],
        },
    },

    # ------------------------------------------------- cann extension builtins
    "ext.insert_slice": {
        "names": ["triton.language.extra.cann.extension.insert_slice"],
        "unsupported": {"a3": ["bool"], "a5": ["bool"]},
    },
    "ext.extract_slice": {
        "names": ["triton.language.extra.cann.extension.extract_slice"],
        "unsupported": {"a3": ["bool"], "a5": ["bool"]},
    },
    "ext.get_element": {
        "names": ["triton.language.extra.cann.extension.get_element"],
        "unsupported": {"a3": ["bool"], "a5": ["bool"]},
    },
    "ext.sort": {
        "names": ["triton.language.extra.cann.extension.sort"],
        "tensor_args": [0],
        "pointer_args": [0],
        "unsupported": {
            "a3": ["bool", "fp64", "int32", "int64", "uint8"],
            "a5": ["bool", "fp64", "int32", "int64", "uint8"],
        },
    },
    "ext.index_select_simd": {
        "names": ["triton.language.extra.cann.extension.index_select_simd"],
        "tensor_args": [0],
        "unsupported": {"a3": list(_LAYOUT_a3)},
    },
    "ext.dot": {
        # Ascend cube: only f16/bf16/f32/int8 matrix operands are supported.
        "names": ["triton.language.extra.cann.extension.dot"],
        "tensor_args": [0, 1],
        "unsupported": {
            "a3": ["fp8", "fp64", "fp4", "int16", "int32", "int64",
                   "uint8", "uint16", "uint32", "uint64", "bool"],
            "a5": ["fp8", "fp64", "fp4", "int16", "int32", "int64",
                   "uint8", "uint16", "uint32", "uint64", "bool"],
        },
    },
    "dot_scaled": {
        "names": ["triton.language.dot_scaled"],
        "semantic": ["dot_scaled"],
        # The two matrix operands are lhs (0) and rhs (3).
        "tensor_args": [0, 3],
        "unsupported": {
            "a3": ["fp4", "fp8"],
            "a5": ["fp4", "fp8"],
        },
    },
}


def _build_lookup_indexes():
    by_name = {}
    by_semantic = {}
    for rule_id, rule in OP_DTYPE_RULES.items():
        for name in rule.get("names", []):
            by_name[name] = rule_id
        for method in rule.get("semantic", []):
            by_semantic[method] = rule_id
    return by_name, by_semantic


RULES_BY_NAME, RULES_BY_SEMANTIC = _build_lookup_indexes()
