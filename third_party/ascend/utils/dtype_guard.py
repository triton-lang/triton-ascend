# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Unified, non-intrusive dtype interception for unsupported Ascend operators.
#
# Two thin monkey-patch layers are installed (see ``install()``):
#
#   * The *semantic layer* wraps the arithmetic / comparison leaf methods of
#     ``triton.language.semantic.TritonSemantic``. Operator overloads
#     (``a + b``, ``a > b``, ``-a`` ...) are turned into tensor dunder builtins
#     and invoked directly by ``CodeGenerator._apply_binary_method``; they
#     never pass through ``call_Function``, so this is the only place where
#     they can be intercepted.
#
#   * The *entry layer* wraps ``CodeGenerator.call_Function`` and validates
#     named builtins (``tl.*``), ``@jit`` composite ops (``tl.max``,
#     ``tl.sum`` ...) and cann extension builtins before they run.
#
# The rules themselves live in ``op_dtype_config.py``. Only the operator
# leaves are wrapped at the semantic layer: infrastructure helpers such as
# ``cast`` / ``broadcast_impl_value`` / ``reshape`` are intentionally *not*
# wrapped there because the front-end type promotion / broadcasting machinery
# reuses them internally (they are still covered by the entry layer when the
# user calls them by name). A per-semantic-instance re-entrancy guard further
# suppresses checks for calls issued from inside another wrapped method (for
# example ``sub`` lowers pointer subtraction through ``minus``/``add``).

import importlib
import inspect

from .op_dtype_config import (
    OP_DTYPE_RULES,
    RULES_BY_NAME,
    RULES_BY_SEMANTIC,
    FP8_DTYPES,
    FP8E4_DTYPES,
    FP8E5_DTYPES,
)

_A5_PREFIXES = ("Ascend910_95", "Ascend950")
_ARCH_BUCKETS = ("a2", "a5")

# Semantic methods that are user-facing arithmetic / comparison leaves and are
# reachable *only* through operator overloads / unary minus. Named ops that are
# also callable via ``tl.*`` (gather, atomics, dot_scaled ...) are intentionally
# NOT patched here: they are validated at the entry layer, and wrapping their
# semantic method could double-check or misfire when the method is reused by
# internal lowering. Keep this set limited to true operator leaves.
_PATCHED_SEMANTIC_METHODS = (
    "add",
    "sub",
    "truediv",
    "floordiv",
    "fdiv",
    "mod",
    "minus",
    "maximum",
    "minimum",
    "clamp",
    "where",
    "equal",
    "greater_than",
    "greater_equal",
    "less_than",
    "less_equal",
)

_GUARD_ATTR = "_ascend_dtype_guard_active"

_LANG_MODULE_PREFIX = "triton.language"
_EXTRA_MARKER = ".extra"


class UnsupportedDtypeError(TypeError):
    """Raised at the front-end when an Ascend operator gets an unsupported dtype."""


def arch_bucket(arch):
    """Classify an ``options.arch`` string into an ``"a2"`` / ``"a5"`` bucket."""
    if isinstance(arch, str) and arch.startswith(_A5_PREFIXES):
        return "a5"
    # Fall back to the backend helper / runtime device name when the arch
    # option is not populated yet.
    try:
        from triton.backends.ascend.utils import is_compile_on_910_95

        if is_compile_on_910_95(arch if isinstance(arch, str) else None):
            return "a5"
    except Exception:
        pass
    return "a2"


def _bucket_from_semantic(semantic):
    options = getattr(getattr(semantic, "builder", None), "options", None)
    arch = getattr(options, "arch", None)
    return arch_bucket(arch)


def _is_fp8_token(token, name):
    if token == "fp8":
        return name in FP8_DTYPES
    if token == "fp8e4":
        return name in FP8E4_DTYPES
    if token == "fp8e5":
        return name in FP8E5_DTYPES
    return False


def _token_matches(token, name):
    if token == name:
        return True
    # ``bool`` is represented internally as ``int1``.
    if token == "bool" and name == "int1":
        return True
    if token == "int1" and name == "bool":
        return True
    return _is_fp8_token(token, name)


def _normalize_dtype_name(name):
    return "int1" if name == "bool" else name


def _extract_scalar_name(value, read_pointer=False):
    """Return the scalar dtype ``name`` of a traced value, or ``None``.

    Understands:
      * a ``tl.dtype`` itself (destination dtype of ``cast`` / ``full`` ...)
      * a tensor / block (``value.type.scalar``)
      * ``constexpr`` wrapping one of the above
      * a pointer tensor: its pointee element dtype is returned only when
        ``read_pointer`` is True (i.e. the position is declared in the rule's
        ``pointer_args``). Otherwise pointers are skipped, because arithmetic
        leaves are reused for address computation (``ptr + offsets``) and a
        pointee dtype there does not describe the op itself.
    Python scalars, shapes, strings and ``None`` yield ``None``.
    """
    # Unwrap constexpr without importing the (compile-time) class directly.
    value = getattr(value, "value", value) if value.__class__.__name__ == "constexpr" else value

    # A bare dtype object (e.g. the ``dtype`` argument of ``cast`` / ``full``).
    name = getattr(value, "name", None)
    if isinstance(name, str) and hasattr(value, "primitive_bitwidth") and not hasattr(value, "handle"):
        return _normalize_dtype_name(name)

    type_obj = getattr(value, "type", None)
    if type_obj is None:
        return None

    is_ptr = getattr(type_obj, "is_ptr", None)
    if callable(is_ptr) and is_ptr():
        if not read_pointer:
            return None
        element_ty = getattr(type_obj, "element_ty", None)
        return _normalize_dtype_name(getattr(element_ty, "name", None))

    scalar = type_obj
    scalar_prop = getattr(type_obj, "scalar", None)
    if scalar_prop is not None and not callable(scalar_prop):
        scalar = scalar_prop
    return _normalize_dtype_name(getattr(scalar, "name", None))


def _rule_label(rule_id):
    names = OP_DTYPE_RULES[rule_id].get("names")
    return names[0] if names else rule_id


def _unsupported_tokens(rule, bucket):
    unsupported = rule.get("unsupported", {})
    return unsupported.get(bucket, unsupported.get("a2", []))


def _align_operands(fn, args, kws):
    """Map runtime args/kws onto the public builtin positional parameters.

    This returns a list indexed exactly like the documented builtin signature
    (self of a member-style call included), so that ``tensor_args`` positions
    are stable regardless of whether an operand was passed positionally or by
    keyword (e.g. ``cast(x, dtype=...)`` / ``zeros(shape, dtype=...)``).
    Falls back to the raw args when introspection is unavailable.
    """
    try:
        signature_target = _unwrap_callable(fn)
        try:
            sig = inspect.signature(signature_target)
        except (TypeError, ValueError):
            return list(args)
        try:
            bound = sig.bind_partial(*args, **dict(kws or ()))
        except TypeError:
            return list(args)
        arguments = bound.arguments
        ordered = []
        for name, param in sig.parameters.items():
            if name in ("_semantic", "_generator"):
                continue
            if name in arguments:
                value = arguments[name]
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    ordered.extend(value)
                else:
                    ordered.append(value)
        return ordered
    except Exception:
        return list(args)


def validate(rule_id, args, bucket, op_label=None, kws=None, fn=None):
    """Validate selected operands against one rule.

    ``args`` are the raw runtime positional arguments. When ``fn``/``kws`` are
    provided (the entry layer), operands are aligned onto the builtin
    signature so keyword dst-dtypes land on their configured position.
    """
    rule = OP_DTYPE_RULES[rule_id]
    tokens = _unsupported_tokens(rule, bucket)
    if not tokens:
        return

    operands = _align_operands(fn, args, kws) if fn is not None else list(args)

    positions = rule.get("tensor_args")
    if positions is None:
        positions = range(len(operands))
    pointer_positions = set(rule.get("pointer_args") or ())

    label = op_label or _rule_label(rule_id)
    for index in positions:
        if index >= len(operands):
            continue
        name = _extract_scalar_name(operands[index], read_pointer=index in pointer_positions)
        if name is None:
            continue
        for token in tokens:
            if _token_matches(token, name):
                raise UnsupportedDtypeError(
                    f"[Ascend] operator '{label}' does not support dtype '{name}' "
                    f"(matched unsupported dtype '{token}') on Ascend "
                    f"{'910_95/950' if bucket == 'a5' else 'A2/A3'}; operand #{index}. "
                    f"Please convert the operand to a supported dtype "
                    f"(e.g. tl.cast(x, tl.float32)) before calling '{label}'."
                )


# ---------------------------------------------------------------------------
# Semantic layer
# ---------------------------------------------------------------------------

def _install_semantic_patches():
    from triton.language.semantic import TritonSemantic

    if getattr(TritonSemantic, "_ascend_dtype_semantic_patch", False):
        return

    for method_name in _PATCHED_SEMANTIC_METHODS:
        rule_id = RULES_BY_SEMANTIC.get(method_name)
        if rule_id is None:
            continue
        original = getattr(TritonSemantic, method_name, None)
        if original is None or getattr(original, "_ascend_guard_wrapped", False):
            continue

        def make_wrapper(orig, rid):
            def wrapper(self, *args, **kwargs):
                if not getattr(self, _GUARD_ATTR, False):
                    try:
                        bucket = _bucket_from_semantic(self)
                        validate(rid, list(args), bucket)
                    except UnsupportedDtypeError:
                        raise
                    except Exception:
                        # Never let guard bookkeeping break compilation; the
                        # original method still runs below as a fallback.
                        return orig(self, *args, **kwargs)
                    setattr(self, _GUARD_ATTR, True)
                    try:
                        return orig(self, *args, **kwargs)
                    finally:
                        setattr(self, _GUARD_ATTR, False)
                return orig(self, *args, **kwargs)

            wrapper._ascend_guard_wrapped = True
            wrapper.__name__ = getattr(orig, "__name__", rid)
            return wrapper

        setattr(TritonSemantic, method_name, make_wrapper(original, rule_id))

    TritonSemantic._ascend_dtype_semantic_patch = True


# ---------------------------------------------------------------------------
# Entry layer
# ---------------------------------------------------------------------------

def _build_short_name_index():
    lang_index = {}
    ext_index = {}
    for full_name in RULES_BY_NAME:
        short = full_name.rsplit(".", 1)[-1]
        if _EXTRA_MARKER in full_name:
            ext_index.setdefault(short, RULES_BY_NAME[full_name])
        else:
            lang_index.setdefault(short, RULES_BY_NAME[full_name])
    return lang_index, ext_index


_LANG_SHORT_INDEX, _EXT_SHORT_INDEX = _build_short_name_index()


def _unwrap_callable(fn):
    seen = set()
    for _ in range(8):
        if id(fn) in seen:
            break
        seen.add(id(fn))
        func = getattr(fn, "__func__", None)
        if func is not None:
            fn = func
            continue
        # JITFunction / ConstexprFunction wrap the raw Python function.
        inner = getattr(fn, "fn", None)
        if callable(inner):
            fn = inner
            continue
        wrapped = getattr(fn, "__wrapped__", None)
        if callable(wrapped):
            fn = wrapped
            continue
        # ``_tensor_member_fn`` registers a closure wrapper named "wrapper".
        if getattr(fn, "__name__", None) == "wrapper" and getattr(fn, "__closure__", None):
            candidates = []
            for cell in fn.__closure__:
                try:
                    contents = cell.cell_contents
                except ValueError:
                    continue
                if callable(contents) and getattr(contents, "__name__", None) not in (None, "wrapper"):
                    candidates.append(contents)
            if candidates:
                fn = candidates[0]
                continue
        break
    return fn


def _resolve_entry_rule(fn):
    target = _unwrap_callable(fn)
    module = getattr(target, "__module__", "") or ""
    qualname = getattr(target, "__qualname__", "") or getattr(target, "__name__", "") or ""
    if not qualname:
        return None

    basename = qualname.rsplit(".", 1)[-1]

    # Exact match, progressively dropping trailing module components:
    #   triton.language.standard.max -> triton.language.max
    #   triton.language.extra.cann.extension.vec_ops.insert_slice
    #       -> triton.language.extra.cann.extension.insert_slice
    parts = module.split(".")
    for cut in range(len(parts), 0, -1):
        candidate = ".".join(parts[:cut]) + "." + basename
        rule_id = RULES_BY_NAME.get(candidate)
        if rule_id is not None:
            return rule_id

    # Short-name fallback, constrained to the right operator family so that
    # e.g. the extension cube ``dot`` can never shadow the regular ``tl.dot``.
    if module.startswith(_LANG_MODULE_PREFIX):
        if _EXTRA_MARKER in module:
            return _EXT_SHORT_INDEX.get(basename)
        return _LANG_SHORT_INDEX.get(basename)
    return None


def _install_entry_patch():
    from triton.compiler.code_generator import (
        CodeGenerator,
        BoundJITMethod,
        BoundConstexprFunction,
        _is_triton_value,
    )

    if getattr(CodeGenerator, "_ascend_dtype_entry_patch", False):
        return

    original_call_function = CodeGenerator.call_Function

    def call_function(self, node, fn, args, kws):
        try:
            check_fn = fn
            check_args = args
            # Member-style calls (``x.sum()``, ``x.exp()``) bind the tensor as
            # the first operand outside of ``args``; restore it logically for
            # validation without mutating the arguments handed to the original
            # implementation (which performs the same insertion itself).
            if isinstance(fn, (BoundJITMethod, BoundConstexprFunction)):
                check_fn = fn.__func__
                check_args = [fn.__self__, *args]
            else:
                self_obj = getattr(fn, "__self__", None)
                if self_obj is not None and _is_triton_value(self_obj):
                    check_fn = getattr(fn, "__func__", fn)
                    check_args = [self_obj, *args]

            rule_id = _resolve_entry_rule(check_fn)
            if rule_id is not None:
                bucket = arch_bucket(getattr(getattr(self.builder, "options", None), "arch", None))
                validate(rule_id, list(check_args), bucket, fn=check_fn, kws=kws)
        except UnsupportedDtypeError as exc:
            # Re-raise the actual rejection attached to the user source node,
            # mirroring how ``call_Function`` wraps builtin failures so the
            # message points at the offending call rather than the compiler.
            from triton import knobs

            if knobs.compilation.front_end_debugging:
                raise
            from triton.compiler.errors import CompilationError

            raise CompilationError(self.jit_fn.src, node, str(exc)) from None
        except Exception:
            # Validation must never break compilation; only explicit dtype
            # rejections are allowed to propagate.
            pass
        return original_call_function(self, node, fn, args, kws)

    CodeGenerator.call_Function = call_function
    CodeGenerator._ascend_dtype_entry_patch = True


def install():
    """Install both interception layers. Idempotent."""
    _install_semantic_patches()
    _install_entry_patch()
