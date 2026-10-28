"""Operation identities and the first, deliberately finite semantic vocabulary."""
import ast
import builtins
import inspect
import types
from functools import lru_cache


def resolve(node, scope):
    if isinstance(node, ast.Name):
        return scope.get(node.id, vars(builtins).get(node.id))
    if isinstance(node, ast.Attribute):
        base = resolve(node.value, scope)
        # Never execute arbitrary user descriptors while analyzing a kernel.
        if isinstance(base, types.ModuleType):
            return vars(base).get(node.attr)
    return None


@lru_cache(maxsize=None)
def _identity_table(names):
    import triton.language as tl

    table = {}
    for name in names:
        obj = getattr(tl, name, None)
        if obj is not None:
            # First listed name wins, matching an ordered identity scan.
            table.setdefault(id(obj), (obj, name))
    return table


def identity_name(obj, names):
    """Name of the triton.language object in `names` that is `obj`, if any."""
    entry = _identity_table(names).get(id(obj))
    return entry[1] if entry is not None and entry[0] is obj else None


_OPERATIONS = ("arange", "program_id", "num_programs", "load", "store", "where", "cast", "full", "zeros", "reshape",
               "broadcast_to", "expand_dims", "static_range", "range", "static_assert", "sum", "min", "max", "gather",
               "cdiv", "minimum", "maximum", "sqrt", "pointer_type", "dot", "make_block_ptr", "advance")


def operation(obj):
    name = identity_name(obj, _OPERATIONS)
    if name is not None:
        return name
    if obj is range:
        return "range"
    if obj is builtins.min:
        return "builtin_min"
    return None


def is_language_builtin(obj):
    """Bounded DSL call identity, without invoking user attribute descriptors."""
    import triton.language as tl

    if inspect.getattr_static(obj, "__triton_builtin__", False) is not True:
        return False
    # A marker on an arbitrary user callable is not sufficient. Only exports
    # from these language namespaces have the DSL value/effect contract.
    return any(value is obj for module in (tl, tl.math) for value in vars(module).values())


STATIC_ARGUMENTS = {
    "make_block_ptr": ((4, 5), ("block_shape", "order")),
    "dot": ((3, 4, 5, 6), ("input_precision", "allow_tf32", "max_num_imprecise_acc", "out_dtype")),
    "arange": ((0, 1), ("start", "end")),
    "program_id": ((0, ), ("axis", )),
    "num_programs": ((0, ), ("axis", )),
    "full": ((0, 2), ("shape", "dtype")),
    "zeros": ((0, 1), ("shape", "dtype")),
    "reshape": ((1, 2), ("shape", "can_reorder")),
    "broadcast_to": ((1, ), ("shape", )),
    "expand_dims": ((1, ), ("axis", )),
    "cast": ((1, 2, 3), ("dtype", "bitcast", "fp_downcast_rounding")),
    "sum": ((1, 2), ("axis", "keep_dims")),
    "min": ((1, 2, 3, 4), ("axis", "keep_dims", "return_indices", "return_indices_tie_break_left")),
    "max": ((1, 2, 3, 4), ("axis", "keep_dims", "return_indices", "return_indices_tie_break_left")),
    "minimum": ((2, ), ("propagate_nan", )),
    "builtin_min": ((), ("propagate_nan", )),
    "maximum": ((2, ), ("propagate_nan", )),
    "gather": ((2, ), ("axis", )),
    "load": ((3, 4, 5, 6, 7), ("cache_modifier", "eviction_policy", "volatile", "boundary_check", "padding_option")),
    "store": ((3, 4, 5), ("cache_modifier", "eviction_policy", "boundary_check")),
}
