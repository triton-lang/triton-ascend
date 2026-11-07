"""Actual flat JIT-binder argument facts, without pointer/device inspection.

Only the real binder's ordered values and specialization describe positions.
The result is immutable. No names, tensor shape, geometry, data_ptr(), device
contents, caller proof fields or compiler fingerprints are used or generated.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Tuple


class RuntimeArgMappingError(ValueError):

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class RuntimeArgMapping:
    full_to_compact: Tuple[Optional[int], ...]
    compact_to_full: Tuple[int, ...]
    compact_types: Tuple[str, ...]
    scalar_values: Tuple[Tuple[int, int], ...]
    constexpr_full_indices: Tuple[int, ...]

    @property
    def full_arity(self):
        return len(self.full_to_compact)

    @property
    def compact_arity(self):
        return len(self.compact_to_full)

    @property
    def actual_scalar_facts(self):
        return self.scalar_values


def _require(condition, reason):
    if not condition:
        raise RuntimeArgMappingError(reason)


def _flat_value(value):
    # Do not traverse objects or ask for pointer/device/shape attributes.
    return not isinstance(value, (tuple, list, dict, set, frozenset, Mapping))


def _check_override(override, facts, compact_arity):
    if override is None:
        return
    _require(type(override) in (tuple, list), 'runtime_scalar_override_must_be_pair_sequence')
    if not override:
        return
    parsed = []
    seen = set()
    for pair in override:
        _require(
            type(pair) in (tuple, list) and len(pair) == 2, 'runtime_scalar_override_requires_flat_index_value_pairs')
        index, value = pair
        _require(type(index) is int and 0 <= index < compact_arity, 'runtime_scalar_override_invalid_compact_index')
        _require(
            type(value) is int and -(1 << 63) <= value < (1 << 63),
            'runtime_scalar_override_requires_signed_integer_not_bool')
        _require(index not in seen, 'runtime_scalar_override_duplicate_compact_index')
        seen.add(index)
        parsed.append((index, value))
    # Complete values AND binder order must match. Partial/reordered records
    # cannot replace actual facts or silently produce a different cache key.
    _require(tuple(parsed) == facts, 'runtime_scalar_override_does_not_match_actual_arguments')


def map_runtime_arguments(bound_args, specialization, *, scalar_override=None):
    """Map actual flat binder arguments to compact non-constexpr TTIR indices.

    ``specialization`` is the main binder's list/tuple of exact two-item tuples
    ``(type_tag, specialization_attribute)``. This finite API rejects nested
    values/types and descriptor/aggregate expansion. Signed i32/i64 facts use
    actual builtin Python integers and the corresponding signed width. Boolean
    i1/u1 parameters keep their position but never become integer scalar facts.
    A nonempty existing ``auto_simt_runtime_scalar_values`` override must match
    all actual facts exactly; the caller should pass that option as
    ``scalar_override`` and turn this candidate off on the explicit exception.
    """
    _require(isinstance(bound_args, Mapping), 'actual_ordered_binder_mapping_required')
    _require(type(specialization) in (tuple, list), 'actual_flat_specialization_sequence_required')
    items = tuple(bound_args.items())
    _require(len(items) == len(bound_args) == len(specialization), 'binder_specialization_arity_mismatch')
    _require(all(type(name) is str and name for name, _ in items), 'actual_binder_parameter_keys_required')
    _require(len({name for name, _ in items}) == len(items), 'duplicate_binder_parameter_key')
    full_to_compact = []
    compact_to_full = []
    compact_types = []
    facts = []
    constexpr = []
    for full, ((_, value), spec) in enumerate(zip(items, specialization)):
        _require(type(spec) is tuple and len(spec) == 2, 'flat_type_specialization_pair_required')
        tag, attribute = spec
        _require(type(tag) is str and bool(tag) and len(tag) <= 128, 'flat_scalar_or_pointer_type_tag_required')
        _require(attribute is None or type(attribute) in (str, int, bool, float),
                 'nested_specialization_attribute_not_supported')
        _require(_flat_value(value), 'nested_or_aggregate_binder_argument_not_supported')
        if tag == 'constexpr':
            _require(value is None or type(value) in (str, int, bool, float), 'non_scalar_constexpr_not_supported')
            full_to_compact.append(None)
            constexpr.append(full)
            continue
        compact = len(compact_to_full)
        if tag in ('i32', 'i64'):
            _require(type(value) is int, 'signed_integer_specialization_requires_actual_int_not_bool')
            bits = 32 if tag == 'i32' else 64
            _require(-(1 << (bits - 1)) <= value < (1 << (bits - 1)),
                     'actual_integer_outside_signed_specialization_width')
            facts.append((compact, value))
        elif tag in ('i1', 'u1'):
            _require(type(value) is bool, 'boolean_specialization_requires_actual_bool')
        elif tag.startswith('*') and len(tag) > 1 and all(c.isalnum() or c == '_' for c in tag[1:]):
            _require(value is not None and type(value) not in (str, int, bool, float, bytes, bytearray),
                     'pointer_specialization_requires_opaque_actual_argument')
        elif tag in ('fp16', 'bf16', 'fp32', 'fp64', 'fp8e4nv', 'fp8e4b15', 'fp8e5', 'fp8e4b8', 'fp8e5b16'):
            _require(type(value) is float, 'floating_specialization_requires_actual_float')
        else:
            raise RuntimeArgMappingError('unsupported_flat_specialization_type')
        full_to_compact.append(compact)
        compact_to_full.append(full)
        compact_types.append(tag)
    result = RuntimeArgMapping(tuple(full_to_compact), tuple(compact_to_full), tuple(compact_types), tuple(facts),
                               tuple(constexpr))
    _check_override(scalar_override, result.scalar_values, result.compact_arity)
    return result


def try_map_runtime_arguments(bound_args, specialization, *, scalar_override=None):
    """Return ``(immutable_mapping, None)`` or ``(None, explicit_off_reason)``."""
    try:
        return map_runtime_arguments(bound_args, specialization, scalar_override=scalar_override), None
    except RuntimeArgMappingError as error:
        return None, error.reason
