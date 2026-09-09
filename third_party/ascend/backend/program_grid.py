# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
"""Versioned host-launch contract for transformed Triton program grids.

``hacc.program_grid_transforms`` is deliberately a small, fail-closed ABI.  A
compiler rule publishes it only after it has retained the original logical
extent used by its tail masks.  The launcher then applies the listed transforms
in order.  This module contains the Python-side validation used both when the
compiler exports the MLIR attr and when the generated launcher consumes the
serialized metadata.
"""

from __future__ import annotations

import json
import operator
from typing import Any, Mapping, Sequence

PROGRAM_GRID_TRANSFORMS_ATTR = "hacc.program_grid_transforms"
PROGRAM_GRID_TRANSFORMS_VERSION = 1

# ``hacc.grid_specialization`` is intentionally independent from the launcher
# transform contract above.  It records the *unmodified* launch grid observed
# by JIT before cache lookup, so a graph rule never has to infer a logical
# extent from a grid which another rule may already have shrunk.
PROGRAM_GRID_SPECIALIZATION_ATTR = "hacc.grid_specialization"
PROGRAM_GRID_SPECIALIZATION_VERSION = 1

# Runtime scalar values are normally intentionally dynamic in TTIR.  Program
# mapping, however, needs a proof for the *particular* launch before it can
# turn one physical program into multiple logical lanes.  This opt-in contract
# carries exact integer argument values from the JIT cache key to
# GraphOptimize, where they are substituted into the entry function before
# the dependence analysis runs.  It is separate from the launch-grid contract:
# the scalar values never reach the generated launcher ABI.
PROGRAM_MAPPING_SCALAR_SPECIALIZATION_ATTR = ("hacc.program_mapping_scalar_specialization")
# Version 1 keyed a value by the TTIR entry argument position.  That position
# is not stable when frontend canonicalization drops a known-contiguous
# runtime parameter, so JIT-generated contracts use version 2: the original
# JIT position stays in the cache key while the retained TTIR block argument
# is selected by its preserved NameLoc spelling.
PROGRAM_MAPPING_SCALAR_SPECIALIZATION_LEGACY_VERSION = 1
PROGRAM_MAPPING_SCALAR_SPECIALIZATION_VERSION = 2

# These values are owned by the append-only GraphOptimize registry (task_0001).
# Keep the bridge's trigger set explicit: unrelated future rules must not make
# a legacy JIT launch evaluate its grid before cache lookup.
INDEPENDENT_AXIS_TENSORIZE_RULE_BIT = 1 << 9
STATIC_PROGRAM_AXIS_FUSION_RULE_BIT = 1 << 10
PERSISTENT_TASK_STRIP_MINING_RULE_BIT = 1 << 11
PROGRAM_MAPPING_RULE_MASK = (INDEPENDENT_AXIS_TENSORIZE_RULE_BIT
                             | STATIC_PROGRAM_AXIS_FUSION_RULE_BIT
                             | PERSISTENT_TASK_STRIP_MINING_RULE_BIT)


class ProgramGridContractError(ValueError):
    """Raised for metadata that is unsafe to pass to a launcher."""


_TOP_LEVEL_KEYS = frozenset(("version", "transforms"))
_TRANSFORM_KEYS = frozenset((
    "order",
    "kind",
    "axis",
    "factor",
    "logical_extent",
    "persistent_coverage",
    "grid_stride_abi_verified",
))
_SPECIALIZATION_KEYS = frozenset(("version", "grid", "rule_mask"))
_SCALAR_SPECIALIZATION_KEYS = frozenset(("version", "arguments"))
_SCALAR_ARGUMENT_V1_KEYS = frozenset(("index", "value"))
_SCALAR_ARGUMENT_V2_KEYS = frozenset(("index", "name", "value"))
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_UINT32_MAX = (1 << 32) - 1


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    # ``bool`` is an ``int`` subclass, but accepting it would make malformed
    # metadata silently change launch dimensions.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramGridContractError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ProgramGridContractError(f"{name} must be >= {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramGridContractError(f"{name} must be a boolean")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgramGridContractError(f"{name} must be a mapping")
    return value


def normalize_program_mapping_rule_mask(raw: Any) -> int:
    """Validate the explicit opt-in bits which require original grid state.

    This is deliberately not the legacy GraphOptimize rule mask.  The legacy
    mask remains backend-managed (and default-off for these bits); exposing a
    separate narrow option prevents an old ``graph_optimize_rule_mask`` value
    from silently changing when JIT evaluates callable grids.
    """
    mask = _integer(raw, "program_mapping_rule_mask", minimum=0)
    unknown = mask & ~PROGRAM_MAPPING_RULE_MASK
    if unknown:
        raise ProgramGridContractError("program_mapping_rule_mask contains unsupported bits "
                                       f"0x{unknown:x}; supported bits are 0x{PROGRAM_MAPPING_RULE_MASK:x}")
    return mask


def program_grid_specialization_enabled(rule_mask: Any) -> bool:
    return bool(normalize_program_mapping_rule_mask(rule_mask))


def _grid_dimension(value: Any, name: str) -> int:
    """Accept indexable runtime grid values while rejecting booleans/tails."""
    if isinstance(value, bool):
        raise ProgramGridContractError(f"{name} must be a positive integer")
    try:
        dimension = operator.index(value)
    except TypeError as error:
        raise ProgramGridContractError(f"{name} must be a positive integer") from error
    if dimension <= 0:
        raise ProgramGridContractError(f"{name} must be >= 1")
    return int(dimension)


def canonicalize_program_grid(grid: Any) -> tuple[int, int, int]:
    """Convert a runtime 1--3D grid to the complete cache-safe triplet."""
    if isinstance(grid, (str, bytes)) or not isinstance(grid, Sequence):
        raise ProgramGridContractError("grid must be a sequence with one to three dimensions")
    if not 1 <= len(grid) <= 3:
        raise ProgramGridContractError("grid must contain one to three dimensions")
    dimensions = [_grid_dimension(value, f"grid[{axis}]") for axis, value in enumerate(grid)]
    return tuple((dimensions + [1, 1, 1])[:3])  # type: ignore[return-value]


def resolve_program_grid_for_specialization(grid: Any, bound_args: Any) -> tuple[int, int, int]:
    """Resolve the original grid under the opt-in pure-callable contract.

    Callable grids are evaluated twice *before* cache lookup.  A changed
    canonical result is rejected rather than compiling one variant and
    launching another with stale tail/loop bounds.  The caller launches the
    returned triplet, so it never evaluates the callable a third time.
    """
    if not callable(grid):
        return canonicalize_program_grid(grid)

    try:
        first = canonicalize_program_grid(grid(bound_args))
        second = canonicalize_program_grid(grid(bound_args))
    except Exception as error:
        raise ProgramGridContractError(
            "program-mapping grid callable could not be resolved before cache lookup") from error
    if first != second:
        raise ProgramGridContractError("program-mapping grid callable is not reproducible before cache lookup")
    return first


def normalize_program_grid_specialization(raw: Any) -> dict[str, Any]:
    """Validate the static original-grid compiler input and canonicalize it."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProgramGridContractError(
                "program_grid_specialization must be valid JSON when encoded as text") from error

    contract = _mapping(raw, "program_grid_specialization")
    if set(contract) != _SPECIALIZATION_KEYS:
        missing = sorted(_SPECIALIZATION_KEYS - set(contract))
        unknown = sorted(set(contract) - _SPECIALIZATION_KEYS)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise ProgramGridContractError("program_grid_specialization has an invalid schema" +
                                       (": " + "; ".join(detail) if detail else ""))

    version = _integer(contract["version"], "program_grid_specialization.version", minimum=1)
    if version != PROGRAM_GRID_SPECIALIZATION_VERSION:
        raise ProgramGridContractError("unsupported program_grid_specialization version "
                                       f"{version}; expected {PROGRAM_GRID_SPECIALIZATION_VERSION}")

    raw_grid = contract["grid"]
    if isinstance(raw_grid, (str, bytes)) or not isinstance(raw_grid, Sequence) or len(raw_grid) != 3:
        raise ProgramGridContractError("program_grid_specialization.grid must contain exactly three dimensions")
    canonical_grid = canonicalize_program_grid(raw_grid)
    rule_mask = normalize_program_mapping_rule_mask(contract["rule_mask"])
    if not rule_mask:
        raise ProgramGridContractError("program_grid_specialization requires at least one program-mapping rule bit")
    return {
        "version": PROGRAM_GRID_SPECIALIZATION_VERSION,
        "grid": list(canonical_grid),
        "rule_mask": rule_mask,
    }


def make_program_grid_specialization(grid: Any, rule_mask: Any) -> dict[str, Any]:
    """Create the one canonical compiler/cache representation for a raw grid."""
    normalized_mask = normalize_program_mapping_rule_mask(rule_mask)
    if not normalized_mask:
        raise ProgramGridContractError(
            "cannot create program_grid_specialization with all program-mapping bits disabled")
    return normalize_program_grid_specialization({
        "version": PROGRAM_GRID_SPECIALIZATION_VERSION,
        "grid": list(canonicalize_program_grid(grid)),
        "rule_mask": normalized_mask,
    })


def canonical_program_grid_specialization_json(raw: Any) -> str:
    """Return the stable compiler-cache representation of an original grid."""
    return json.dumps(normalize_program_grid_specialization(raw), sort_keys=True, separators=(",", ":"))


def normalize_program_mapping_scalar_specialization(raw: Any) -> dict[str, Any]:
    """Validate exact runtime integer arguments used by program mapping.

    Version 1 is the original positional format for already-published static
    fixtures.  Version 2 carries both the original JIT parameter index and
    its name: frontend canonicalization may remove an earlier argument before
    GraphOptimize runs, so selecting a surviving TTIR argument by position can
    silently rewrite a different ABI value.  The name is therefore part of
    the cache-keyed contract and is matched against the retained argument's
    NameLoc in C++.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProgramGridContractError("program_mapping_scalar_specialization must be valid JSON "
                                           "when encoded as text") from error

    contract = _mapping(raw, "program_mapping_scalar_specialization")
    if set(contract) != _SCALAR_SPECIALIZATION_KEYS:
        missing = sorted(_SCALAR_SPECIALIZATION_KEYS - set(contract))
        unknown = sorted(set(contract) - _SCALAR_SPECIALIZATION_KEYS)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise ProgramGridContractError("program_mapping_scalar_specialization has an invalid schema" +
                                       (": " + "; ".join(detail) if detail else ""))

    version = _integer(
        contract["version"],
        "program_mapping_scalar_specialization.version",
        minimum=1,
    )
    if version not in (
            PROGRAM_MAPPING_SCALAR_SPECIALIZATION_LEGACY_VERSION,
            PROGRAM_MAPPING_SCALAR_SPECIALIZATION_VERSION,
    ):
        raise ProgramGridContractError("unsupported program_mapping_scalar_specialization version "
                                       f"{version}; expected "
                                       f"{PROGRAM_MAPPING_SCALAR_SPECIALIZATION_LEGACY_VERSION} or "
                                       f"{PROGRAM_MAPPING_SCALAR_SPECIALIZATION_VERSION}")

    raw_arguments = contract["arguments"]
    if isinstance(raw_arguments, (str, bytes)) or not isinstance(raw_arguments, Sequence):
        raise ProgramGridContractError("program_mapping_scalar_specialization.arguments must be a sequence")
    if not raw_arguments:
        raise ProgramGridContractError("program_mapping_scalar_specialization.arguments must not be empty")

    expected_argument_keys = (_SCALAR_ARGUMENT_V1_KEYS if version
                              == PROGRAM_MAPPING_SCALAR_SPECIALIZATION_LEGACY_VERSION else _SCALAR_ARGUMENT_V2_KEYS)
    arguments: list[dict[str, Any]] = []
    previous_index = -1
    seen_names: set[str] = set()
    for position, raw_argument in enumerate(raw_arguments):
        argument = _mapping(raw_argument, f"arguments[{position}]")
        if set(argument) != expected_argument_keys:
            missing = sorted(expected_argument_keys - set(argument))
            unknown = sorted(set(argument) - expected_argument_keys)
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if unknown:
                detail.append("unknown " + ", ".join(unknown))
            raise ProgramGridContractError(f"arguments[{position}] has an invalid schema" +
                                           (": " + "; ".join(detail) if detail else ""))
        index = _integer(argument["index"], f"arguments[{position}].index", minimum=0)
        value = _integer(argument["value"], f"arguments[{position}].value")
        if index > _UINT32_MAX:
            raise ProgramGridContractError(f"arguments[{position}].index must fit in uint32")
        if value < _INT64_MIN or value > _INT64_MAX:
            raise ProgramGridContractError(f"arguments[{position}].value must fit in int64")
        if index <= previous_index:
            raise ProgramGridContractError("program_mapping_scalar_specialization argument indices must be "
                                           "strictly increasing")
        item: dict[str, Any] = {"index": index, "value": value}
        if version == PROGRAM_MAPPING_SCALAR_SPECIALIZATION_VERSION:
            name = argument["name"]
            if not isinstance(name, str) or not name:
                raise ProgramGridContractError(f"arguments[{position}].name must be a non-empty string")
            if name in seen_names:
                raise ProgramGridContractError("program_mapping_scalar_specialization argument names must be "
                                               "unique")
            item["name"] = name
            seen_names.add(name)
        arguments.append(item)
        previous_index = index

    return {
        "version": version,
        "arguments": arguments,
    }


def make_program_mapping_scalar_specialization(
    arguments: Sequence[tuple[int, int] | tuple[int, str, int]], ) -> dict[str, Any]:
    """Create canonical scalar-specialization metadata from exact JIT values."""
    serialized_arguments: list[dict[str, Any]] = []
    version: int | None = None
    for position, argument in enumerate(arguments):
        if isinstance(argument, (str, bytes)) or not isinstance(argument, Sequence):
            raise ProgramGridContractError(f"arguments[{position}] must be a two- or three-element sequence")
        if len(argument) == 2:
            if version is None:
                version = PROGRAM_MAPPING_SCALAR_SPECIALIZATION_LEGACY_VERSION
            elif version != PROGRAM_MAPPING_SCALAR_SPECIALIZATION_LEGACY_VERSION:
                raise ProgramGridContractError("program_mapping_scalar_specialization cannot mix positional "
                                               "and named arguments")
            index, value = argument
            serialized_arguments.append({"index": index, "value": value})
        elif len(argument) == 3:
            if version is None:
                version = PROGRAM_MAPPING_SCALAR_SPECIALIZATION_VERSION
            elif version != PROGRAM_MAPPING_SCALAR_SPECIALIZATION_VERSION:
                raise ProgramGridContractError("program_mapping_scalar_specialization cannot mix positional "
                                               "and named arguments")
            index, name, value = argument
            serialized_arguments.append({"index": index, "name": name, "value": value})
        else:
            raise ProgramGridContractError(f"arguments[{position}] must contain index/value or index/name/value")
    return normalize_program_mapping_scalar_specialization({
        "version": (PROGRAM_MAPPING_SCALAR_SPECIALIZATION_LEGACY_VERSION if version is None else version),
        "arguments":
        serialized_arguments,
    })


def canonical_program_mapping_scalar_specialization_json(raw: Any) -> str:
    """Return the stable cache representation of scalar specialization."""
    return json.dumps(
        normalize_program_mapping_scalar_specialization(raw),
        sort_keys=True,
        separators=(",", ":"),
    )


def normalize_program_grid_transforms(raw: Any) -> dict[str, Any]:
    """Validate and canonicalize a program-grid-transform contract.

    The canonical representation is JSON-serializable and intentionally has
    no optional fields.  This makes it safe to store in Triton's metadata JSON
    and makes launcher source generation deterministic.

    A persistent transform is allowed only when its corresponding kernel
    rewrite has explicitly verified the grid-stride ABI.  The one-persistent
    transform restriction keeps the physical-core cap unambiguous: it caps the
    transform's target axis after every ceil-div transform has been applied.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProgramGridContractError("program_grid_transforms must be valid JSON when encoded as text") from error

    contract = _mapping(raw, "program_grid_transforms")
    if set(contract) != _TOP_LEVEL_KEYS:
        missing = sorted(_TOP_LEVEL_KEYS - set(contract))
        unknown = sorted(set(contract) - _TOP_LEVEL_KEYS)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise ProgramGridContractError("program_grid_transforms has an invalid top-level schema" +
                                       (": " + "; ".join(detail) if detail else ""))

    version = _integer(contract["version"], "version", minimum=1)
    if version != PROGRAM_GRID_TRANSFORMS_VERSION:
        raise ProgramGridContractError(f"unsupported program_grid_transforms version {version}; "
                                       f"expected {PROGRAM_GRID_TRANSFORMS_VERSION}")

    transforms_raw = contract["transforms"]
    if isinstance(transforms_raw, (str, bytes)) or not isinstance(transforms_raw, Sequence):
        raise ProgramGridContractError("transforms must be a sequence")
    if not transforms_raw:
        raise ProgramGridContractError("transforms must not be empty")

    canonical_transforms: list[dict[str, Any]] = []
    original_extent_by_axis: dict[int, int] = {}
    persistent_count = 0
    for expected_order, transform_raw in enumerate(transforms_raw):
        transform = _mapping(transform_raw, f"transforms[{expected_order}]")
        if set(transform) != _TRANSFORM_KEYS:
            missing = sorted(_TRANSFORM_KEYS - set(transform))
            unknown = sorted(set(transform) - _TRANSFORM_KEYS)
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if unknown:
                detail.append("unknown " + ", ".join(unknown))
            raise ProgramGridContractError(f"transforms[{expected_order}] has an invalid schema" +
                                           (": " + "; ".join(detail) if detail else ""))

        order = _integer(transform["order"], f"transforms[{expected_order}].order", minimum=0)
        if order != expected_order:
            raise ProgramGridContractError("transform order must be the contiguous launch order starting at zero")
        kind = transform["kind"]
        if kind != "ceil_div":
            raise ProgramGridContractError(f"transforms[{expected_order}].kind must be 'ceil_div'")
        axis = _integer(transform["axis"], f"transforms[{expected_order}].axis", minimum=0)
        if axis > 2:
            raise ProgramGridContractError(f"transforms[{expected_order}].axis must be 0, 1, or 2")
        factor = _integer(transform["factor"], f"transforms[{expected_order}].factor", minimum=2)
        logical_extent = _integer(
            transform["logical_extent"],
            f"transforms[{expected_order}].logical_extent",
            minimum=1,
        )
        previous_extent = original_extent_by_axis.setdefault(axis, logical_extent)
        if previous_extent != logical_extent:
            raise ProgramGridContractError("all transforms for one axis must retain the same original logical_extent")

        persistent_coverage = _boolean(
            transform["persistent_coverage"],
            f"transforms[{expected_order}].persistent_coverage",
        )
        grid_stride_abi_verified = _boolean(
            transform["grid_stride_abi_verified"],
            f"transforms[{expected_order}].grid_stride_abi_verified",
        )
        if grid_stride_abi_verified and not persistent_coverage:
            raise ProgramGridContractError("grid_stride_abi_verified is meaningful only for persistent_coverage")
        if persistent_coverage and not grid_stride_abi_verified:
            raise ProgramGridContractError("persistent_coverage requires grid_stride_abi_verified")
        persistent_count += int(persistent_coverage)
        if persistent_count > 1:
            raise ProgramGridContractError("at most one transform may request persistent_coverage")

        canonical_transforms.append({
            "order": order,
            "kind": kind,
            "axis": axis,
            "factor": factor,
            "logical_extent": logical_extent,
            "persistent_coverage": persistent_coverage,
            "grid_stride_abi_verified": grid_stride_abi_verified,
        })

    return {
        "version": PROGRAM_GRID_TRANSFORMS_VERSION,
        "transforms": canonical_transforms,
    }


def canonical_program_grid_transforms_json(raw: Any) -> str:
    """Return the stable cache/metadata representation of a contract."""
    return json.dumps(normalize_program_grid_transforms(raw), sort_keys=True, separators=(",", ":"))


def get_persistent_transform(contract: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the single cap-authorizing transform, if the contract has one."""
    normalized = normalize_program_grid_transforms(contract)
    for transform in normalized["transforms"]:
        if transform["persistent_coverage"]:
            return transform
    return None


def apply_program_grid_transforms(
    grid: Sequence[int],
    contract: Mapping[str, Any],
    *,
    physical_core_count: int | None = None,
) -> tuple[int, int, int]:
    """Reference implementation of the generated launcher arithmetic.

    It is intentionally pure Python so the contract's ceil-div and persistent
    coverage semantics can be unit-tested without an NPU.  The generated C++
    launcher mirrors this order exactly and retains the pre-transform grid for
    each axis before applying the first transform on that axis.
    """
    if len(grid) != 3:
        raise ProgramGridContractError("grid must contain exactly three dimensions")
    launch_grid = [_integer(value, f"grid[{axis}]", minimum=1) for axis, value in enumerate(grid)]
    normalized = normalize_program_grid_transforms(contract)
    checked_original_axes: set[int] = set()
    for transform in normalized["transforms"]:
        axis = transform["axis"]
        if axis not in checked_original_axes:
            checked_original_axes.add(axis)
            if launch_grid[axis] != transform["logical_extent"]:
                raise ProgramGridContractError(f"grid[{axis}] differs from proven logical_extent "
                                               f"{transform['logical_extent']}")
        launch_grid[axis] = (launch_grid[axis] + transform["factor"] - 1) // transform["factor"]

    persistent_transform = get_persistent_transform(normalized)
    if persistent_transform is not None:
        cores = _integer(physical_core_count, "physical_core_count", minimum=1)
        axis = persistent_transform["axis"]
        other_axes = [launch_grid[index] for index in range(3) if index != axis]
        other_axis_programs = other_axes[0] * other_axes[1]
        # A grid-stride ABI on one axis cannot reduce work on the two others.
        # Only cap when that axis can actually enforce the physical bound;
        # otherwise retain the exact transformed launch shape.
        if other_axis_programs <= cores:
            axis_cap = max(1, cores // other_axis_programs)
            launch_grid[axis] = min(launch_grid[axis], axis_cap)
    return tuple(launch_grid)
