# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Type and parameter contracts shared by TLA frontends.

No Triton imports or device queries: callers supply normalized type names,
compile-time parameters and, when available, the compilation target. Integer
signedness and floating-point formats must remain distinct (i32/u32, f16/bf16).
Each interface defines the positions in its type signature, e.g. (src, dst)
for a conversion or (element,) for a same-typed elementwise operation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterRule:
    """Allowed values of one parameter for an exact type combination."""

    types: tuple[str, ...]
    parameter: str
    allowed: tuple


@dataclass(frozen=True)
class TypeContract:
    """An optional complete type list and additional combination constraints.

    ``signatures=None`` leaves type coverage to the existing interface checks;
    it is useful when only a specific combination's restriction is established.
    An empty tuple instead means no type signature is supported. All matching
    parameter rules must hold; parameters named by those rules must be supplied.
    """

    signatures: tuple[tuple[str, ...], ...] | None = None
    parameters: tuple[ParameterRule, ...] = ()


def check_type_contract(op, types, contract, *, options=None, target=None):
    """Reject unsupported combinations before an interface constructs IR.

    This checks the declared contract, not unlisted backend capabilities.
    Target-specific callers select a contract using their compilation options;
    ``target`` supplies diagnostic context and never queries the active device.
    """
    types = tuple(types)
    options = {} if options is None else options
    location = f" on {target}" if target else ""
    signature = " -> ".join(types)
    if contract.signatures is not None and types not in contract.signatures:
        allowed = ", ".join(" -> ".join(entry) for entry in contract.signatures) or "none"
        what = "element type" if len(types) == 1 else "type combination"
        raise ValueError(f"{op} does not support {what} {signature}{location}; supported: {allowed}")
    for rule in contract.parameters:
        if types != rule.types:
            continue
        if rule.parameter not in options:
            raise ValueError(f"{op} requires {rule.parameter} to check {signature}{location}")
        value = options[rule.parameter]
        if value not in rule.allowed:
            allowed = ", ".join(str(item) for item in rule.allowed) or "none"
            raise ValueError(f"{op} {signature}{location} does not support {rule.parameter}={value}; "
                             f"supported {rule.parameter}: {allowed}")
