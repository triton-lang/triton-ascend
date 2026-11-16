"""Canonical index expressions and explicit logical-axis provenance.

These rewrites describe integer indices, not floating-point computation.
Original expressions are retained by the caller for runtime overflow checks.
"""
from dataclasses import dataclass

ZERO = ("lit", 0)
ONE = ("lit", 1)


@dataclass(frozen=True)
class AxisIdentity:
    tile: int
    extent: tuple


@dataclass(frozen=True)
class GridAxisOrigin:
    tile: int
    dimension: int


@dataclass(frozen=True)
class LoopOrigin:
    axis: AxisIdentity
    site: int
    counts_tiles: bool
    function: str = ""
    line: int = 0


@dataclass(frozen=True)
class OuterLoopOrigin:
    """Unchanged range(pid(axis), extent, num_programs(axis))."""
    extent: tuple
    dimension: int


@dataclass(frozen=True)
class PreservedLoopOrigin:
    """An unchanged scalar range; expressions retain their original arithmetic."""
    start: tuple
    stop: tuple
    step: tuple


def terms(expression, operation):
    if expression[0] == operation:
        return terms(expression[1], operation) + terms(expression[2], operation)
    return [expression]


def integer_literal(expression):
    return expression is not None and expression[0] == "lit" and type(expression[1]) is int


def fold(operation, values, identity):
    values = sorted(values, key=repr)
    if not values:
        return identity
    result = values[0]
    for value in values[1:]:
        result = (operation, result, value)
    return result


def normalize(operation, left, right, *, tile=None):
    if left is None or right is None:
        return None
    if operation == "gt":
        return ("lt", right, left)
    if operation == "sub":
        if right == ZERO:
            return left
        if integer_literal(left) and integer_literal(right):
            return ("lit", left[1] - right[1])
        if integer_literal(right):
            return normalize("add", left, ("lit", -right[1]), tile=tile)
    if operation in ("add", "mul"):
        parts = terms(left, operation) + terms(right, operation)
        scalar = 0 if operation == "add" else 1
        values = []
        for item in parts:
            if integer_literal(item):
                scalar = scalar + item[1] if operation == "add" else scalar * item[1]
            else:
                values.append(item)
        if operation == "mul":
            if scalar == 0:
                return ZERO
            # A tile-count induction multiplied by its own tile is an offset.
            for item in tuple(values):
                if item[0] != "induction" or not item[1].counts_tiles:
                    continue
                tile = ("arg", item[1].axis.tile)
                if tile in values:
                    values.remove(item)
                    values.remove(tile)
                    values.append(("offset", item[1]))
            if scalar != 1:
                values.append(("lit", scalar))
            return fold(operation, values, ONE)
        # Match a complete logical coordinate anywhere in an associative sum.
        if tile is not None:
            for item in tuple(values):
                if item[0] != "mul":
                    continue
                factors = terms(item, "mul")
                programs = [v for v in factors if v[0] == "program_id"]
                lanes = [v for v in values if v[0] == "lane" and v[1] == ("arg", tile)]
                if len(factors) == 2 and len(programs) == 1 and ("arg", tile) in factors and len(lanes) == 1:
                    lane = lanes[0]
                    values.remove(item)
                    values.remove(lane)
                    values.append(("grid_coordinate", GridAxisOrigin(tile, programs[0][1]), lane[-1]))
        for item in tuple(values):
            if item[0] not in ("offset", "induction"):
                continue
            origin = item[1]
            if item[0] == "induction" and origin.counts_tiles:
                continue
            lanes = [v for v in values if v[0] == "lane" and v[1] == ("arg", origin.axis.tile)]
            if len(lanes) == 1:
                lane = lanes[0]
                values.remove(item)
                values.remove(lane)
                values.append(("coordinate", origin.axis, lane[-1]))
        if scalar:
            values.append(("lit", scalar))
        return fold(operation, values, ZERO)
    if operation == "floordiv":
        # Positive N/B are checked by the traversal guard, including N+B-1.
        numerator = terms(left, "add")
        if right in numerator and ("lit", -1) in numerator:
            numerator.remove(right)
            numerator.remove(("lit", -1))
            return ("cdiv", fold("add", numerator, ZERO), right)
    if operation == "and":
        predicates = set(terms(left, "and") + terms(right, "and"))
        predicates.discard(("lit", True))
        return fold("and", predicates, ("lit", True))
    return (operation, left, right)


def schedule_dependent(expression, tile):
    if expression is None:
        return False
    if expression == ("arg", tile) or expression[0] in ("induction", "offset"):
        return True
    if expression[0] in ("coordinate", "grid_coordinate"):
        return False
    return any(schedule_dependent(v, tile) for v in expression[1:] if isinstance(v, tuple))


def has_index(expression):
    if expression is None:
        return False
    return expression[0] in ("induction", "outer_induction", "offset", "lane", "coordinate", "grid_coordinate") or any(
        has_index(v) for v in expression[1:] if isinstance(v, tuple))


def grid_dependencies(expression):
    if expression is None or expression[0] in ("coordinate", "grid_coordinate"):
        return frozenset()
    if expression[0] in ("program_id", "num_programs"):
        return frozenset((expression[1], ))
    return frozenset().union(*(grid_dependencies(v) for v in expression[1:] if isinstance(v, tuple)))
