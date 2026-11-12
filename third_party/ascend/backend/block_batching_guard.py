# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Finite original-domain memory proof for the block batching v0 closure.

Pointer roots are not assumed disjoint. All write/read and write/write spans
are checked on every call, including roots unaffected by the batched axis.
No tensor data or allocation capacity is read. Unknown domains select baseline.
"""
from dataclasses import dataclass
import operator

I32_MAX = (1 << 31) - 1
U64_MAX = (1 << 64) - 1


class GuardFailure(ValueError):
    pass


def integer(value):
    if isinstance(value, bool):
        raise GuardFailure("BOOLEAN_INDEX")
    try:
        return operator.index(value)
    except TypeError as error:
        raise GuardFailure("NON_INTEGER_INDEX") from error


def checked(lo, hi, bits=64, unsigned=False):
    minimum = 0 if unsigned else -(1 << (bits - 1))
    maximum = (1 << bits) - 1 if unsigned else (1 << (bits - 1)) - 1
    if lo < minimum or hi > maximum or lo > hi:
        raise GuardFailure("INDEX_OVERFLOW")
    return lo, hi


@dataclass(frozen=True)
class Affine:
    offset: int
    coefficients: tuple = (0, 0, 0, 0)

    @property
    def uniform(self):
        return not any(self.coefficients)

    def add(self, other, sign=1):
        return Affine(self.offset + sign * other.offset,
                      tuple(a + sign * b for a, b in zip(self.coefficients, other.coefficients)))

    def scale(self, scale):
        return Affine(self.offset * scale, tuple(c * scale for c in self.coefficients))


class Domain:

    def __init__(self, arguments, grid):
        self.arguments = arguments
        self.grid = grid
        self.extent = None
        self.checks = []

    def expression(self, node):
        if not isinstance(node, list) or not node:
            raise GuardFailure("INVALID_GUARD_EXPRESSION")
        op = node[0]
        if op == "arg":
            value = integer(self.arguments[node[1]])
            checked(value, value, node[2])
            result = Affine(value)
        elif op == "const":
            checked(node[1], node[1], node[2])
            result = Affine(node[1])
        elif op == "grid":
            result = Affine(self.grid[node[1]])
        elif op == "pid":
            coeff = [0, 0, 0, 0]
            coeff[node[1]] = 1
            result = Affine(0, tuple(coeff))
        elif op == "range":
            extent = node[2] - node[1]
            if extent <= 0 or (self.extent is not None and self.extent != extent):
                raise GuardFailure("INCOMPATIBLE_INDEX_RANGES")
            self.extent = extent
            result = Affine(node[1], (0, 0, 0, 1))
        elif op in ("addi", "subi", "muli"):
            lhs, rhs = self.expression(node[2]), self.expression(node[3])
            if op == "muli":
                if not lhs.uniform and not rhs.uniform:
                    raise GuardFailure("NON_AFFINE_ADDRESS")
                result = rhs.scale(lhs.offset) if lhs.uniform else lhs.scale(rhs.offset)
            else:
                result = lhs.add(rhs, -1 if op == "subi" else 1)
            self.checks.append((result, node[1]))
        elif op in ("extsi", "extui", "trunci"):
            result = self.expression(node[2])
            # The v0 proof permits unsigned extension only for nonnegative
            # values, and truncation only when it does not change the value.
            self.checks.append((result, node[1]))
            if op == "extui":
                self.checks.append((result, -node[1]))
        else:
            raise GuardFailure("UNSUPPORTED_GUARD_EXPRESSION")
        return result

    @staticmethod
    def bounds(affine, counts):
        lo = hi = affine.offset
        for coefficient, count in zip(affine.coefficients, counts):
            delta = coefficient * (count - 1)
            lo += min(0, delta)
            hi += max(0, delta)
        return lo, hi

    def access(self, access):
        address = self.expression(access["offset"])
        counts = [*self.grid, self.extent or 1]
        dimensions = None
        mask = access["mask"]
        if mask is not None:
            if mask[0] != "lt":
                raise GuardFailure("UNSUPPORTED_ACCESS_MASK")
            index, limit = self.expression(mask[1]), self.expression(mask[2])
            counts[3] = self.extent or 1
        # A mask can trim the proof domain only after its source arithmetic
        # is proved nonwrapping on the full original domain. Otherwise a
        # wrapped large PID could satisfy a predicate assumed to be false.
        for expression, bits in self.checks:
            lo, hi = self.bounds(expression, counts)
            checked(lo, hi, abs(bits), unsigned=bits < 0)
        if mask is not None:
            if (not limit.uniform or limit.offset <= 0 or index.offset < 0 or any(c < 0 for c in index.coefficients)
                    or index.coefficients[3] != 1):
                raise GuardFailure("UNSUPPORTED_MASK_DOMAIN")
            bound = limit.offset - index.offset
            if bound <= 0:
                raise GuardFailure("EMPTY_ACCESS_DOMAIN")
            columns = [axis for axis in range(3) if index.coefficients[axis]]
            if len(columns) > 1:
                raise GuardFailure("MULTI_AXIS_MASK")
            if columns:
                column_axis = columns[0]
                block = counts[3]
                if index.coefficients[column_axis] != block:
                    raise GuardFailure("NON_CONTIGUOUS_COLUMN_DOMAIN")
                if address.coefficients[column_axis] != address.coefficients[3] * block:
                    raise GuardFailure("MASK_ADDRESS_DOMAIN_MISMATCH")
                total_columns = min(bound, self.grid[column_axis] * block)
                counts[column_axis] = min(self.grid[column_axis], (bound + block - 1) // block)
                counts[3] = min(block, bound)
                dimensions = [(address.coefficients[a], self.grid[a]) for a in range(3) if a != column_axis]
                dimensions.append((address.coefficients[3], total_columns))
            else:
                counts[3] = min(counts[3], bound)
        if self.extent is not None and address.coefficients[3] != 1:
            raise GuardFailure("NON_UNIT_COLUMN_STRIDE")
        if dimensions is None:
            dimensions = list(zip(address.coefficients, counts))
        for stride, count in dimensions:
            if stride < 0:
                raise GuardFailure("NEGATIVE_STRIDE")
            if count > 1 and stride == 0 and access["write"]:
                raise GuardFailure("OVERLAPPING_BLOCK_WRITES")
        active = sorted((stride, count) for stride, count in dimensions if count > 1 and stride)
        span = 0
        for stride, count in active:
            if access["write"] and stride <= span:
                raise GuardFailure("OVERLAPPING_BLOCK_WRITES")
            span += stride * (count - 1)
        # Read-only zero-stride dimensions are shared accesses, not extra work.
        end = address.offset + span
        checked(address.offset, end)
        return address.offset, end


def check_guard(guard, arguments, grid, factor):
    """Return (accepted, reason), without changing any argument or device state."""
    try:
        if type(guard.get("version")) is not int:
            raise GuardFailure("INVALID_GUARD_SCHEMA")
        if guard["version"] == 3:
            from .block_batching_guard_v3 import check_guard_v3
            return check_guard_v3(guard, arguments, grid, factor)
        if guard["version"] == 2:
            from .block_batching_guard_v2 import check_guard_v2
            return check_guard_v2(guard, arguments, grid, factor)
        if guard.get("version") != 1 or not guard.get("accesses"):
            raise GuardFailure("INVALID_GUARD_SCHEMA")
        grid = tuple(integer(x) for x in grid)
        if len(grid) != 3 or any(x < 0 or x > I32_MAX for x in grid):
            raise GuardFailure("GRID_OUT_OF_RANGE")
        if any(x == 0 for x in grid):
            return False, "EMPTY_GRID"
        axis = guard["axis"]
        if type(axis) is not int or axis not in (0, 1, 2) or type(factor) is not int or factor < 2:
            raise GuardFailure("INVALID_BATCH_REQUEST")
        tasks = 1
        for a, extent in enumerate(grid):
            tasks *= (extent + factor - 1) // factor if a == axis else extent
        if tasks > I32_MAX:
            raise GuardFailure("G0_TASK_COUNT_OVERFLOW")
        ranges = []
        for access in guard["accesses"]:
            lo, hi = Domain(arguments, grid).access(access)
            pointer = arguments[access["root"]]
            pointer = integer(pointer.data_ptr()) if hasattr(pointer, "data_ptr") else integer(pointer)
            width = access["bytes"]
            if width not in (2, 4) or pointer <= 0:
                raise GuardFailure("INVALID_MEMORY_ARGUMENT")
            start, end = pointer + lo * width, pointer + (hi + 1) * width
            checked(start, end, 64, unsigned=True)
            if end <= start:
                raise GuardFailure("INVALID_MEMORY_INTERVAL")
            ranges.append((start, end, access["write"]))
        for i, (start, end, write) in enumerate(ranges):
            for other_start, other_end, other_write in ranges[:i]:
                if (write or other_write) and start < other_end and other_start < end:
                    raise GuardFailure("WRITE_ALIAS")
        return True, "PROVEN_AT_RUNTIME"
    except (GuardFailure, IndexError, KeyError, TypeError, OverflowError) as error:
        return False, str(error)


class RuntimeGuard:
    """Cache a proof only under all its current scalar and pointer dependencies.

    Every invocation reads the current addresses, scalar indices and grid. Keys
    retain integers, never tensor owners; changing an alias, stride or extent
    therefore cannot reuse a proof made for different memory geometry.
    """

    def __init__(self, guard, factor):
        from copy import deepcopy
        from functools import lru_cache
        specification = deepcopy(guard)
        self._roots = frozenset(access["root"] for access in specification["accesses"])
        dependencies = set(self._roots)

        def visit(node):
            if not isinstance(node, list):
                return
            if node and node[0] == "arg":
                dependencies.add(node[1])
            else:
                for operand in node:
                    visit(operand)

        for access in specification["accesses"]:
            visit(access["offset"])
            visit(access["mask"])
            for expression in access.get("pointer_checks", ()):
                visit(expression)
            for loop in access.get("loops", ()):
                for field in ("lower", "upper", "step"):
                    visit(loop[field])
        for layout in specification.get("layout_checks", ()):
            for field in ("positive", "checks", "pointer_checks"):
                for expression in layout[field]:
                    visit(expression)
        self._indices = indices = tuple(sorted(dependencies))

        @lru_cache(maxsize=64)
        def prove(values, grid):
            return check_guard(specification, dict(zip(indices, values)), grid, factor)

        self._prove = prove

    def __call__(self, arguments, grid):
        try:
            grid = tuple(integer(x) for x in grid)
            if len(grid) != 3 or any(x < 0 or x > I32_MAX for x in grid):
                raise GuardFailure("GRID_OUT_OF_RANGE")
            if any(x == 0 for x in grid):
                return False, "EMPTY_GRID"
            values = []
            for index in self._indices:
                value = arguments[index]
                if index in self._roots and hasattr(value, "data_ptr"):
                    value = value.data_ptr()
                values.append(integer(value))
            return self._prove(tuple(values), grid)
        except (GuardFailure, IndexError, KeyError, TypeError, OverflowError) as error:
            return False, str(error)
