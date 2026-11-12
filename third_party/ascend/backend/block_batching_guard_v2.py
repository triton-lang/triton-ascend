# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Byte-domain proof for static tensor-axis-aware access summaries."""
from dataclasses import dataclass

from .block_batching_guard import GuardFailure, I32_MAX, checked, integer


@dataclass(frozen=True)
class AffineND:
    offset: int
    coefficients: tuple

    @property
    def uniform(self):
        return not any(self.coefficients)

    def add(self, other, sign=1):
        return AffineND(self.offset + sign * other.offset,
                        tuple(a + sign * b for a, b in zip(self.coefficients, other.coefficients)))

    def scale(self, value):
        return AffineND(self.offset * value, tuple(c * value for c in self.coefficients))


class ByteDomain:

    def __init__(self, arguments, grid, access):
        self.arguments, self.grid, self.access = arguments, grid, access
        self.shape = tuple(integer(n) for n in access.get("shape", ()))
        if len(self.shape) > 8 or any(n <= 0 for n in self.shape):
            raise GuardFailure("UNSUPPORTED_ACCESS_RANK")
        self.checks = []
        self.budget = 1024
        self.loop_axes = {}
        # Program axes that a loop reparameterized away; see the lower-bound
        # handling below.
        self.absorbed = {}
        loops = access.get("loops", ())
        if not isinstance(loops, (tuple, list)) or len(loops) > 8:
            raise GuardFailure("INVALID_LOOP_DOMAIN")
        counts = []
        for loop in loops:
            identity = integer(loop["id"])
            if identity < 0 or identity in self.loop_axes:
                raise GuardFailure("INVALID_LOOP_DOMAIN")
            # Bounds may mention a quotient/remainder of an unmerged program
            # axis. Register its split BEFORE evaluating those expressions.
            # No nested IV is invented: unsupported outer captures still fail.
            domain = ByteDomain(arguments, grid, {
                "shape": [],
                "pointer_checks": [loop[name] for name in ("lower", "upper", "step")],
            })
            # A program-axis loop starts at pid*R by definition, so its lower
            # bound is not a scalar. The VV absorbed-axis rule below still
            # requires a uniform upper bound; ordinary CV loops may instead
            # use an affine upper envelope over all original programs.
            lower = domain.expression(loop["lower"])
            upper_expr = domain.expression(loop["upper"])
            step_expr = domain.expression(loop["step"])
            if not step_expr.uniform:
                raise GuardFailure("NON_UNIFORM_LOOP_DOMAIN")
            for value, bits in domain.checks:
                checked(*domain.bounds(value, domain.full), bits)
            upper_lo, upper = domain.bounds(upper_expr, domain.full)
            step = step_expr.offset
            bits = self.bits(loop["bits"])
            if step <= 0:
                raise GuardFailure("UNPROVEN_POSITIVE_LOOP_STEP")
            stride, absorbed = step, None
            if not lower.uniform:
                # lower = p*R with step = G*R. Then p runs over every residue
                # class, so the induction variable takes exactly the values
                # off + R*m below the upper bound -- one axis of stride R, not
                # a program axis crossed with a trip axis. Modelling it as the
                # cross product would admit (p, k) pairs that never happen and
                # push the proved address range a whole step past the buffer.
                axes = [a for a, c in enumerate(lower.coefficients) if c]
                rows = lower.coefficients[axes[0]] if len(axes) == 1 else 0
                if (not upper_expr.uniform or rows <= 0 or axes[0] >= 3 or step != grid[axes[0]] * rows):
                    raise GuardFailure("NON_UNIFORM_LOOP_DOMAIN")
                stride, absorbed = rows, axes[0]
            checked(min(lower.offset, upper_lo), max(lower.offset, upper), bits)
            # With an affine upper bound this box is a SUPERSET of every
            # actual (program, iv) pair. Do not use it to erase masks or infer
            # independence. The byte interval, write injection and alias
            # checks below must all succeed on the larger box.
            count = max(0, (upper - lower.offset + stride - 1) // stride)
            # Keep the original induction increment representable as well.
            checked(lower.offset, lower.offset + count * stride, bits)
            position = 3 + len(self.shape) + len(counts)
            self.loop_axes[identity] = (position, lower.offset, stride, bits)
            if absorbed is not None:
                self.absorbed[absorbed] = identity
            counts.append(count)
        self.loop_counts = tuple(counts)
        self.program_splits = {}
        splits = {}
        budget = 1024

        def collect(node):
            nonlocal budget
            budget -= 1
            if budget < 0:
                raise GuardFailure("ANALYSIS_BUDGET_EXCEEDED")
            if not isinstance(node, list) or not node:
                return
            if node[0] in ("pid_div", "pid_rem"):
                if len(node) != 4 or self.bits(node[3]) != 32:
                    raise GuardFailure("INVALID_PROGRAM_SPLIT")
                axis, divisor = integer(node[1]), integer(node[2])
                if not 0 <= axis < 3 or not 0 < divisor <= I32_MAX:
                    raise GuardFailure("INVALID_PROGRAM_SPLIT")
                if axis in splits and splits[axis] != divisor:
                    raise GuardFailure("INCOMPATIBLE_PROGRAM_SPLITS")
                splits[axis] = divisor
                return
            for operand in node:
                if isinstance(operand, list):
                    collect(operand)

        loop_bounds = [loop[name] for loop in loops for name in ("lower", "upper", "step")]
        for node in (access.get("offset"), access.get("mask"), *access.get("pointer_checks", ()), *loop_bounds):
            collect(node)
        full = [*grid, *self.shape, *counts]
        for axis, divisor in sorted(splits.items()):
            if grid[axis] % divisor:
                raise GuardFailure("NON_RECTANGULAR_PROGRAM_SPLIT")
            self.program_splits[axis] = (divisor, len(full))
            full[axis] //= divisor
            full.append(divisor)
        self.full = tuple(full)

    def constant(self, value):
        return AffineND(value, (0, ) * len(self.full))

    @staticmethod
    def bounds(value, counts):
        lo = hi = value.offset
        for coefficient, count in zip(value.coefficients, counts):
            delta = coefficient * (count - 1)
            lo += min(0, delta)
            hi += max(0, delta)
        return lo, hi

    @staticmethod
    def bits(value):
        if type(value) is not int or value not in (1, 32, 64):
            raise GuardFailure("INVALID_INDEX_WIDTH")
        return value

    def expression(self, node):
        self.budget -= 1
        if self.budget < 0:
            raise GuardFailure("ANALYSIS_BUDGET_EXCEEDED")
        if not isinstance(node, list) or not node:
            raise GuardFailure("INVALID_GUARD_EXPRESSION")
        name = node[0]
        if name in ("arg", "const"):
            value = integer(self.arguments[node[1]]) if name == "arg" else integer(node[1])
            checked(value, value, self.bits(node[2]))
            return self.constant(value)
        if name == "grid":
            axis = integer(node[1])
            if not 0 <= axis < 3:
                raise GuardFailure("INVALID_PROGRAM_AXIS")
            return self.constant(self.grid[axis])
        if name == "iv":
            identity = integer(node[1])
            if identity not in self.loop_axes:
                raise GuardFailure("UNSUPPORTED_LOOP_CAPTURE")
            axis, lower, step, bits = self.loop_axes[identity]
            if self.bits(node[2]) != bits:
                raise GuardFailure("INVALID_LOOP_DOMAIN")
            coefficients = [0] * len(self.full)
            coefficients[axis] = step
            return AffineND(lower, tuple(coefficients))
        if name in ("pid", "coord"):
            axis = integer(node[1]) + (3 if name == "coord" else 0)
            if name == "pid" and not 0 <= axis < 3:
                raise GuardFailure("INVALID_PROGRAM_AXIS")
            if name == "coord" and (not 3 <= axis < 3 + len(self.shape) or self.full[axis] != integer(node[2])):
                raise GuardFailure("INVALID_LOCAL_AXIS")
            coefficients = [0] * len(self.full)
            if name == "pid" and axis in self.program_splits:
                divisor, remainder_axis = self.program_splits[axis]
                coefficients[axis] = divisor
                coefficients[remainder_axis] = 1
            else:
                coefficients[axis] = 1
            return AffineND(0, tuple(coefficients))
        if name in ("pid_div", "pid_rem"):
            axis, divisor = integer(node[1]), integer(node[2])
            if axis not in self.program_splits or self.program_splits[axis][0] != divisor:
                raise GuardFailure("INVALID_PROGRAM_SPLIT")
            coefficients = [0] * len(self.full)
            coefficients[axis if name == "pid_div" else self.program_splits[axis][1]] = 1
            return AffineND(0, tuple(coefficients))
        if name in ("minsi", "maxsi"):
            lhs, rhs = self.expression(node[2]), self.expression(node[3])
            if not lhs.uniform or not rhs.uniform:
                raise GuardFailure("NON_UNIFORM_SCALAR_BOUND")
            result = self.constant((min if name == "minsi" else max)(lhs.offset, rhs.offset))
            self.checks.append((result, self.bits(node[1])))
            return result
        if name in ("addi", "subi", "muli"):
            lhs, rhs = self.expression(node[2]), self.expression(node[3])
            if name == "muli":
                if not lhs.uniform and not rhs.uniform:
                    raise GuardFailure("NON_AFFINE_ADDRESS")
                result = rhs.scale(lhs.offset) if lhs.uniform else lhs.scale(rhs.offset)
            else:
                result = lhs.add(rhs, -1 if name == "subi" else 1)
            self.checks.append((result, self.bits(node[1])))
            return result
        if name in ("extsi", "extui", "trunci"):
            result = self.expression(node[2])
            if name == "extui":
                source_bits = self.bits(node[3])
                lo, hi = self.bounds(result, self.full)
                if lo < 0 <= hi:
                    raise GuardFailure("NON_AFFINE_UNSIGNED_EXTENSION")
                if hi < 0:
                    result = result.add(self.constant(1 << source_bits))
            self.checks.append((result, self.bits(node[1])))
            return result
        raise GuardFailure("UNSUPPORTED_GUARD_EXPRESSION")

    def predicates(self, mask):
        if mask is None or mask == ["unknown"]:
            return []
        if mask == ["true"]:
            return []
        if mask == ["false"]:
            return [("false", None, None, 0)]
        if mask[0] == "and":
            return self.predicates(mask[1]) + self.predicates(mask[2])
        if mask[0] in ("slt", "ult", "eq"):
            return [(mask[0], self.expression(mask[1]), self.expression(mask[2]), self.bits(mask[3]))]
        # No use of an unmodeled predicate to shrink the original domain.
        return []

    def require_full_mask(self, mask):
        """Prove every original predicate true before any rank-three load.

        Unlike predicates(), this proof cannot ignore an unknown expression.
        Evaluate bounds over the original full grid and local tensor box;
        never use a mask-trimmed domain to justify deleting that same mask.
        """
        if mask is None or mask == ["true"]:
            return
        if not isinstance(mask, list) or not mask:
            raise GuardFailure("CV_FULL_TILE_MASK_NOT_PROVEN")
        if mask[0] == "and" and len(mask) == 3:
            self.require_full_mask(mask[1])
            self.require_full_mask(mask[2])
            return
        if mask[0] not in ("slt", "ult", "eq") or len(mask) != 4:
            raise GuardFailure("CV_FULL_TILE_MASK_NOT_PROVEN")
        lhs, rhs = self.expression(mask[1]), self.expression(mask[2])
        bits = self.bits(mask[3])
        for value, width in self.checks:
            checked(*self.bounds(value, self.full), width)
        if mask[0] == "ult":
            for value in (lhs, rhs):
                checked(*self.bounds(value, self.full), bits, unsigned=True)
        lo, hi = self.bounds(lhs.add(rhs, -1), self.full)
        if (mask[0] == "eq" and (lo != 0 or hi != 0)) or (mask[0] != "eq" and hi >= 0):
            raise GuardFailure("CV_FULL_TILE_MASK_NOT_PROVEN")

    def require_same_fractal_bounds(self):
        """Allow tails only inside the final 32-byte padding quantum per axis.

        Evaluate prefixes over the original program grid. Retained masks still
        control GM access. A matching NZ allocation footprint is insufficient:
        f32 rows 23/32 can occupy two NZ row blocks while batch DMA still leaves
        rows 24:32 undefined. Require the stricter 32-byte extent in BOTH axes,
        including before/after a normalized transpose. Never reduce original K.
        """
        mask = self.access.get("mask")
        if self.access["write"] or mask is None or mask == ["true"]:
            return
        reason = "CV_BATCH_PADDING_NOT_PROVEN"
        width = self.access["bytes"]
        if len(self.shape) != 2 or type(width) is not int or width not in (2, 4):
            try:
                self.require_full_mask(mask)
            except GuardFailure:
                raise GuardFailure(reason)
            return
        address = self.expression(self.access["offset"])
        strides = address.coefficients[3:5]
        if any(s <= 0 for s in strides) or sum(s == width for s in strides) != 1:
            raise GuardFailure(reason)
        # This also implies the 16-row NZ allocation extent is unchanged.
        # Keep both axes constrained; the physical layout alone is not enough
        # to prove the zero-filled reduction lanes of a transposed operand.
        quantum = (min(16, 32 // width), ) * 2
        active = list(self.shape)

        def prefix(node):
            if node is None or node == ["true"]:
                return
            if not isinstance(node, list) or not node:
                raise GuardFailure(reason)
            if node[0] == "and" and len(node) == 3:
                prefix(node[1])
                prefix(node[2])
                return
            if node[0] not in ("slt", "ult") or len(node) != 4:
                raise GuardFailure(reason)
            lhs, rhs = self.expression(node[1]), self.expression(node[2])
            if node[0] == "slt" and lhs.uniform and lhs.offset == -1:
                # x >= 0 canonicalized to -1 < x. Prove it over the untrimmed
                # box; an active lower tail would violate the padding proof.
                try:
                    self.require_full_mask(node)
                except GuardFailure:
                    raise GuardFailure(reason)
                return
            if node[0] == "ult":
                for value in (lhs, rhs):
                    checked(*self.bounds(value, self.full), self.bits(node[3]), unsigned=True)
            delta = lhs.add(rhs, -1)
            local = delta.coefficients[3:5]
            axes = [axis for axis, stride in enumerate(local) if stride]
            if len(axes) != 1 or local[axes[0]] != 1:
                raise GuardFailure(reason)
            counts = list(self.full)
            counts[3:5] = [1, 1]
            _, highest_offset = self.bounds(delta, counts)
            axis = axes[0]
            active[axis] = max(0, min(active[axis], -highest_offset))

        prefix(mask)
        for value, bits in self.checks:
            checked(*self.bounds(value, self.full), bits)
        if any(n == 0 or (n + q - 1) // q != (full + q - 1) // q for n, full, q in zip(active, self.shape, quantum)):
            raise GuardFailure(reason)

    def require_full_axes(self, axes):
        if axes == []:
            return  # Auxiliary rank-one accesses are not dot reduction inputs.
        if len(self.shape) != 2 or not all(type(axis) is int and axis in (0, 1) for axis in axes):
            raise GuardFailure("CV_BATCH_DOT_CONTRACT_NOT_PROVEN")
        mask = self.access.get("mask")
        if mask is None or mask == ["true"]:
            return
        reason = "CV_BATCH_REDUCTION_MASK_NOT_PROVEN"

        def prove(node):
            if node is None or node == ["true"]:
                return
            if not isinstance(node, list) or not node:
                raise GuardFailure(reason)
            if node[0] == "and" and len(node) == 3:
                prove(node[1])
                prove(node[2])
                return
            if node[0] not in ("slt", "ult") or len(node) != 4:
                raise GuardFailure(reason)
            lhs, rhs = self.expression(node[1]), self.expression(node[2])
            if node[0] == "slt" and lhs.uniform and lhs.offset == -1:
                try:
                    self.require_full_mask(node)
                except GuardFailure:
                    raise GuardFailure(reason)
                return
            delta = lhs.add(rhs, -1)
            local = delta.coefficients[3:5]
            touched = [axis for axis, stride in enumerate(local) if stride]
            if len(touched) != 1 or local[touched[0]] != 1:
                raise GuardFailure(reason)
            if node[0] == "ult":
                for value in (lhs, rhs):
                    checked(*self.bounds(value, self.full), self.bits(node[3]), unsigned=True)
            if touched[0] in axes and self.bounds(delta, self.full)[1] >= 0:
                raise GuardFailure(reason)

        prove(mask)
        for value, bits in self.checks:
            checked(*self.bounds(value, self.full), bits)

    def interval(self, pointer):
        if any(count == 0 for count in self.loop_counts):
            return None
        width = self.access["bytes"]
        volatile = self.access.get("volatile", False)
        if type(volatile) is not bool:
            raise GuardFailure("INVALID_MEMORY_ARGUMENT")
        exclusive = self.access["write"] or volatile
        overlap_reason = "OVERLAPPING_VOLATILE_READS" if volatile else "OVERLAPPING_BLOCK_WRITES"
        if type(width) is not int or width not in (1, 2, 4, 8):
            raise GuardFailure("INVALID_MEMORY_ARGUMENT")
        address = self.expression(self.access["offset"])
        predicates = self.predicates(self.access.get("mask"))
        steps = [self.expression(node) for node in self.access.get("pointer_checks", ())]
        # Validate all source arithmetic before a mask is allowed to trim it.
        for expression, bits in self.checks:
            lo, hi = self.bounds(expression, self.full)
            checked(lo, hi, bits)
        for step in steps:
            lo, hi = self.bounds(step, self.full)
            checked(pointer + lo, pointer + hi, 64, unsigned=True)
        # Unit column stride is a layout preference, not a requirement of a
        # byte-domain proof. Positive strided/gapped accesses are enclosed by
        # the same interval below. For writes/volatile reads the sorted-stride
        # test proves disjoint element-byte sets (stride > prior span); gaps
        # stay in the conservative alias interval. Retain all integer/pointer
        # overflow, negative-stride and cross-access alias checks.
        counts = list(self.full)
        # An absorbed program axis has been folded into its loop's single
        # stride. Using it again in the same address would mean the address
        # depends on p and m independently, and the reparameterization no
        # longer describes what is touched.
        for program in self.absorbed:
            if address.coefficients[program]:
                raise GuardFailure("LOOP_AXIS_REUSES_PROGRAM")
            counts[program] = 1
        merged = {}
        for predicate, index, bound, bits in predicates:
            if predicate == "false":
                return None
            if not bound.uniform:
                continue
            if predicate == "ult":
                lo, hi = self.bounds(index, self.full)
                if lo < 0:
                    continue
                limit = bound.offset % (1 << bits)
            else:
                limit = bound.offset
            if index.uniform:
                truth = index.offset == limit if predicate == "eq" else index.offset < limit
                if not truth:
                    return None
                continue
            nonzero = [axis for axis, c in enumerate(index.coefficients) if c]
            if index.offset < 0 or any(c < 0 for c in index.coefficients):
                continue
            remaining = limit - index.offset
            if predicate == "eq":
                # A nonnegative affine sum equals zero only at zero on each
                # participating axis. Other equalities retain the full box.
                if remaining == 0:
                    for axis in nonzero:
                        counts[axis] = min(counts[axis], 1)
                continue
            if remaining <= 0:
                return None
            if len(nonzero) == 1 and index.coefficients[nonzero[0]] == 1:
                axis = nonzero[0]
                counts[axis] = min(counts[axis], remaining)
            elif len(nonzero) == 2:
                # A uniform loop tile has the same exact flattening as a
                # program tile: j = iteration * tile + local. When the
                # predicate/address coefficients agree, j covers precisely
                # [0, min(limit, trip_count * tile)); padded K lanes are not
                # part of the memory footprint. Keep all arithmetic checks
                # above, and do not approximate non-unit/gapped recurrences.
                loop_positions = {value[0] for value in self.loop_axes.values()}
                grids = [axis for axis in nonzero if axis < 3 or axis in loop_positions]
                locals_ = [axis for axis in nonzero if 3 <= axis < 3 + len(self.shape)]
                if len(grids) != 1 or len(locals_) != 1:
                    continue
                g, local = grids[0], locals_[0]
                block = self.full[local]
                if (index.coefficients[g] != block or index.coefficients[local] != 1
                        or address.coefficients[g] != address.coefficients[local] * block):
                    continue
                total = min(remaining, counts[g] * block)
                key = (g, local)
                merged[key] = min(merged.get(key, total), total)
                counts[g] = min(counts[g], (total + block - 1) // block)
                counts[local] = min(counts[local], total)
        covered = set()
        dimensions = []
        for (g, local), count in merged.items():
            if g in covered or local in covered:
                continue
            covered.update((g, local))
            dimensions.append((address.coefficients[local], count))
        dimensions += [(stride, counts[axis])
                       for axis, stride in enumerate(address.coefficients)
                       if axis not in covered]
        if any(count <= 0 for _, count in dimensions):
            return None
        for stride, count in dimensions:
            if stride < 0:
                raise GuardFailure("NEGATIVE_STRIDE")
            if exclusive and count > 1 and stride == 0:
                raise GuardFailure(overlap_reason)
        span = width - 1
        for stride, count in sorted((s, n) for s, n in dimensions if s and n > 1):
            if exclusive and stride <= span:
                raise GuardFailure(overlap_reason)
            span += stride * (count - 1)
        start, end = pointer + address.offset, pointer + address.offset + span + 1
        checked(start, end, 64, unsigned=True)
        if start >= end:
            raise GuardFailure("INVALID_MEMORY_INTERVAL")
        return start, end, self.access["write"]


def check_layout_guards(guard, arguments, grid, factor):
    """Check emitted wide arithmetic before masks trim any padded batch lane.

    Program coordinates use the physical grid; `grid` leaves still denote the
    original hidden extents. The original access guard remains responsible for
    effects/aliases and source-width arithmetic.
    """
    layouts = guard.get("layout_checks", [])
    if not isinstance(layouts, list) or len(layouts) > min(1024, len(guard["accesses"])):
        raise GuardFailure("INVALID_LAYOUT_GUARD")
    fields = {
        "version", "access_id", "axis", "factor", "root", "bytes", "shape", "loops", "positive", "checks",
        "pointer_checks"
    }
    seen = set()
    physical = list(grid)
    physical[guard["axis"]] = (grid[guard["axis"]] + factor - 1) // factor
    for layout in layouts:
        if (not isinstance(layout, dict) or set(layout) != fields or type(layout["version"]) is not int
                or layout["version"] != 1 or type(layout["axis"]) is not int or layout["axis"] != guard["axis"]
                or type(layout["factor"]) is not int or layout["factor"] != factor):
            raise GuardFailure("INVALID_LAYOUT_GUARD")
        identity = layout["access_id"]
        root = layout["root"]
        if (type(identity) is not int or not 0 <= identity < len(guard["accesses"]) or identity in seen
                or type(root) is not int or root < 0 or root != guard["accesses"][identity]["root"]
                or type(layout["bytes"]) is not int or layout["bytes"] != guard["accesses"][identity]["bytes"]):
            raise GuardFailure("INVALID_LAYOUT_GUARD")
        seen.add(identity)
        for key, minimum, maximum in (("positive", 1, 1), ("checks", 1, 16), ("pointer_checks", 2, 2)):
            if not isinstance(layout[key], list) or not minimum <= len(layout[key]) <= maximum:
                raise GuardFailure("INVALID_LAYOUT_GUARD")
        if not isinstance(layout["shape"], list) or not layout["shape"]:
            raise GuardFailure("INVALID_LAYOUT_GUARD")
        # The bridged address carries the in-loop offset, so the loop domain
        # belongs here too; it is read by the same constructor as the original
        # access guard. Program coordinates then range over the physical grid.
        domain = ByteDomain(arguments, grid, {"shape": layout["shape"], "loops": layout["loops"]})
        domain.full = (*physical, *domain.shape, *domain.loop_counts)
        for expr in layout["positive"]:
            value = domain.expression(expr)
            if not value.uniform or value.offset <= 0:
                raise GuardFailure("NON_POSITIVE_LAYOUT_STRIDE")
        for expr in layout["checks"]:
            domain.expression(expr)
        pointers = [domain.expression(expr) for expr in layout["pointer_checks"]]
        for value, bits in domain.checks:
            checked(*domain.bounds(value, domain.full), bits)
        value = arguments[root]
        pointer = integer(value.data_ptr()) if hasattr(value, "data_ptr") else integer(value)
        for expr in pointers:
            lo, hi = domain.bounds(expr, domain.full)
            checked(pointer + lo, pointer + hi + layout["bytes"] - 1, 64, unsigned=True)


def static_cv_mask_refusal(guard):
    """Reject a provably ineligible constant prefix before backend compilation.

    This only rejects; it never authorizes a launch or substitutes fabricated
    values for runtime arguments/grid. Unknown expressions remain the existing
    runtime guard's responsibility. In particular JIT specialization of scalar
    one can otherwise reach an unsupported ND2NZ shape before that guard runs.
    """
    if guard.get("require_same_fractal_bounds") is not True:
        return None
    inputs = guard.get("cv_mask_contract", {}).get("inputs", [])
    reduction_axes = {item["access"]: item["axes"][1 if item["operand"] == 0 else 0] for item in inputs}
    for aid, access in enumerate(guard.get("accesses", [])):
        shape, width = access.get("shape", []), access.get("bytes")
        if access.get("write") or len(shape) != 2 or width not in (2, 4):
            continue
        active = list(shape)
        pending = [access.get("mask")]
        budget = 256

        def static_expression(node):
            nonlocal budget
            budget -= 1
            if budget < 0 or not isinstance(node, list) or not node:
                return False
            if node[0] not in ("const", "coord", "addi", "subi", "muli", "extsi", "extui", "trunci"):
                return False
            return all(static_expression(x) for x in node[1:] if isinstance(x, list))

        while pending and budget > 0:
            mask = pending.pop()
            budget -= 1
            if not isinstance(mask, list) or not mask:
                continue
            if mask[0] == "and" and len(mask) == 3:
                pending.extend(mask[1:])
                continue
            if (len(mask) != 4 or mask[0] not in ("slt", "ult") or not static_expression(mask[1])
                    or not static_expression(mask[2])):
                continue
            try:
                domain = ByteDomain([], (1, 1, 1), {"shape": shape})
                lhs, rhs = domain.expression(mask[1]), domain.expression(mask[2])
                delta = lhs.add(rhs, -1)
                for value, bits in domain.checks:
                    checked(*domain.bounds(value, domain.full), bits)
                if mask[0] == "ult":
                    for value in (lhs, rhs):
                        checked(*domain.bounds(value, domain.full), domain.bits(mask[3]), unsigned=True)
                local = delta.coefficients[3:5]
                axes = [axis for axis, coefficient in enumerate(local) if coefficient]
                if len(axes) == 1 and local[axes[0]] == 1 and not any(delta.coefficients[:3]):
                    axis = axes[0]
                    active[axis] = max(0, min(active[axis], -delta.offset))
            except (GuardFailure, KeyError, IndexError, TypeError, OverflowError):
                continue
        quantum = min(16, 32 // width)
        if any(n == 0 or (n + quantum - 1) // quantum != (full + quantum - 1) // quantum
               for n, full in zip(active, shape)):
            return "CV_BATCH_PADDING_NOT_PROVEN"
        reduction = reduction_axes.get(aid)
        if reduction is not None and active[reduction] < shape[reduction]:
            return "CV_BATCH_REDUCTION_MASK_NOT_PROVEN"
    return None


def check_guard_v2(guard, arguments, grid, factor):
    try:
        if type(guard.get("version")) is not int or guard["version"] != 2 or not guard.get("accesses"):
            raise GuardFailure("INVALID_GUARD_SCHEMA")
        require_full_masks = guard.get("require_full_masks", False)
        require_same_fractal_bounds = guard.get("require_same_fractal_bounds", False)
        if type(require_full_masks) is not bool or type(require_same_fractal_bounds) is not bool:
            raise GuardFailure("INVALID_GUARD_SCHEMA")
        grid = tuple(integer(n) for n in grid)
        if len(grid) != 3 or any(n < 0 or n > I32_MAX for n in grid):
            raise GuardFailure("GRID_OUT_OF_RANGE")
        if any(n == 0 for n in grid):
            return False, "EMPTY_GRID"
        axis = guard["axis"]
        if type(axis) is not int or axis not in (0, 1, 2) or type(factor) is not int or factor < 2:
            raise GuardFailure("INVALID_BATCH_REQUEST")
        tasks = 1
        for d, n in enumerate(grid):
            tasks *= (n + factor - 1) // factor if d == axis else n
        if tasks > I32_MAX:
            raise GuardFailure("G0_TASK_COUNT_OVERFLOW")
        ranges = []
        cv_contract = guard.get("cv_mask_contract") if require_same_fractal_bounds else None
        if require_same_fractal_bounds:
            if (type(cv_contract) is not dict or cv_contract.get("version") != 1
                    or type(cv_contract.get("inputs")) is not list or len(cv_contract["inputs"]) != 2
                    or type(cv_contract.get("outputs")) is not list or not cv_contract["outputs"]):
                raise GuardFailure("CV_BATCH_DOT_CONTRACT_NOT_PROVEN")
            if any(type(x) is not dict for x in cv_contract["inputs"]):
                raise GuardFailure("CV_BATCH_DOT_CONTRACT_NOT_PROVEN")
            if {x.get("operand") for x in cv_contract["inputs"]} != {0, 1}:
                raise GuardFailure("CV_BATCH_DOT_CONTRACT_NOT_PROVEN")
            input_axes = {}
            for item in cv_contract["inputs"]:
                aid, operand, axes = item.get("access"), item["operand"], item.get("axes")
                if (type(aid) is not int or not 0 <= aid < len(guard["accesses"]) or type(axes) is not list
                        or sorted(axes) != [0, 1] or guard["accesses"][aid]["write"]):
                    raise GuardFailure("CV_BATCH_DOT_CONTRACT_NOT_PROVEN")
                # One load can legally supply both dot operands (A @ A.T).
                # Keep every reduction-axis obligation in original load
                # coordinates; overwriting the first operand's axis could
                # incorrectly admit a masked K dimension on that operand.
                reduction_axis = axes[1] if operand == 0 else axes[0]
                input_axes.setdefault(aid, [])
                if reduction_axis not in input_axes[aid]:
                    input_axes[aid].append(reduction_axis)
            if (any(
                    type(aid) is not int or not 0 <= aid < len(guard["accesses"]) or not guard["accesses"][aid]["write"]
                    for aid in cv_contract["outputs"])):
                raise GuardFailure("CV_BATCH_DOT_CONTRACT_NOT_PROVEN")
        else:
            input_axes = {}
        for aid, access in enumerate(guard["accesses"]):
            value = arguments[access["root"]]
            pointer = integer(value.data_ptr()) if hasattr(value, "data_ptr") else integer(value)
            if not 0 < pointer < (1 << 64):
                raise GuardFailure("INVALID_MEMORY_ARGUMENT")
            domain = ByteDomain(arguments, grid, access)
            if require_full_masks:
                domain.require_full_mask(access.get("mask"))
            if require_same_fractal_bounds:
                domain.require_same_fractal_bounds()
                domain.require_full_axes(input_axes.get(aid, []))
            result = domain.interval(pointer)
            if result is not None:
                ranges.append(result)
        check_layout_guards(guard, arguments, grid, factor)
        for i, (lo, hi, write) in enumerate(ranges):
            for a, b, other_write in ranges[:i]:
                if (write or other_write) and lo < b and a < hi:
                    raise GuardFailure("WRITE_ALIAS")
        return True, "PROVEN_AT_RUNTIME"
    except (GuardFailure, KeyError, IndexError, TypeError, OverflowError) as error:
        return False, str(error)
