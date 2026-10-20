"""Runtime bounds and launch grids for symbolic, masked tile-axis footprints."""
from .index_expr import PreservedLoopOrigin, terms

LIMIT = 2**31 - 1


def integer(value):
    if type(value) is not int or not -LIMIT - 1 <= value <= LIMIT:
        raise ValueError("unsupported index scalar")
    return value


class Footprint:

    def __init__(self, arguments, grid, outer_domains=None):
        self.arguments, self.grid = arguments, grid
        self.outer_domains = outer_domains or {}
        self.sizes = {("program_id", axis): size for axis, size in enumerate(grid)}

    def scalar(self, expression):
        origin, coefficients = self.affine(expression)
        if coefficients:
            raise ValueError("expected a program-independent scalar")
        return origin

    def bounds(self, layout):
        origin, coefficients = layout
        spans = [(self.sizes[key] - 1) * step for key, step in coefficients.items()]
        return origin + sum(min(0, span) for span in spans), origin + sum(max(0, span) for span in spans)

    def affine(self, expression):
        op, *args = expression
        if op in ("lit", "arg", "num_programs"):
            if op == "lit":
                value = args[0]
            elif op == "arg":
                value = self.arguments[args[0]]
            else:
                value = self.grid[args[0]]
            layout = integer(value), {}
        elif op in ("program_id", "lane", "coordinate"):
            if op != "program_id":
                size = self.scalar(args[0].extent if op == "coordinate" else args[0])
                if size <= 0:
                    raise ValueError("nonpositive logical axis")
                self.sizes[expression] = size
            layout = 0, {expression: 1}
        elif op == "outer_induction":
            low = 0
            if isinstance(args[0], PreservedLoopOrigin):
                low, high = self.outer_domains[args[0]]
                size = high - low + 1
            else:
                size = self.scalar(args[0].extent)
            if size <= 0:
                raise ValueError("nonpositive outer extent")
            self.sizes[expression] = size
            layout = low, {expression: 1}
        elif op in ("induction", "offset"):
            origin = args[0]
            n = self.scalar(origin.axis.extent)
            tile = integer(self.arguments[origin.axis.tile])
            if tile <= 0 or n <= 0:
                raise ValueError("nonpositive loop range")
            last = (n - 1) // tile
            if op == "offset" or not origin.counts_tiles:
                last *= tile
            self.sizes[expression] = last + 1
            layout = 0, {expression: 1}
        elif op in ("floordiv", "cdiv"):
            left, right = (self.scalar(value) for value in args)
            if left < 0 or right <= 0:
                raise ValueError("division needs nonnegative numerator and positive divisor")
            layout = (left // right if op == "floordiv" else (left + right - 1) // right), {}
        elif op in ("add", "sub", "mul"):
            a, left = self.affine(args[0])
            b, right = self.affine(args[1])
            if op in ("add", "sub"):
                sign = 1 if op == "add" else -1
                layout = a + sign * b, {k: left.get(k, 0) + sign * right.get(k, 0) for k in left.keys() | right.keys()}
            elif not left:
                layout = a * b, {k: a * v for k, v in right.items()}
            elif not right:
                layout = a * b, {k: b * v for k, v in left.items()}
            else:
                raise ValueError("non-affine indexing")
        else:
            raise ValueError("unmodeled address expression")
        for bound in self.bounds(layout):
            integer(bound)
        return layout

    def check_original(self, expression, pointer=None):
        if expression[0] == "lit" and type(expression[1]) is bool:
            return
        if expression[0] in ("lt", "gt", "le", "ge", "eq", "ne", "and"):
            for value in expression[1:]:
                self.check_original(value, pointer)
            return

        def without_pointer(value):
            if pointer is not None and value == ("arg", pointer):
                return ("lit", 0)
            return tuple(without_pointer(v) if isinstance(v, tuple) else v for v in value)

        self.affine(without_pointer(expression))

    def masked_bounds(self, layout, mask):
        low, high = self.bounds(layout)
        used = set()
        for predicate in terms(mask, "and"):
            if predicate == ("lit", True):
                continue
            if predicate[0] != "lt":
                raise ValueError("unmodeled mask")
            left = self.affine(predicate[1])
            limit = self.scalar(predicate[2])
            left_low, left_high = self.bounds(left)
            if limit <= left_low:
                raise ValueError("empty masked footprint")
            keys = {k for k, v in left[1].items() if v}
            if not keys or keys & used:
                continue
            ratios = {layout[1].get(k, 0) // left[1][k] for k in keys}
            if len(ratios) != 1:
                continue
            ratio = ratios.pop()
            if ratio <= 0 or any(layout[1].get(k, 0) != ratio * left[1][k] for k in keys):
                continue
            high -= ratio * (left_high - min(left_high, limit - 1))
            used.update(keys)
        return low, high

    def injective(self, layout):
        span = 1
        for step, size in sorted((abs(layout[1].get(key, 0)), size) for key, size in self.sizes.items() if size > 1):
            if step < span:
                return False
            span += step * (size - 1)
        return True


def adapt_grid(recipe, arguments, constants, grid):
    """Preserve complete logical coverage when one tile maps directly to a pid."""
    if recipe.grid_axis is None:
        return grid
    if not isinstance(grid, tuple) or not 1 <= len(grid) <= 3:
        return None
    try:
        if any(integer(g) <= 0 for g in grid):
            return None
        dimensions = (*grid, *((1, ) * (3 - len(grid))))
        extent = Footprint(arguments, dimensions).scalar(recipe.extent)
        old, requested = integer(constants.get((recipe.tile, ))), integer(arguments[recipe.tile])
        if min(extent, old, requested) <= 0:
            return None
        if dimensions[recipe.grid_axis] != (extent + requested - 1) // requested:
            return None
        result = list(dimensions)
        result[recipe.grid_axis] = (extent + old - 1) // old
        return tuple(result[:max(len(grid), recipe.grid_axis + 1)])
    except (ValueError, TypeError, IndexError, KeyError):
        return None


def adapt_local_axis(recipe, arguments, constants, grid, device):
    import torch
    from triton._utils import TRITON_MAX_TENSOR_NUMEL

    old = constants.get((recipe.tile, ))
    try:
        if integer(old) <= 0 or integer(arguments[recipe.tile]) <= 0:
            return None
        adapted = list(arguments)
        adapted[recipe.tile] = old
        if any((i, ) in constants and constants[(i, )] != arguments[i] for i in recipe.static_parameters):
            return None
        selected_grid = adapt_grid(recipe, arguments, constants, grid)
        if selected_grid is None:
            return None

        from .preserved_context import domain

        for actual, launch_grid in ((arguments, grid), (adapted, selected_grid)):
            outer_domains = {
                loop: domain(loop, actual, launch_grid)
                for loop in recipe.outer_loops
                if isinstance(loop, PreservedLoopOrigin)
            }
            scalar = Footprint(actual, launch_grid, outer_domains)
            for axis in recipe.axes:
                extent = scalar.scalar(axis.extent)
                if extent <= 0 or integer(actual[axis.tile]) <= 0 or extent + actual[axis.tile] > LIMIT:
                    return None
            if recipe.grid_stride_axis is not None:
                extent = scalar.scalar(recipe.extent)
                count = (extent + actual[recipe.tile] - 1) // actual[recipe.tile]
                if count + launch_grid[recipe.grid_stride_axis] > LIMIT:
                    return None
            for loop in recipe.outer_loops:
                if isinstance(loop, PreservedLoopOrigin):
                    continue  # Original bounds and last increments checked by domain().
                count = scalar.scalar(loop.extent)
                if count <= 0 or count + launch_grid[loop.dimension] > LIMIT:
                    return None
            for shape in recipe.shapes:
                count = 1
                for size in shape:
                    size = scalar.scalar(size)
                    if size <= 0:
                        return None
                    count *= size
                if count > TRITON_MAX_TENSOR_NUMEL:
                    return None
            # Axis sizes are a pure function of this context, so one instance
            # serves every check. check_original never calls injective().
            for expression, pointer in recipe.index_checks:
                scalar.check_original(expression, pointer)
        coverage_grid = list(grid)
        if recipe.grid_axis is not None:
            # The coordinate domain already includes all programs on this axis.
            coverage_grid[recipe.grid_axis] = 1
        if recipe.grid_stride_axis is not None:
            coverage_grid[recipe.grid_stride_axis] = 1
        for loop in recipe.outer_loops:
            if isinstance(loop, PreservedLoopOrigin):
                # domain() proved disjoint per-program intervals. The body may
                # refer to the outer coordinate, but cannot observe program IDs.
                coverage_grid[:] = [1] * len(coverage_grid)
                continue
            # The union of range(pid, count, programs) is exactly [0, count).
            coverage_grid[loop.dimension] = 1
        ranges = []
        for access in recipe.accesses:
            # Fresh per access: injective() considers every axis size seen so
            # far, so sharing would reject writes over another access's axes.
            footprint = Footprint(arguments, coverage_grid, outer_domains)
            layout = footprint.affine(access.offset)
            low, high = footprint.masked_bounds(layout, access.mask)
            if low < 0 or (access.write and not footprint.injective(layout)):
                return None
            tensor = arguments[access.pointer]
            if type(tensor) is not torch.Tensor:
                return None
            if tensor.device.type != "npu" or tensor.device.index != device:
                return None
            start, width = tensor.data_ptr(), tensor.element_size()
            if tensor.is_contiguous():
                if tensor.numel() <= high:
                    return None
            else:
                # Triton receives a data pointer and explicit strides. numel()
                # is not the accessible span of a strided view; check storage
                # bytes after its offset. The layout proof above checks writes.
                available = tensor.untyped_storage().nbytes() - tensor.storage_offset() * width
                if (high + 1) * width > available:
                    return None
            mapping = (start + layout[0] * width, frozenset(
                (key, step * width) for key, step in layout[1].items()), width)
            ranges.append((start + low * width, start + (high + 1) * width, mapping, access.write))
        for i, (a, b, mapping, write) in enumerate(ranges):
            for c, d, other_mapping, other_write in ranges[i + 1:]:
                if (write or other_write) and max(a, c) < min(b, d) and mapping != other_mapping:
                    return None
        return tuple(adapted)
    except (ValueError, TypeError, IndexError, KeyError):
        return None
