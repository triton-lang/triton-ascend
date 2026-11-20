"""Call guards for the implemented cross-tile launch recipe.

Dynamic scalar types, numeric operations and their errors belong to native JIT.
The index bounds below are specific to this T recipe, not D eligibility.
"""
from .footprint import LIMIT, adapt_local_axis
from .model import ComposedTileRecipe, LocalTileRecipe

# One signed 32-bit index domain shared with the symbolic footprint proof.
TILE_INDEX_MIN, TILE_INDEX_MAX = -LIMIT - 1, LIMIT


def _tile_index(value):
    return type(value) is int and TILE_INDEX_MIN <= value <= TILE_INDEX_MAX


def _offset_bounds(layout, grid):
    origin, *steps = layout
    spans = [(size - 1) * step for size, step in zip(grid, steps)]
    return origin + sum(min(0, span) for span in spans), origin + sum(max(0, span) for span in spans)


def _program_layout(expression, arguments, grid):
    """Evaluate a tile-independent affine base without enumerating the grid."""
    op, *operands = expression
    if op in ("lit", "arg", "num_programs"):
        if op == "lit":
            value = operands[0]
        elif op == "arg":
            value = arguments[operands[0]]
        else:
            value = grid[operands[0]]
        if not _tile_index(value):
            raise ValueError("unsupported base scalar")
        layout = (value, 0, 0, 0)
    elif op == "program_id":
        layout = (0, *(int(axis == operands[0]) for axis in range(3)))
    elif op in ("add", "sub", "mul"):
        left, right = (_program_layout(value, arguments, grid) for value in operands)
        if op == "add":
            layout = tuple(a + b for a, b in zip(left, right))
        elif op == "sub":
            layout = tuple(a - b for a, b in zip(left, right))
        elif not any(left[1:]):
            layout = tuple(left[0] * b for b in right)
        elif not any(right[1:]):
            layout = tuple(a * right[0] for a in left)
        else:
            raise ValueError("non-affine program base")
    else:
        raise ValueError("unmodeled program base")
    if any(not _tile_index(v) for v in _offset_bounds(layout, grid)):
        raise ValueError("program base can overflow")
    return layout


def _disjoint_programs(layout, grid, extent):
    span = extent
    for step, size in sorted((abs(step), size) for step, size in zip(layout[1:], grid) if size > 1):
        if step < span:
            return False
        span += step * (size - 1)
    return True


def _grid_stride_programs(recipe, arguments, grid):
    """Prove a dense mixed-radix program numbering for this grid, in O(rank)."""
    programs = grid[0] * grid[1] * grid[2]
    if not _tile_index(programs):
        raise ValueError("grid product can overflow")
    # Evaluate original expressions recursively, including canceled terms, so
    # algebraically equivalent flattenings cannot hide intermediate overflow.
    start = _program_layout(recipe.grid_start, arguments, grid)
    step = _program_layout(recipe.grid_step, arguments, grid)
    if start[0] != 0 or step != (programs, 0, 0, 0):
        raise ValueError("grid-stride origin or step does not cover the grid")
    span = 1
    for weight, size in sorted((weight, size) for weight, size in zip(start[1:], grid) if size > 1):
        if weight != span:
            raise ValueError("program IDs overlap or leave gaps")
        span *= size
    return programs


def adapt_tile(recipe, arguments, candidate_constants, grid, device, *, compile_mode):
    if not isinstance(grid, tuple) or not 1 <= len(grid) <= 3:
        return None
    if any(not _tile_index(g) or g <= 0 for g in grid):
        return None
    # The simt_only checks below are defense in depth: config.supports_tile_reuse
    # already withholds recipes in that mode, so dispatch never reaches them.
    if isinstance(recipe, ComposedTileRecipe):
        # Each proof describes the entire kernel with one axis changing. Prove
        # the chain request -> intermediate configurations -> candidate; no
        # intermediate is compiled or launched.
        if compile_mode == "simt_only" or any((tile, ) not in candidate_constants for tile in recipe.tiles):
            return None
        if any(not _tile_index(value) or value <= 0
               for tile in recipe.tiles
               for value in (arguments[tile], candidate_constants[(tile, )])):
            return None
        pending = tuple(i for i, proof in enumerate(recipe.recipes)
                        if candidate_constants[(proof.tile, )] != arguments[proof.tile])
        for index in pending:
            proof = recipe.recipes[index]
            hard_static = (set(proof.static_parameters) - set(proof.partition_parameters)) | set(
                proof.control_parameters)
            if any((i, ) in candidate_constants and candidate_constants[(i, )] != arguments[i] for i in hard_static):
                return None

        from .local_axis import independent_tile_axes

        if not independent_tile_axes(recipe.recipes):
            return None
        # Each changed axis must be valid on its own in the requested context.
        # Do not rescue a rejected substitution by first changing another tile.
        for index in pending:
            proof = recipe.recipes[index]
            constants = dict(candidate_constants)
            constants.update({(tile, ): arguments[tile] for tile in recipe.tiles if tile != proof.tile})
            if adapt_tile(proof, arguments, constants, grid, device, compile_mode=compile_mode) is None:
                return None
        # Validate one deterministic chain through the complete candidate.
        # Shared shape, overflow and memory guards still apply at every step.
        actual = arguments
        for index in pending:
            proof = recipe.recipes[index]
            constants = dict(candidate_constants)
            constants.update({(tile, ): actual[tile] for tile in recipe.tiles if tile != proof.tile})
            actual = adapt_tile(proof, actual, constants, grid, device, compile_mode=compile_mode)
            if actual is None:
                return None
        return actual
    if isinstance(recipe, LocalTileRecipe):
        if compile_mode == "simt_only":
            return None
        dimensions = (*grid, *((1, ) * (3 - len(grid))))
        return adapt_local_axis(recipe, arguments, candidate_constants, dimensions, device)
    grid = (*grid, *((1, ) * (3 - len(grid))))
    if not recipe.local_loop:
        try:
            programs = _grid_stride_programs(recipe, arguments, grid)
        except (ValueError, TypeError, IndexError):
            return None
    n, requested = (arguments[i] for i in (recipe.extent, recipe.tile))
    old = candidate_constants.get((recipe.tile, ))
    if any(not _tile_index(v) or v <= 0 for v in (n, requested, old)):
        return None

    from triton._utils import TRITON_MAX_TENSOR_NUMEL

    if any(b > TRITON_MAX_TENSOR_NUMEL for b in (requested, old)):
        return None
    # Match the Ascend frontend's arange restriction; SIMD permits other sizes.
    if compile_mode == "simt_only" and any(b & (b - 1) for b in (requested, old)):
        return None
    if n + max(old, requested) > TILE_INDEX_MAX:
        return None
    requested_blocks = (n + requested - 1) // requested
    old_blocks = (n + old - 1) // old
    if recipe.bound is not None:
        blocks = arguments[recipe.bound]
        if not _tile_index(blocks) or blocks != requested_blocks:
            return None
    if not recipe.local_loop and max(requested_blocks, old_blocks) - 1 + programs > TILE_INDEX_MAX:
        # The final loop increment must remain representable for either tile.
        return None
    ranges = []
    offsets = dict(recipe.program_offsets)

    import torch

    for index in recipe.pointers:
        tensor = arguments[index]
        try:
            layout = _program_layout(offsets.get(index, ("lit", 0)), arguments, grid)
        except (ValueError, TypeError, IndexError):
            return None
        low, high = _offset_bounds(layout, grid)
        if low < 0 or high + n + max(old, requested) > TILE_INDEX_MAX:
            return None
        if recipe.local_loop and index in recipe.stores and not _disjoint_programs(layout, grid, n):
            return None
        # Keep the supported host metadata protocol narrow and side-effect free.
        if type(tensor) is not torch.Tensor or not tensor.is_contiguous() or tensor.numel() < high + n:
            return None
        if tensor.device.type != "npu" or tensor.device.index != device:
            return None
        start = tensor.data_ptr()
        width = tensor.element_size()
        mapping = (start + layout[0] * width, *(step * width for step in layout[1:]), width)
        ranges.append((index, start + low * width, start + (high + n) * width, mapping))
    for i, a, b, mapping in ranges:
        for j, c, d, other_mapping in ranges:
            if i == j or (i not in recipe.stores and j not in recipe.stores):
                continue
            # In-place access is safe only with the same element-to-byte mapping.
            if max(a, c) < min(b, d) and mapping != other_mapping:
                return None
    adapted = list(arguments)
    adapted[recipe.tile] = old
    if recipe.bound is not None:
        adapted[recipe.bound] = old_blocks
    return tuple(adapted)
