"""Bound unchanged scalar loops without selecting a source-level partition template.

Only host integer expressions are evaluated, never device data. The existing
grid-stride proof stays symbolic; other partitions have a bounded per-program
check. Disjoint interval envelopes suffice (but are not necessary) for proving
outer iterations independent, so interleaved envelopes may conservatively miss.
"""
from itertools import pairwise, product


def scalar(expression, arguments, grid, program):
    from .footprint import integer

    op, *args = expression
    if op == "lit":
        result = args[0]
    elif op == "arg":
        result = arguments[args[0]]
    elif op == "program_id":
        result = program[args[0]]
    elif op == "num_programs":
        result = grid[args[0]]
    elif op in ("add", "sub", "mul", "floordiv", "cdiv", "minimum", "maximum"):
        a, b = (scalar(v, arguments, grid, program) for v in args)
        if op in ("floordiv", "cdiv"):
            if a < 0 or b <= 0:
                raise ValueError("unsupported range division")
            if op == "cdiv":
                integer(a + b - 1)
            result = a // b if op == "floordiv" else (a + b - 1) // b
        else:
            result = {
                "add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "minimum": lambda: min(a, b),
                "maximum": lambda: max(a, b)
            }[op]()
    else:
        raise ValueError("outer range needs a host integer expression")
    return integer(result)


def domain(loop, arguments, grid):
    """Return a conservative value interval, after proving program disjointness."""
    from .footprint import integer

    count = 1
    for size in grid:
        if integer(size) <= 0:
            raise ValueError("nonpositive grid")
        count *= size
    if count > 4096:
        raise ValueError("outer context proof budget")
    intervals = []
    for program in product(*(range(size) for size in grid)):
        start, stop, step = (scalar(v, arguments, grid, program) for v in (loop.start, loop.stop, loop.step))
        if step <= 0:
            raise ValueError("outer step must be positive")
        if start >= stop:
            continue
        last = start + ((stop - start - 1) // step) * step
        integer(last + step)  # Include the final loop increment, not just accessed indices.
        intervals.append((start, last))
    if not intervals:
        raise ValueError("empty outer domain")
    intervals.sort()
    if any(a[1] >= b[0] for a, b in pairwise(intervals)):
        raise ValueError("outer program domains may overlap")
    return intervals[0][0], intervals[-1][1]
