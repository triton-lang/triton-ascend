"""At most three VV candidates, using grid size before performance scoring."""
import math


def plan_candidates(grid, candidates):
    if len(grid) != 3 or any(type(n) is not int or n <= 0 for n in grid):
        raise ValueError("grid must have three positive integer dimensions")
    baseline = candidates.get("unmerged", {})
    if type(baseline.get("factor")) is not int or baseline.get("factor") != 1 or baseline.get("axis") != -1:
        raise ValueError("an explicit unmerged baseline is required")
    chosen, reasons = ["unmerged"], {"unmerged": "original kernel"}
    for factor in (2, 4):
        available = []
        for key, value in candidates.items():
            axis = value.get("axis")
            if type(value.get("factor")) is not int or value.get("factor") != factor or type(
                    axis) is not int or not 0 <= axis < 3:
                continue
            if key != f"a{axis}_f{factor}":
                raise ValueError("candidate key differs from axis/factor")
            if grid[axis] < factor:
                continue
            # Already prepared candidates supply legal axes. Prefer less tail
            # padding, then the longest grid axis. No operator name or score.
            padding_ratio = math.ceil(grid[axis] / factor) * factor / grid[axis]
            available.append((padding_ratio, -grid[axis], axis, key))
        if available:
            key = min(available)[-1]
            chosen.append(key)
            reasons[key] = "least tail padding, then longest available grid axis"
    return dict(keys=chosen, reasons=reasons, grid=list(grid), omitted=[k for k in candidates if k not in chosen],
                limit=3, scope="one prepared legal axis per factor; does not prove axis-optimality")
