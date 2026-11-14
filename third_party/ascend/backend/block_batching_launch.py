# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pre-guard launch policy. Handle selection and guard evaluation stay in the facade."""


def candidate_launch_refusal(request, report, grid, preserve_grid, capture, *, cv, reduces_blocks, batched_blocks,
                             core_floor, auto_selected=False):
    """Return a refusal or None, preserving lazy queries and refusal precedence.

    The caller queries capture only after finding a candidate. The core query
    remains lazy: earlier refusals must not read device/environment state.
    Report reads intentionally retain their existing timing; isolating public
    reports from execution is a separate behavior change.
    """
    if cv and grid[report["selected_axis"]] % request.factor:
        # VV uses lane masks for tails; CV still requires complete groups.
        return "CV_GRID_NOT_DIVISIBLE"
    if not preserve_grid and not reduces_blocks(grid, report["selected_axis"], request.factor):
        return "GRID_NOT_REDUCED"
    # The native selector scores small CV grids. Manual CV and VV keep the floor.
    if (not (cv and auto_selected)
            and batched_blocks(grid, report["selected_axis"], 1 if preserve_grid else request.factor) < core_floor()):
        return "GRID_BELOW_CORE_FLOOR"
    if capture is not False:
        return "CAPTURE_BASELINE" if capture else "CAPTURE_QUERY_UNAVAILABLE"
    return None
