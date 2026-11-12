"""Enumerate CV candidates without assuming that a grid axis means M or N.

The compiler establishes the merge kind. Extents only establish whether a full
group exists. In particular, grid=(8,1,1) still has axis-0 candidates even when
the two spatial axes cannot be merged.
"""

VERSION = "cv-all-grid-axes-v1"


def plan_candidates(grid, factors=(2, 4)):
    if len(grid) != 3 or any(type(n) is not int or n <= 0 for n in grid):
        raise ValueError("grid must contain three positive integer extents")
    factors = tuple(sorted(set(factors)))
    if not factors or any(type(f) is not int or f not in (2, 4, 6) for f in factors):
        raise ValueError("invalid requested CV factors")
    result = {"unmerged": dict(axis=-1, factor=1, status="NEEDS_COMPILATION")}
    for axis, extent in enumerate(grid):
        for factor in factors:
            value = dict(axis=axis, factor=factor, status="NEEDS_COMPILATION")
            if extent < factor or extent % factor:
                value.update(status="NO_MERGE_OPPORTUNITY",
                             error=f"grid[{axis}]={extent} cannot form full groups of {factor}")
            result[f"a{axis}_f{factor}"] = value
    return result


def merge_kind(metadata, factor):
    if factor == 1:
        return "unmerged"
    report = metadata.get("block_batching", {})
    mode, lowering = report.get("cv_mode"), report.get("cv_lowering")
    if mode == "independent_batch_v1":
        return "independent_batch"
    if mode == "spatial_concat_v1":
        if lowering == "spatial_m_v1":
            return "M"
        if lowering == "spatial_n_v1":
            return "N"
    raise ValueError(f"unrecognized compiler CV merge kind: {mode}/{lowering}")


def coverage(grid, candidates, factors=(2, 4)):
    required = {key for key, item in plan_candidates(grid, factors).items() if item["status"] == "NEEDS_COMPILATION"}
    missing = sorted(required - candidates.keys())
    # Missing resource/analysis evidence remains incomplete; a genuine compile
    # rejection or UB overflow is a completed exclusion, not a rankable score.
    terminal = {"STATIC_FEATURES_READY", "UB_REJECTED", "COMPILE_REJECTED", "FRONTEND_REJECTED"}
    unresolved = sorted(key for key in required & candidates.keys() if candidates[key].get("status") not in terminal)
    return dict(candidate_policy=VERSION, all_axes_considered=not missing, all_candidates_resolved=not missing
                and not unresolved, missing_candidates=missing, unresolved_candidates=unresolved)
