"""Static counterpart of v38 launch guards for the fixed A5 test profile."""
import math


def rejection(grid, axis, factor, core_floor, metadata=None):
    if len(grid) != 3 or any(type(x) is not int or x <= 0 for x in grid):
        raise ValueError('invalid launch grid')
    if type(factor) is not int or factor not in (1, 2, 4):
        raise ValueError('invalid factor')
    if type(core_floor) is not int or core_floor < 0:
        raise ValueError('invalid core floor')
    if factor == 1:
        return None
    if type(axis) is not int or not 0 <= axis < 3:
        raise ValueError('invalid axis')
    merged = list(grid)
    preserve = preserves_grid(metadata or {}, factor)
    if not preserve:
        merged[axis] = (merged[axis] + factor - 1) // factor
    if not preserve and math.prod(merged) >= math.prod(grid):
        return 'GRID_NOT_REDUCED'
    if math.prod(merged) < core_floor:
        return 'GRID_BELOW_CORE_FLOOR'
    return None


def preserves_grid(metadata, factor):
    """Validate the native loop contract; unknown or inconsistent forms fail closed."""
    report = metadata.get('block_batching') or {}
    guard = report.get('guard') or {}
    grouped = report.get('vv_lowering') == 'sequential_iterations_v1'
    if not grouped and 'loop_schedule' not in guard:
        return False
    schedule = guard.get('loop_schedule')
    if (factor == 1 or not grouped or report.get('lane_width') != factor or report.get('cv_mode') != 'disabled'
            or metadata.get('program_grid_transforms') is not None or any(
                metadata.get(key) is not False
                for key in ('program_grid_mapping_applied', 'row_coalescing_applied', 'ptsm_cap_authorized'))
            or metadata.get('coalesce_factor', 1) != 1 or type(schedule) is not dict
            or set(schedule) != {'version', 'kind', 'grid_policy', 'factor', 'loop'}
            or type(schedule['version']) is not int or schedule['version'] != 1
            or schedule['kind'] != 'sequential_iterations_v1' or schedule['grid_policy'] != 'preserve'
            or type(schedule['factor']) is not int or schedule['factor'] != factor):
        raise ValueError('VV_LOOP_GRID_CONTRACT_MISMATCH')
    loop = schedule['loop']
    if (type(loop) is not dict or set(loop) != {'id', 'bits', 'lower', 'upper', 'step'}
            or not any(loop == fact for access in guard.get('accesses', []) for fact in access.get('loops', []))
            or guard.get('layout_checks')):
        raise ValueError('INVALID_LOOP_SCHEDULE')
    return True
