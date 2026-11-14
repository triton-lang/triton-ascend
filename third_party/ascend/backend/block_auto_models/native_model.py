"""Host metadata adapter for the in-process C++ cost analyzer."""
import json
import math
from pathlib import Path
import re
from ..block_auto_artifacts import file_digest as sha256

DATA = Path(__file__).parent

UNKNOWN = None


class Unsupported(ValueError):
    pass


def reject(condition, message, error=Unsupported):
    if condition:
        raise error(message)


def launch_arguments(ttir, contract, axis, factor, *, preserve_grid=False):
    signature = next(line for line in ttir.splitlines() if "tt.func public @" in line)
    arg_names = re.findall(r"%([\w.$-]+):", signature)
    grid = list(contract["grid"])
    if factor != 1 and not preserve_grid:
        grid[axis] = math.ceil(grid[axis] / factor)
    values = [UNKNOWN, UNKNOWN]  # standard TA synchronization and workspace pointers
    for name in arg_names:
        if name.startswith("__hacc_original_grid_"):
            values.append(contract["grid"]["xyz".index(name[-1])])
        elif name in contract["arguments"]:
            arg = contract["arguments"][name]
            values.append(arg["value"] if arg["kind"] == "scalar" else UNKNOWN)
        else:
            raise Unsupported("missing launch scalar/argument " + name)
    return values + grid, grid


def launch_configuration(ttir, final_ir, metadata, contract, axis, factor, core):
    m = re.search(r'"' + core + r'_CORE_COUNT", (\d+) : i32', final_ir)
    reject(not m, 'CPU target ' + core.lower() + ' core count missing')
    core_count = int(m.group(1))
    for transform in ("ptsm_cap_authorized", "row_coalescing_applied"):
        if metadata.get(transform):
            raise Unsupported("additional grid transform " + transform)
    reject(metadata.get('coalesce_factor', 1) != 1, 'additional coalesce transform')
    preserve_grid = core == 'VECTOR' and preserves_grid(metadata, factor)
    args, grid = launch_arguments(ttir, contract, axis, factor, preserve_grid=preserve_grid)
    auto = metadata["auto_blockify_enabled"]
    tasks = math.prod(grid)
    blocks = min(tasks, core_count) if auto else tasks
    reject(blocks <= 0 or core_count <= 0, 'empty launch or invalid core count')
    return args, grid, core_count, blocks


def rejection(grid, axis, factor, core_floor, metadata=None):
    reject(len(grid) != 3 or any((type(x) is not int or x <= 0 for x in grid)), 'invalid launch grid', ValueError)
    reject(type(factor) is not int or factor not in (1, 2, 4), 'invalid factor', ValueError)
    reject(type(core_floor) is not int or core_floor < 0, 'invalid core floor', ValueError)
    if factor == 1:
        return None
    reject(type(axis) is not int or not 0 <= axis < 3, 'invalid axis', ValueError)
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
    reject(factor == 1 or report.get('lane_width') != factor, 'VV_LOOP_GRID_CONTRACT_MISMATCH', ValueError)
    schedule = guard.get('loop_schedule')
    reject(
        type(schedule) is not dict or set(schedule) != {'version', 'kind', 'grid_policy', 'factor', 'loop'}
        or type(schedule['version']) is not int or schedule['version'] != 1
        or schedule['kind'] != 'sequential_iterations_v1' or schedule['grid_policy'] != 'preserve'
        or type(schedule['factor']) is not int or schedule['factor'] != factor, 'VV_LOOP_SCHEDULE_MISMATCH', ValueError)
    loop = schedule['loop']
    reject(
        type(loop) is not dict or set(loop) != {'id', 'bits', 'lower', 'upper', 'step'}
        or not any(loop == fact for access in guard.get('accesses', []) for fact in access.get('loops', []))
        or guard.get('layout_checks'), 'VV_LOOP_SCHEDULE_MISMATCH', ValueError)
    reject(
        not grouped or report.get('cv_mode') != 'disabled' or metadata.get('program_grid_transforms') is not None
        or metadata.get('program_grid_mapping_applied') is not False
        or metadata.get('row_coalescing_applied') is not False or metadata.get('ptsm_cap_authorized') is not False
        or metadata.get('coalesce_factor', 1) != 1, 'VV_LOOP_GRID_CONTRACT_MISMATCH', ValueError)
    return True


def meta_path(ir):
    paths = re.findall(r'hivm\.aiv_bitcode = #hivm\.aiv_bitcode<"([^"]+)">', ir)
    if len(paths) != 1:
        raise Unsupported('MIX needs one final-IR AIV bitcode reference')
    return Path(paths[0])


def mixed_parameters(final_ir, files):
    """Check the installed profile once; helper bodies come from optimized LLVM."""
    provenance = json.loads((DATA / 'vv_mixed_data/provenance.json').read_text())
    parameters = provenance['assumptions']
    for name in ('simt_transaction_bytes', 'simt_transition_cycles', 'simt_instruction_cycles'):
        reject(type(parameters.get(name)) is not int or parameters[name] <= 0, 'invalid MIX parameter: ' + name)
    compiler = files['compiler']
    reject(compiler['sha256'] != provenance['compiler_sha256'], 'MIX requires the pinned compiler')
    path = meta_path(final_ir)
    expected_dir = Path(compiler['path']).resolve().parent.parent / 'lib'
    reject(path.resolve().parent != expected_dir, 'MIX library is outside the candidate compiler installation',
           ValueError)
    reject(path.name not in provenance['meta_variants'] or sha256(path) != provenance['meta_variants'][path.name],
           'MIX needs matching declared AIV meta-library evidence')
    entries = [line for line in final_ir.splitlines() if 'func.func ' in line and 'hacc.entry' in line]
    reject(
        len(entries) != 1 or 'hivm.vf_mode<MIX>' not in entries[0] or 'hivm.func_core_type<AIV>' not in entries[0],
        'MIX adapter supports one AIV SIMD/SIMT entry only')
    return parameters


def analyze(modules, ttir, final_ir, metadata, contract, axis, factor, *, raw=None, mixed=None, library=None):
    from triton._C.libtriton import ascend
    vv = raw is not None
    reject(metadata.get('target', {}).get('arch') != 'Ascend950PR_9579', 'unsupported target profile')
    reject(factor not in (1, 2, 4), 'factor outside automatic shortlist')
    reject(
        metadata.get('parallel_mode') not in (('simd', 'mix_simd_simt') if vv else ('simd', )),
        'unmodelled execution mode')
    args, grid, cores, blocks = launch_configuration(ttir, final_ir, metadata, contract, axis, factor,
                                                     'VECTOR' if vv else 'CUBE')
    if not vv and 'aiv' in modules:
        counts = re.findall(r'"VECTOR_CORE_COUNT", (\d+) : i32', final_ir)
        reject(counts != [str(cores * 2)], 'only one Cube with two Vector subblocks is modelled')
    config = dict(mode='vv' if vv else 'cv', modules=modules, name=metadata['name'], arguments=args, grid=grid,
                  cores=cores, blocks=blocks, auto_blockify=metadata['auto_blockify_enabled'])
    if vv:
        config['raw_llvm'] = raw
        if metadata['parallel_mode'] == 'mix_simd_simt':
            config['mixed'] = mixed_parameters(final_ir, mixed)
        if re.search(r'\bcall\b[^\n]*@pow_fp32\(', modules['aiv']):
            proof = json.loads((DATA / 'vv_library_data/provenance.json').read_text())
            path = DATA / 'vv_library_data/pow_fp32.ll'
            reject(
                not library or library['sha256'] != proof['libdevice_sha256'] or sha256(path) != proof['llvm_sha256'],
                'pow needs matching pinned library evidence')
            config.update(pow_llvm=path.read_text(), pow_wrapper_steps=proof['descriptor_scalar_steps'])
    try:
        return json.loads(ascend.analyze_block_candidate(json.dumps(config, allow_nan=False)))
    except RuntimeError as error:
        raise Unsupported(str(error)) from error
