# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only proof tests; also runnable without a Triton extension."""
import importlib
import json
from pathlib import Path
import random
import sys
import types

import pytest

pytestmark = pytest.mark.backend("native")
PACKAGE = "_block_batching_host_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).parents[2] / "backend")]
    sys.modules[PACKAGE] = package
guard_module = importlib.import_module(PACKAGE + ".block_batching_guard")
report_module = importlib.import_module(PACKAGE + ".block_batching_report")
batching = importlib.import_module(PACKAGE + ".block_batching")
check_guard = guard_module.check_guard


def arg(index, bits=64):
    return ["arg", index, bits]


def const(value, bits=64):
    return ["const", value, bits]


def pid(axis):
    return ["pid", axis, 32]


def add(a, b, bits=64):
    return ["addi", bits, a, b]


def mul(a, b, bits=64):
    return ["muli", bits, a, b]


def rows_guard(block=8, column_axis=1):
    lane = ["range", 0, block, 32]
    column = lane if column_axis is None else add(mul(pid(column_axis), const(block)), lane)
    offset = add(mul(pid(0), arg(3)), column)
    return {
        "version":
        1, "axis":
        0, "accesses": [
            {"root": 0, "offset": offset, "mask": ["lt", column, arg(2)], "write": False, "bytes": 4},
            {"root": 1, "offset": offset, "mask": ["lt", column, arg(2)], "write": True, "bytes": 4},
        ]
    }


@pytest.mark.parametrize("factor", [2, 3, 4, 6, 16])
@pytest.mark.parametrize("rows,cols,block", [(1, 1, 4), (10, 7, 8), (13, 129, 128), (128, 512, 128)])
def test_masked_rows_and_tail_groups(factor, rows, cols, block):
    guard = rows_guard(block)
    args = [1 << 32, 1 << 40, cols, cols]
    for axis in (0, 1):
        guard["axis"] = axis
        assert check_guard(guard, args, (rows, (cols + block - 1) // block, 1), factor)[0]


def test_shared_weight_and_scalar_rrms_output():
    guard = rows_guard(256, None)
    guard["accesses"] += [
        {
            "root": 4, "offset": ["range", 0, 256, 32], "mask": ["lt", ["range", 0, 256, 32],
                                                                 arg(2)], "write": False, "bytes": 4
        },
        {"root": 5, "offset": pid(0), "mask": None, "write": True, "bytes": 4},
    ]
    args = [1 << 32, 1 << 40, 129, 129, 1 << 44, 1 << 48]
    assert check_guard(guard, args, (10, 1, 1), 3)[0]
    args[4] = args[1]  # Output aliases the supposedly shared weight.
    assert check_guard(guard, args, (10, 1, 1), 3) == (False, "WRITE_ALIAS")


@pytest.mark.parametrize("offset", [0, 4, -4, 32])
def test_exact_and_shifted_write_alias(offset):
    args = [1 << 32, (1 << 32) + offset, 7, 7]
    assert check_guard(rows_guard(), args, (10, 1, 1), 3)[1] == "WRITE_ALIAS"


def test_read_read_alias_is_allowed():
    guard = rows_guard()
    guard["accesses"].append(dict(guard["accesses"][0]))
    assert check_guard(guard, [10000, 100000, 7, 7], (10, 1, 1), 4)[0]


def test_stride_must_separate_all_original_rows():
    assert check_guard(rows_guard(), [10000, 100000, 7, 6], (10, 1, 1), 3)[1] == "OVERLAPPING_BLOCK_WRITES"
    assert check_guard(rows_guard(), [10000, 100000, 7, -7], (10, 1, 1), 3)[1] == "NEGATIVE_STRIDE"
    assert check_guard(rows_guard(), [10000, 100000, 7, 7], (10, 1, 2), 3)[1] == "OVERLAPPING_BLOCK_WRITES"


def test_empty_grid_does_not_touch_pointers():

    class BadPointer:

        def data_ptr(self):
            raise AssertionError("empty launch examined a pointer")

    assert check_guard(rows_guard(), [BadPointer(), BadPointer(), 7, 7], (0, 1, 1), 4)[1] == "EMPTY_GRID"


def test_pointer_and_index_overflow():
    guard = rows_guard()
    assert check_guard(guard, [10000, (1 << 64) - 8, 7, 7], (10, 1, 1), 3)[1] == "INDEX_OVERFLOW"
    guard = rows_guard(column_axis=None)
    guard["accesses"][1]["offset"] = add(mul(pid(0), arg(3), 32), ["range", 0, 8, 32], 32)
    assert check_guard(guard, [10000, 100000, 7, (1 << 31) - 1], (3, 1, 1), 3)[1] == "INDEX_OVERFLOW"


def test_unknown_and_non_affine_expressions_reject():
    guard = rows_guard()
    guard["accesses"][1]["offset"] = mul(pid(0), pid(1))
    assert check_guard(guard, [10000, 100000, 7, 7], (3, 2, 1), 3)[1] == "NON_AFFINE_ADDRESS"
    guard["accesses"][1]["offset"] = ["evil", "ignored"]
    assert not check_guard(guard, [10000, 100000, 7, 7], (3, 2, 1), 3)[0]


def test_negative_unsigned_mask_limit_cannot_shrink_the_proof_domain():
    guard = rows_guard()
    for access in guard["accesses"]:
        access["mask"] = ["lt", add(["range", 0, 8, 32], const(-10)), arg(2)]
    assert not check_guard(guard, [10000, 100000, -1, 8], (3, 1, 1), 3)[0]


def test_randomized_acceptance_implies_disjoint_original_writes():
    randomizer = random.Random(381)
    for _ in range(300):
        rows, cols, block = randomizer.randrange(1, 7), randomizer.randrange(1, 20), randomizer.choice((2, 4, 8))
        stride = randomizer.randrange(0, cols + 8)
        source, output = 100000, randomizer.choice((100000, 100004, 200000))
        accepted, _ = check_guard(rows_guard(block), [source, output, cols, stride],
                                  (rows, (cols + block - 1) // block, 1), 3)
        if accepted:
            addresses = [output + 4 * (row * stride + col) for row in range(rows) for col in range(cols)]
            reads = {source + 4 * (row * stride + col) for row in range(rows) for col in range(cols)}
            assert len(set(addresses)) == len(addresses)
            assert not set(addresses).intersection(reads)


def compiled_metadata():
    return {"block_batching": {}, "mix_mode": "aiv", "is_pure_simt": False}


def test_backend_compile_success_needs_no_npuir_report():
    metadata = compiled_metadata()
    report_module.validate_compilation_result(metadata)
    report = metadata["block_batching"]
    assert report["compile_status"] == "COMPILED_GUARDED"
    assert report["validation_policy"] == "ta_only_v1"
    assert report["workspace_bytes"] == 0
    assert report["workspace_source"] == "standard_backend_default"
    assert report["internal_simt"] == "not_inspected"
    assert "block_batching_backend_report" not in metadata


@pytest.mark.parametrize("workspace", [0, 1024])
def test_workspace_uses_standard_callback_result(workspace):
    metadata = compiled_metadata()
    metadata["workspace_size"] = workspace
    report_module.validate_compilation_result(metadata)
    assert metadata["block_batching"]["workspace_bytes"] == workspace
    assert metadata["block_batching"]["workspace_source"] == "callback"


@pytest.mark.parametrize("workspace", [None, -1, True, 1.5, "1024"])
def test_invalid_workspace_is_not_assumed_zero(workspace):
    metadata = compiled_metadata()
    metadata["workspace_size"] = workspace
    with pytest.raises(batching.BatchingRejected, match="INVALID_WORKSPACE_SIZE"):
        report_module.validate_compilation_result(metadata)


@pytest.mark.parametrize("mode,pure", [("mix", False), ("aic", False), (None, False), ("aiv", True)])
def test_unsupported_launch_mode_rejected(mode, pure):
    metadata = compiled_metadata()
    metadata.update(mix_mode=mode, is_pure_simt=pure)
    with pytest.raises(batching.BatchingRejected, match="UNSUPPORTED_FINAL_EXECUTION_MODE"):
        report_module.validate_compilation_result(metadata)


@pytest.mark.parametrize("mode", ["simd", "simd_simt_template"])
def test_compile_mode_is_preserved_without_claiming_internal_simd(mode):
    metadata = compiled_metadata()
    metadata["compile_mode"] = mode
    report_module.validate_compilation_result(metadata)
    assert metadata["compile_mode"] == mode
    assert metadata["block_batching"]["internal_simt"] == "not_inspected"


@pytest.mark.parametrize("ub", [None, 0, 4096])
def test_ub_usage_is_optional_diagnostic(ub):
    metadata = compiled_metadata()
    if ub is not None:
        metadata["required_ub_bits"] = ub
    report_module.validate_compilation_result(metadata)
    assert metadata["block_batching"]["resource_validation"] == "backend_compile_success"


@pytest.fixture
def bundle_environment(monkeypatch):
    from collections import namedtuple
    modules = {}
    for name in ('triton', 'triton.runtime', 'triton.runtime.cache', 'triton.runtime.errors', 'triton.compiler',
                 'triton.compiler.errors'):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
    knobs = types.SimpleNamespace(ascend=types.SimpleNamespace(compile_only=False))
    modules['triton'].knobs = knobs

    class Cache:

        def put(self, data, name, **kwargs):
            return '/tmp/' + name

    modules['triton.runtime.cache'].get_cache_manager = lambda key: Cache()
    modules['triton.compiler.errors'].MLIRCompilationError = type('MLIRCompilationError', (RuntimeError, ), {})
    modules['triton.runtime.errors'].OutOfResources = type('OutOfResources', (RuntimeError, ), {})
    utils = types.ModuleType(PACKAGE + '.utils')
    utils.backend_policy = 'torch_npu'
    monkeypatch.setitem(sys.modules, PACKAGE + '.utils', utils)
    monkeypatch.setattr(batching, '_capture_state', lambda: False)
    monkeypatch.setattr(batching, '_core_floor', lambda: 0)
    monkeypatch.setattr(batching, '_freeze_source', lambda backend, src, opts, revision: src)
    launches = []

    class Kernel:

        def __init__(self, name, report=None):
            self.name = self.hash = name
            fields = {'name': name}
            if report is not None: fields['block_batching'] = report
            self.metadata = namedtuple('Metadata', fields)(**fields)
            self.function = 101 if name == 'baseline' else 202
            self.packed_metadata = {'hash': name}
            self._run = None
            self.initializations = 0

        def _init_handles(self):
            if self._run is None:
                self.initializations += 1
                self._run = True

        def launch_metadata(self, grid, stream, *args):
            return {'selected': self.name}

        def run(self, *launch):
            assert launch[4] == self.function
            assert launch[5]['hash'] == self.name
            assert launch[6]['selected'] == self.name
            if not knobs.ascend.compile_only: launches.append((self.name, launch))

    report = {
        'compile_status': 'COMPILED_GUARDED', 'selected_axis': 0, 'owner': 'block_batching', 'guard':
        rows_guard(4, None), 'candidate_factor': 3, 'implementation_version': batching.IMPLEMENTATION_VERSION,
        'lane_width': 3, 'cv_mode': 'disabled'
    }
    base, candidate = Kernel('baseline'), Kernel('candidate', report)
    wrapper = batching.BatchKernel(base, candidate, batching.BatchRequest(3, 0), report, [0, 1, 2, 3])
    launch = (2, 1, 1, 42, -999, {}, None, None, None, 10000, 100000, 4, 4)
    return types.SimpleNamespace(wrapper=wrapper, base=base, candidate=candidate, report=report, launch=launch,
                                 launches=launches, knobs=knobs, Kernel=Kernel)


def test_bundle_selects_complete_variant_and_rechecks_alias(bundle_environment):
    env = bundle_environment
    env.wrapper.run(*env.launch)
    assert env.wrapper.last_launch['runtime_status'] == 'APPLIED'
    assert env.wrapper.last_launch['batch_grid'] == (1, 1, 1)
    alias = list(env.launch)
    alias[10] = alias[9]
    env.wrapper.run(*alias)
    assert env.wrapper.last_launch['runtime_status'] == 'BASELINE_SELECTED'
    assert env.wrapper.last_launch['reason'] == 'WRITE_ALIAS'
    env.wrapper.run(*env.launch)
    assert [name for name, _ in env.launches] == ['candidate', 'baseline', 'candidate']
    assert env.base.initializations == env.candidate.initializations == 1


def test_cv_partial_group_selects_complete_baseline(bundle_environment):
    env = bundle_environment
    env.wrapper.request = batching.BatchRequest(2, 0)
    env.report['cv_mode'] = 'independent_batch_v1'
    env.wrapper.run(*env.launch)
    assert env.wrapper.last_launch['runtime_status'] == 'APPLIED'
    partial = list(env.launch)
    partial[0] = 3
    env.wrapper.run(*partial)
    assert env.wrapper.last_launch['reason'] == 'CV_GRID_NOT_DIVISIBLE'
    assert env.wrapper.last_launch['applied_factor'] == 1
    assert [name for name, _ in env.launches] == ['candidate', 'baseline']


@pytest.mark.parametrize('memory', ['ub', 'cbuf', 'ca', 'cb', 'cc'])
def test_cv_resource_refusal_never_hides_crash(memory):
    message = f'{memory} overflow, requires 8388608 bits while 4194304 bits available'
    assert batching._cv_resource_refusal(1, message)
    assert not batching._cv_resource_refusal(-6, message)
    assert not batching._cv_resource_refusal(1, 'LLVM ERROR: ' + message)
    assert not batching._cv_resource_refusal(1, 'possible overflow in ' + memory)


def cv_metadata():
    return dict(
        block_batching=dict(cv_mode='independent_batch_v1', cv_lowering='rank3_batch_v1',
                            tail_policy='full_groups_only', selected_axis=0, lane_width=4), workspace_size=0,
        is_pure_simt=False, mix_mode='mix', program_grid_mapping_applied=True, row_coalescing_applied=False,
        auto_blockify_enabled=False, ptsm_cap_authorized=False, program_grid_transforms=dict(
            version=3, extent_source='runtime_original_grid', hidden_extent_axes=[0, 1, 2],
            hidden_argument_order=['originalGridX', 'originalGridY',
                                   'originalGridZ'], hidden_argument_types=['i32', 'i32', 'i32'], transforms=[
                                       dict(order=0, kind='ceil_div', axis=0, factor=4, persistent_coverage=False,
                                            grid_stride_abi_verified=False)
                                   ]))


def test_cv_final_artifact_requires_matching_launch_contract():
    metadata = cv_metadata()
    report_module.validate_compilation_result(metadata)
    assert metadata['block_batching']['validation_policy'] == report_module.CV_VALIDATION_POLICY
    for field, value, reason in [('sync_block_lock_layout', 1, 'CV_GLOBAL_SYNC_NOT_SUPPORTED'),
                                 ('auto_blockify_enabled', True, 'CV_AUTO_BLOCKIFY_NOT_PROVEN'),
                                 ('mix_mode', 'aiv', 'UNSUPPORTED_FINAL_EXECUTION_MODE'),
                                 ('bs_task_type', 91, 'CV_INVALID_TASK_TYPE')]:
        metadata = cv_metadata()
        metadata[field] = value
        with pytest.raises(batching.BatchingRejected, match=reason):
            report_module.validate_compilation_result(metadata)
    metadata = cv_metadata()
    metadata['block_batching']['lane_width'] = 8
    with pytest.raises(batching.BatchingRejected, match='CV_GRID_CONTRACT_MISMATCH'):
        report_module.validate_compilation_result(metadata)


def test_masked_cv_auto_blockify_requires_compatible_lowering_and_blacklist():
    metadata = cv_metadata()
    metadata['block_batching'].update(
        cv_lowering='rank3_masked_batch_v1', cv_memory_lowering='full_reduction_guard_v3',
        guard=dict(require_same_fractal_bounds=True, cv_mask_contract=dict(version=1, inputs=[{}, {}], outputs=[{}])))
    metadata.update(auto_blockify_enabled=True, has_auto_blockify_blacklist_op=False, enable_hivm_batch_matmul=True)
    report_module.validate_compilation_result(metadata)
    assert metadata['block_batching']['validation_policy'] == report_module.CV_VALIDATION_POLICY
    metadata['has_auto_blockify_blacklist_op'] = True
    with pytest.raises(batching.BatchingRejected, match='CV_AUTO_BLOCKIFY_NOT_PROVEN'):
        report_module.validate_compilation_result(metadata)
    metadata['has_auto_blockify_blacklist_op'] = False
    metadata['block_batching']['cv_lowering'] = 'rank3_batch_v1'
    with pytest.raises(batching.BatchingRejected, match='CV_AUTO_BLOCKIFY_NOT_PROVEN'):
        report_module.validate_compilation_result(metadata)


@pytest.mark.parametrize('lowering', ['spatial_m_v1', 'spatial_n_v1'])
def test_spatial_auto_blockify_preserves_mapping_and_sync_guards(lowering):
    metadata = cv_metadata()
    metadata['block_batching'].update(cv_mode='spatial_concat_v1', cv_lowering=lowering)
    metadata.update(auto_blockify_enabled=True, has_auto_blockify_blacklist_op=False)
    report_module.validate_compilation_result(metadata)
    for key, value, reason in [
        ('has_auto_blockify_blacklist_op', True, 'CV_AUTO_BLOCKIFY_NOT_PROVEN'),
        ('mix_mode', 'aic', 'CV_AUTO_BLOCKIFY_NOT_PROVEN'),
        ('row_coalescing_applied', True, 'CV_MAPPING_CONFLICT'),
        ('ptsm_cap_authorized', True, 'CV_MAPPING_CONFLICT'),
        ('sync_block_lock_layout', 1, 'CV_GLOBAL_SYNC_NOT_SUPPORTED'),
    ]:
        old = metadata.get(key)
        metadata[key] = value
        with pytest.raises(batching.BatchingRejected, match=reason):
            report_module.validate_compilation_result(metadata)
        metadata[key] = old


def test_bundle_capture_and_empty_choose_safe_paths(bundle_environment, monkeypatch):
    env = bundle_environment
    env.wrapper.run(*env.launch)
    monkeypatch.setattr(batching, '_capture_state', lambda: True)
    env.wrapper.run(*env.launch)
    assert env.wrapper.last_launch['reason'] == 'CAPTURE_BASELINE'
    empty = list(env.launch)
    empty[0] = 0
    empty[9] = None
    env.wrapper.run(*empty)
    assert env.wrapper.last_launch['runtime_status'] == 'EMPTY_GRID'
    assert [name for name, _ in env.launches] == ['candidate', 'baseline']


def test_capture_requires_prior_loading(bundle_environment, monkeypatch):
    monkeypatch.setattr(batching, '_capture_state', lambda: True)
    with pytest.raises(RuntimeError, match='warm up'):
        bundle_environment.wrapper.run(*bundle_environment.launch)


def test_compile_only_never_reports_applied(bundle_environment):
    env = bundle_environment
    env.knobs.ascend.compile_only = True
    env.wrapper.run(*env.launch)
    assert env.wrapper.last_launch['runtime_status'] == 'NOT_LAUNCHED'
    assert not env.launches


def test_bundle_compile_rejection_preserves_baseline_options(bundle_environment):
    env = bundle_environment
    backend = types.SimpleNamespace(
        target=types.SimpleNamespace(arch='Ascend950'), parse_options=lambda options: types.SimpleNamespace(
            is_pure_simt=False, compile_mode='simd', block_batching_backend_revision='test-toolchain'))
    src = types.SimpleNamespace(fn=types.SimpleNamespace(arg_names=['x', 'y', 'n', 'stride']),
                                signature={'x': '*fp32', 'y': '*fp32', 'n': 'i32', 'stride': 'i32'}, constants={})
    seen = []

    def compile_variant(options, frozen):
        seen.append(options)
        if options['block_batching_factor'] == 1: return env.base
        raise batching.BatchingRejected('RESOURCE_LIMIT')

    kernel = batching.compile_bundle(
        backend, src, {'block_batching_factor': 3, 'block_batching_axis': 0, 'enable_graph_optimize': False},
        compile_variant)
    assert kernel.candidate is None
    assert kernel.block_batching['reason'] == 'RESOURCE_LIMIT'
    assert [x['block_batching_factor'] for x in seen] == [1, 3]
    assert all(x['enable_graph_optimize'] is False for x in seen)
    kernel.run(*env.launch)
    assert env.launches[-1][0] == 'baseline'


def test_bundle_does_not_swallow_internal_errors(bundle_environment):
    env = bundle_environment
    backend = types.SimpleNamespace(
        target=types.SimpleNamespace(arch='Ascend950'), parse_options=lambda options: types.SimpleNamespace(
            is_pure_simt=False, compile_mode='simd', block_batching_backend_revision='test-toolchain'))
    src = types.SimpleNamespace(fn=types.SimpleNamespace(arg_names=['x']), signature={'x': '*fp32'}, constants={})

    def compile_variant(options, frozen):
        if options['block_batching_factor'] == 1: return env.base
        raise RuntimeError('internal invariant broken')

    with pytest.raises(RuntimeError, match='internal invariant'):
        batching.compile_bundle(backend, src, {'block_batching_factor': 3}, compile_variant)


@pytest.mark.parametrize("returncode,diagnostic", [(-11, b""), (134, b""), (1, b"LLVM ERROR: invalid IR")])
def test_bundle_does_not_hide_compiler_crashes(bundle_environment, returncode, diagnostic):
    import subprocess
    env = bundle_environment
    backend = types.SimpleNamespace(
        target=types.SimpleNamespace(arch='Ascend950'), parse_options=lambda options: types.SimpleNamespace(
            is_pure_simt=False, compile_mode='simd', block_batching_backend_revision='test-toolchain'))
    src = types.SimpleNamespace(fn=types.SimpleNamespace(arg_names=['x']), signature={'x': '*fp32'}, constants={})

    def compile_variant(options, frozen):
        if options['block_batching_factor'] == 1:
            return env.base
        raise subprocess.CalledProcessError(returncode, ['compiler'], stderr=diagnostic)

    with pytest.raises(subprocess.CalledProcessError):
        batching.compile_bundle(backend, src, {'block_batching_factor': 3}, compile_variant)


def test_cached_proof_rechecks_all_current_dependencies():

    class Pointer:

        def __init__(self, address):
            self.address, self.reads = address, 0

        def data_ptr(self):
            self.reads += 1
            return self.address

    source, output = Pointer(10000), Pointer(100000)
    guard = guard_module.RuntimeGuard(rows_guard(4, None), 3)
    arguments = [source, output, 4, 4]
    assert guard(arguments, (10, 1, 1))[0]
    assert guard(arguments, (10, 1, 1))[0]
    output.address = source.address
    assert guard(arguments, (10, 1, 1)) == (False, "WRITE_ALIAS")
    output.address = 100000
    arguments[3] = 0
    assert not guard(arguments, (10, 1, 1))[0]
    arguments[3] = 4
    assert guard(arguments, (10, 1, 1))[0]
    assert source.reads == output.reads == 5
    assert guard._prove.cache_info().hits == 2


def test_cached_proof_empty_grid_does_not_read_pointers():

    class NoAddress:

        def data_ptr(self):
            raise AssertionError("empty launch read a pointer")

    guard = guard_module.RuntimeGuard(rows_guard(), 3)
    assert guard([NoAddress(), NoAddress(), 4, 4], (0, 1, 1)) == (False, "EMPTY_GRID")


def test_cached_proof_does_not_retain_tensor_owners():
    import weakref

    class Pointer:

        def data_ptr(self):
            return 10000

    source = Pointer()
    owner = weakref.ref(source)
    guard = guard_module.RuntimeGuard(rows_guard(4, None), 3)
    assert guard([source, 100000, 4, 4], (10, 1, 1))[0]
    del source
    assert owner() is None


def test_cached_proof_matches_fresh_proofs_across_changing_calls():
    specification = rows_guard(4)
    guard = guard_module.RuntimeGuard(specification, 3)
    rng = random.Random(20261026)
    for _ in range(250):
        rows, cols, stride = rng.randrange(1, 20), rng.randrange(1, 32), rng.randrange(-2, 40)
        arguments = [10000, rng.choice([10000, 10004, 100000]), cols, stride]
        grid = (rows, (cols + 3) // 4, 1)
        assert guard(arguments, grid) == check_guard(specification, arguments, grid, 3)
    assert guard._prove.cache_info().currsize <= 64


@pytest.mark.parametrize("policy", [None, "npuir_report_v1", "ta_only_v1"])
def test_bundle_rejects_cached_candidate_from_old_policy(bundle_environment, policy):
    import hashlib
    from collections import namedtuple
    env = bundle_environment
    backend = types.SimpleNamespace(
        target=types.SimpleNamespace(arch="Ascend950"), parse_options=lambda options: types.SimpleNamespace(
            is_pure_simt=False, compile_mode="simd_simt_template", block_batching_backend_revision="test-toolchain"))
    src = types.SimpleNamespace(fn=types.SimpleNamespace(arg_names=["x", "y", "n", "stride"]),
                                signature={"x": "*fp32", "y": "*fp32", "n": "i32", "stride": "i32"}, constants={})
    report = dict(env.report)
    if policy is not None:
        report["validation_policy"] = policy
    env.candidate.kernel = b"candidate binary"
    fields = {
        "name": "candidate", "block_batching": report, "block_batching_binary_sha256":
        hashlib.sha256(env.candidate.kernel).hexdigest()
    }
    env.candidate.metadata = namedtuple("Metadata", fields)(**fields)
    seen = []

    def compile_variant(options, frozen):
        seen.append(options)
        return env.base if options["block_batching_factor"] == 1 else env.candidate

    kernel = batching.compile_bundle(
        backend, src, {"block_batching_factor": 3, "block_batching_axis": 0, "compile_mode": "simd_simt_template"},
        compile_variant)
    assert all(x["compile_mode"] == "simd_simt_template" for x in seen)
    assert kernel.candidate is None
    assert kernel.block_batching["reason"] == ("CANDIDATE_PLAN_CHANGED"
                                               if policy == "ta_only_v1" else "CANDIDATE_VALIDATION_MISSING")
    kernel.run(*env.launch)
    assert env.launches[-1][0] == "baseline"


def test_ub_compile_failure_selects_complete_baseline(bundle_environment):
    import subprocess
    env = bundle_environment
    backend = types.SimpleNamespace(
        target=types.SimpleNamespace(arch="Ascend950"), parse_options=lambda options: types.SimpleNamespace(
            is_pure_simt=False, compile_mode="simd", block_batching_backend_revision="test-toolchain"))
    src = types.SimpleNamespace(fn=types.SimpleNamespace(arg_names=["x"]), signature={"x": "*fp32"}, constants={})

    def compile_variant(options, frozen):
        if options["block_batching_factor"] == 1:
            return env.base
        raise subprocess.CalledProcessError(1, ["bishengir-compile"],
                                            stderr=b"ub overflow, requires 2134016 bits while 2031616 bits available!")

    kernel = batching.compile_bundle(backend, src, {"block_batching_factor": 64}, compile_variant)
    assert kernel.candidate is None
    assert kernel.block_batching["reason"] == "BATCH_TARGET_RESOURCE_REJECTED"
    assert "ub overflow" in kernel.block_batching["backend_diagnostic"]
    kernel.run(*env.launch)
    assert env.launches[-1][0] == "baseline"


@pytest.mark.parametrize("version,lane_width", [(2, 3), (3, 4), (18, 3), (19, 3), (None, 3)])
def test_bundle_rejects_old_or_padded_layout(bundle_environment, version, lane_width):
    import hashlib
    env = bundle_environment
    report = dict(env.report, validation_policy="ta_only_v1", implementation_version=version, lane_width=lane_width)
    env.candidate.metadata = env.candidate.metadata._replace(block_batching=report)
    env.candidate.kernel = b"candidate-binary"
    from collections import namedtuple
    fields = dict(env.candidate.metadata._asdict(),
                  block_batching_binary_sha256=hashlib.sha256(env.candidate.kernel).hexdigest())
    env.candidate.metadata = namedtuple("Metadata", fields)(**fields)
    backend = types.SimpleNamespace(
        target=types.SimpleNamespace(arch="Ascend950"), parse_options=lambda opts: types.SimpleNamespace(
            is_pure_simt=False, compile_mode="simd", block_batching_backend_revision="test"))
    src = types.SimpleNamespace(fn=types.SimpleNamespace(arg_names=["x"]), signature={"x": "*fp32"}, constants={})
    kernel = batching.compile_bundle(
        backend, src, {"block_batching_factor": 3}, lambda opts, frozen: env.base
        if opts["block_batching_factor"] == 1 else env.candidate)
    assert kernel.candidate is None
    assert kernel.block_batching["reason"] == "CANDIDATE_LAYOUT_MISMATCH"
    kernel.run(*env.launch)
    assert env.launches[-1][0] == "baseline"


def test_mask_cannot_hide_its_own_i32_wraparound():
    lane = ["range", 0, 4, 32]
    narrow = add(mul(pid(0), const(4, 32), 32), lane, 32)
    wide = add(mul(["extsi", 64, pid(0)], const(4)), ["extui", 64, lane])
    specification = {
        "version":
        1, "axis":
        0, "accesses": [
            {"root": 0, "offset": wide, "mask": ["lt", narrow, arg(2, 32)], "write": False, "bytes": 4},
            {"root": 1, "offset": wide, "mask": ["lt", narrow, arg(2, 32)], "write": True, "bytes": 4},
        ]
    }
    # In the original domain p=2**29 makes the signed mask index negative.
    # Proving only the p=0 domain inferred from mask<1 would miss this wrap.
    values, grid = [1 << 40, 1 << 45, 1], ((1 << 30) + 1, 1, 1)
    assert check_guard(specification, values, grid, 3) == (False, "INDEX_OVERFLOW")
    assert guard_module.RuntimeGuard(specification, 3)(values, grid) == (False, "INDEX_OVERFLOW")


@pytest.mark.parametrize('rows', [2, 3, 4, 5, 7])
def test_vv_complete_and_partial_groups_execute_candidate(bundle_environment, rows):
    env = bundle_environment
    launch = list(env.launch)
    launch[0] = rows
    env.wrapper.run(*launch)
    assert env.wrapper.last_launch['runtime_status'] == 'APPLIED'
    assert env.wrapper.last_launch['batch_grid'] == ((rows + 2) // 3, 1, 1)
    assert env.launches[-1][0] == 'candidate'


@pytest.mark.parametrize('floor', [4, 8, 32])
@pytest.mark.parametrize('offset,applied', [(-3, False), (-1, True), (0, True), (1, True)])
def test_core_floor_boundary_uses_ceiling_batch_count(bundle_environment, monkeypatch, floor, offset, applied):
    env = bundle_environment
    monkeypatch.setattr(batching, '_core_floor', lambda: floor)
    launch = list(env.launch)
    launch[0] = floor * 3 + offset
    env.wrapper.run(*launch)
    assert env.wrapper.last_launch['applied_factor'] == (3 if applied else 1)
    if not applied:
        assert env.wrapper.last_launch['reason'] == 'GRID_BELOW_CORE_FLOOR'


def test_baseline_grid_hint_is_removed_before_freezing_and_compilation(bundle_environment, monkeypatch):
    env = bundle_environment
    backend = types.SimpleNamespace(
        target=types.SimpleNamespace(arch='Ascend950'), parse_options=lambda opts: types.SimpleNamespace(
            is_pure_simt=False, compile_mode='simd', block_batching_backend_revision='test'))
    src = types.SimpleNamespace(fn=types.SimpleNamespace(arg_names=['x']), signature={'x': '*fp32'}, constants={})
    frozen_options, baseline_options, candidate_options = [], [], []

    def freeze(backend, source, opts, revision):
        frozen_options.append(dict(opts))
        return source

    def compile_variant(opts, frozen):
        assert frozen is src
        if opts['block_batching_factor'] == 1:
            baseline_options.append(dict(opts))
            return env.base
        candidate_options.append(dict(opts))
        raise batching.BatchingRejected('RESOURCE_LIMIT')

    monkeypatch.setattr(batching, '_freeze_source', freeze)
    for grid in ((1, 128, 1), (256, 1, 1)):
        options = dict(block_batching_factor=2, block_batching_axis=-1, block_batching_grid=grid)
        result = batching.compile_bundle(backend, src, options, compile_variant)
        assert result.baseline is env.base
        assert options['block_batching_grid'] == grid
    assert all('block_batching_grid' not in opts for opts in frozen_options + baseline_options)
    assert frozen_options[0] == frozen_options[1] == baseline_options[0] == baseline_options[1]
    assert [opts['block_batching_grid'] for opts in candidate_options] == [(1, 128, 1), (256, 1, 1)]
