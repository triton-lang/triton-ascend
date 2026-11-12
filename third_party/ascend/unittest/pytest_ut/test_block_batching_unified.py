# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import copy
import json
import pytest
from test_block_batching_guard_v2 import module, matrix_guard

pytestmark = pytest.mark.backend('native')


def unified(old, factor=3):
    spec = copy.deepcopy(old)
    spec.update(version=3, plan_kind='unified', semantic_contract_version=4, factor=factor,
                memory_policy='disjoint_or_terminal_store', terminal_read_ids=[])
    spec['obligations'] = []
    for i, access in enumerate(spec['accesses']):
        access.update(device_address=False)
        access.setdefault('volatile', False)
        access.setdefault('pointer_checks', [])
        access.setdefault('loops', [])
        spec['obligations'].append(dict(access_id=i, rule='SupportedEffect', proof='byte_domain'))
    return spec


@pytest.mark.parametrize('factor', [2, 3, 5, 64])
@pytest.mark.parametrize('rows,cols', [(1, 1), (3, 3), (17, 129)])
def test_existing_guard_positive_inclusion(factor, rows, cols):
    old = matrix_guard(2, 128)
    args = [1 << 32, 1 << 45, rows, cols, cols]
    grid = ((rows + 1) // 2, (cols + 127) // 128, 1)
    assert module.check_guard(old, args, grid, factor)[0]
    assert module.check_guard(unified(old, factor), args, grid, factor)[0]


def test_device_ssa_is_not_a_host_range_or_data_proof():
    spec = unified(matrix_guard())
    access = spec['accesses'][0]
    access.update(device_address=True, offset=None, pointer_checks=[])
    spec['obligations'][0].update(rule='ReplicateEffect', proof='lane_correspondence')
    args = [1 << 32, 1 << 45, 4, 4, 4]
    assert module.check_guard(spec, args, (2, 1, 1), 3) == (False, 'UNPROVEN_MEMORY_NONINTERFERENCE')
    spec['accesses'][1]['write'] = False
    assert module.check_guard(spec, args, (2, 1, 1), 3)[0]
    access['write'] = True
    assert module.check_guard(spec, [1 << 32, 1 << 45, 4, 4, 4], (2, 1, 1), 3) == (False, 'INVALID_EFFECT_PROOF')


def terminal_guard():
    spec = unified(matrix_guard())
    spec['accesses'][0].update(device_address=True, offset=None, pointer_checks=[])
    spec['obligations'][0].update(rule='ReplicateEffect', proof='lane_correspondence')
    spec['terminal_read_ids'] = [0]
    return spec


def test_dependent_terminal_read_needs_neither_ids_data_nor_allocation_query():

    class AddressOnly:

        def data_ptr(self):
            return 1 << 32

        def __getattr__(self, name):
            raise AssertionError('unexpected tensor query: ' + name)

    spec = terminal_guard()
    assert module.check_guard(spec, [AddressOnly(), 1 << 45, 4, 4, 4], (2, 1, 1), 3)[0]


@pytest.mark.parametrize(
    'mutation', ['missing', 'duplicate', 'wrong_id', 'boolean', 'extra_write', 'late_read', 'volatile', 'no_write'])
def test_invalid_terminal_dependency_is_rejected(mutation):
    spec = terminal_guard()
    if mutation == 'missing':
        spec['terminal_read_ids'] = []
    elif mutation == 'duplicate':
        spec['terminal_read_ids'] = [0, 0]
    elif mutation == 'wrong_id':
        spec['terminal_read_ids'] = [1]
    elif mutation == 'boolean':
        spec['terminal_read_ids'] = [False]
    elif mutation == 'extra_write':
        access = copy.deepcopy(spec['accesses'][1])
        spec['accesses'].append(access)
        spec['obligations'].append(dict(access_id=2, rule='SupportedEffect', proof='byte_domain'))
    elif mutation == 'late_read':
        spec['accesses'].reverse()
        spec['obligations'].reverse()
        for i, proof in enumerate(spec['obligations']):
            proof['access_id'] = i
        spec['terminal_read_ids'] = [1]
    elif mutation == 'volatile':
        spec['accesses'][1]['volatile'] = True
    else:
        spec['accesses'][1]['write'] = False
    assert not module.check_guard(spec, [1 << 32, 1 << 45, 4, 4, 4], (2, 1, 1), 3)[0]


def test_terminal_read_proof_does_not_bypass_other_alias_checks():
    spec = terminal_guard()
    access = copy.deepcopy(spec['accesses'][1])
    access['write'] = False
    access['root'] = 0
    spec['accesses'].insert(1, access)
    spec['obligations'].insert(1, dict(access_id=1, rule='ReplicateEffect', proof='lane_correspondence'))
    spec['obligations'][2]['access_id'] = 2
    args = [1 << 32, 1 << 45, 4, 4, 4]
    assert module.check_guard(spec, args, (2, 1, 1), 3)[0]
    args[1] = args[0]
    assert module.check_guard(spec, args, (2, 1, 1), 3) == (False, 'WRITE_ALIAS')


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'rule', 'proof', 'factor'])
def test_corrupt_plan_rejected(mutation):
    spec = unified(matrix_guard())
    if mutation == 'missing': spec['obligations'].pop()
    elif mutation == 'duplicate': spec['obligations'][1]['access_id'] = 0
    elif mutation == 'rule': spec['obligations'][0]['rule'] = 'Unproven'
    elif mutation == 'proof': spec['obligations'][0]['proof'] = 'lane_correspondence'
    else: spec['factor'] = 4
    assert not module.check_guard(spec, [1 << 32, 1 << 45, 4, 4, 4], (2, 1, 1), 3)[0]


INDIRECT = '''module {
  tt.func public @indirect(%x: !tt.ptr<f32>, %ids: !tt.ptr<i32>, %y: !tt.ptr<f32>) {
    %p = tt.get_program_id x : i32
    %ip = tt.addptr %ids, %p : !tt.ptr<i32>, i32
    %id = tt.load %ip : !tt.ptr<i32>
    %xp = tt.addptr %x, %id : !tt.ptr<f32>, i32
    %v = tt.load %xp : !tt.ptr<f32>
    %yp = tt.addptr %y, %p : !tt.ptr<f32>, i32
    tt.store %yp, %v : !tt.ptr<f32>
    tt.return
  }
}'''


@pytest.mark.parametrize('factor', [2, 3, 5])
def test_indirect_read_with_dependent_terminal_output_is_rewritten(tmp_path, factor):
    from test_block_batching_shapes import parse
    from triton._C.libtriton import ascend
    mod = parse(tmp_path, INDIRECT)
    report = ascend.passes.ttir.rewrite_block_batching(mod, factor, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    guard = json.loads(report['guard_json'])
    assert guard['terminal_read_ids'] == [1]
    assert guard['accesses'][1]['offset'] is None
    assert 'hivm.' not in str(mod)
    assert module.check_guard(guard, [1 << 32, 1 << 40, 1 << 48], (7, 1, 1), factor)[0]
    assert mod.verify()


@pytest.mark.parametrize('factor', [2, 3, 5])
@pytest.mark.parametrize('axis', [0, 1])
def test_tiled_statistics_then_selective_read_is_rewritten(tmp_path, factor, axis):
    from test_block_batching_shapes import parse
    from triton._C.libtriton import ascend
    source = INDIRECT.replace('%p = tt.get_program_id x', '%p = tt.get_program_id ' + 'xy'[axis])
    source = source.replace(
        '    %ip = tt.addptr', '''    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %zero = arith.constant 0.0 : f32
    %base = arith.muli %p, %c4 : i32
    %row = tt.addptr %x, %base : !tt.ptr<f32>, i32
    %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> (f32) : i32 {
      %ptr = tt.addptr %row, %i : !tt.ptr<f32>, i32
      %logit = tt.load %ptr : !tt.ptr<f32>
      %exp = math.exp %logit : f32
      %next = arith.addf %acc, %exp : f32
      scf.yield %next : f32
    }
    %lse = math.log %sum : f32
    %ip = tt.addptr''')
    source = source.replace('tt.addptr %x, %id', 'tt.addptr %row, %id')
    source = source.replace('    tt.store %yp, %v', '''    %logp = arith.subf %v, %lse : f32
    tt.store %yp, %logp''')
    mod = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(mod, factor, axis)
    assert report['compile_status'] == 'REWRITTEN', report
    guard = json.loads(report['guard_json'])
    assert guard['terminal_read_ids'] == [2]
    grid = (7, 1, 1) if axis == 0 else (1, 7, 1)
    assert module.check_guard(guard, [1 << 32, 1 << 40, 1 << 48], grid, factor)[0]
    assert 'hivm.' not in str(mod)
    assert mod.verify()


@pytest.mark.parametrize('independent', [True, False])
def test_terminal_store_requires_a_real_pure_value_dependency(tmp_path, independent):
    from test_block_batching_shapes import parse
    from triton._C.libtriton import ascend
    source = INDIRECT
    if independent:
        source = source.replace('    tt.store %yp, %v', '''    %one = arith.constant 1.0 : f32
    tt.store %yp, %one''')
    else:
        # Sharing an SSA address with the read is not a value dependency.
        source = source.replace('    tt.store %yp, %v', '''    %again = tt.load %yp : !tt.ptr<f32>
    tt.store %yp, %again''')
    mod = parse(tmp_path, source)
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 3, 0)
    assert report['compile_status'] == 'REJECTED', report
    assert report['reason'] == 'UNPROVEN_MEMORY_NONINTERFERENCE'
    assert str(mod) == before


def test_read_only_indirect_effect_needs_no_backend_fence(tmp_path):
    from test_block_batching_shapes import parse
    from triton._C.libtriton import ascend
    mod = parse(tmp_path, INDIRECT.replace('    tt.store %yp, %v : !tt.ptr<f32>\n', ''))
    report = ascend.passes.ttir.rewrite_block_batching(mod, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    guard = json.loads(report['guard_json'])
    assert guard['accesses'][1]['offset'] is None
    assert guard['obligations'][1]['rule'] == 'ReplicateEffect'
    assert 'hivm.' not in str(mod)
    assert mod.verify()


def test_data_derived_store_address_remains_out_of_scope(tmp_path):
    from test_block_batching_shapes import parse
    from triton._C.libtriton import ascend
    mod = parse(tmp_path, INDIRECT.replace('tt.addptr %y, %p', 'tt.addptr %y, %id'))
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 3, 0)
    assert report['compile_status'] == 'REJECTED', report
    assert str(mod) == before


def test_gather_remains_unsupported_without_mutation(tmp_path):
    from block_batching_op_cases import case_ir
    from test_block_batching_shapes import parse
    from triton._C.libtriton import ascend
    old = '%op_out = "arith.maximumf"(%v, %denomf) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>'
    body = '%indices = arith.constant dense<0> : tensor<4xi32>\n    %op_out = tt.gather %v[%indices] {axis = 0 : i32} : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>'
    mod = parse(tmp_path, case_ir('arith.maximumf').replace(old, body))
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 3, 0)
    assert report['compile_status'] == 'REJECTED', report
    assert report['reason'] == 'UNSUPPORTED_BATCH_OP'
    assert report['operation'] == 'tt.gather'
    assert str(mod) == before and mod.verify()


def test_replicated_and_shared_loads_both_require_write_noninterference():
    spec = unified(matrix_guard())
    spec['obligations'][0].update(rule='ReplicateEffect', proof='lane_correspondence')
    args = [1 << 32, 1 << 32, 4, 4, 4]
    assert module.check_guard(spec, args, (2, 1, 1), 3) == (False, 'WRITE_ALIAS')
    spec['obligations'][0].update(rule='ShareInvariantLoad', proof='byte_domain')
    assert module.check_guard(spec, args, (2, 1, 1), 3) == (False, 'WRITE_ALIAS')


def test_plan_binding_detects_rule_mutation():
    import importlib
    binding = importlib.import_module(module.__package__ + '.block_batching_guard_v3')
    spec = unified(matrix_guard())
    binding.bind_plan(spec, 'frozen source')
    assert binding.validate_plan_binding(spec)
    spec['obligations'][0]['rule'] = 'ReplicateEffect'
    assert not binding.validate_plan_binding(spec)


@pytest.mark.parametrize('guard', [None, [], 'broken', {'version': 3}])
def test_invalid_cached_plan_type_is_rejected(guard):
    import importlib
    binding = importlib.import_module(module.__package__ + '.block_batching_guard_v3')
    assert not binding.validate_plan_binding(guard)


@pytest.mark.parametrize('mutation', ['version', 'contract', 'policy', 'ordered', 'pairs', 'terminal'])
def test_previous_fenced_plan_cannot_authorize_a_candidate(mutation):
    import importlib
    binding = importlib.import_module(module.__package__ + '.block_batching_guard_v3')
    spec = unified(matrix_guard())
    binding.bind_plan(spec, 'frozen source')
    if mutation == 'version':
        spec['implementation_version'] = 18
    elif mutation == 'contract':
        spec['semantic_contract_version'] = 2
    elif mutation == 'policy':
        del spec['memory_policy']
    elif mutation == 'ordered':
        spec['ordered_effects'] = [0]
    elif mutation == 'pairs':
        spec['noninterference_pairs'] = []
    else:
        del spec['terminal_read_ids']
    spec['plan_fingerprint'] = binding.plan_digest(spec)
    assert not binding.validate_plan_binding(spec)


@pytest.mark.parametrize('same_root', [False, True])
@pytest.mark.parametrize('write_first', [False, True])
def test_noninterference_checks_all_write_pairs(same_root, write_first):
    spec = unified(matrix_guard())
    spec['accesses'][0]['write'] = write_first
    if same_root:
        spec['accesses'][1]['root'] = 0
    args = [1 << 32, 1 << 32, 4, 4, 4]
    assert module.check_guard(spec, args, (2, 1, 1), 3) == (False, 'WRITE_ALIAS')
    if same_root:
        access = spec['accesses'][1]
        access['offset'] = ['addi', 64, access['offset'], ['const', 4096, 64]]
    else:
        args[1] += 4096
    assert module.check_guard(spec, args, (2, 1, 1), 3)[0]


def test_replicated_read_still_needs_a_domain_when_writes_exist():
    spec = unified(matrix_guard())
    spec['obligations'][0].update(rule='ReplicateEffect', proof='lane_correspondence')
    spec['accesses'][0]['offset'] = ['unknown']
    assert not module.check_guard(spec, [1 << 32, 1 << 45, 4, 4, 4], (2, 1, 1), 3)[0]


def test_indirect_read_after_different_root_write_is_rejected(tmp_path):
    from test_block_batching_shapes import parse
    from triton._C.libtriton import ascend
    source = INDIRECT.replace('%y: !tt.ptr<f32>)', '%y: !tt.ptr<f32>, %alias: !tt.ptr<f32>)')
    source = source.replace(
        '%xp = tt.addptr %x, %id', '''%a = tt.addptr %alias, %p : !tt.ptr<f32>, i32
    %two = arith.constant 2.0 : f32
    tt.store %a, %two : !tt.ptr<f32>
    %xp = tt.addptr %x, %id''')
    mod = parse(tmp_path, source)
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 3, 0)
    assert report['compile_status'] == 'REJECTED', report
    assert report['reason'] == 'UNPROVEN_MEMORY_NONINTERFERENCE'
    assert str(mod) == before and mod.verify()


@pytest.mark.parametrize('same_root', [False, True])
def test_known_read_after_write_requires_disjoint_intervals(tmp_path, same_root):
    from test_block_batching_shapes import parse
    from triton._C.libtriton import ascend
    source = '''module {
      tt.func public @ordered(%x: !tt.ptr<f32>, %a: !tt.ptr<f32>, %y: !tt.ptr<f32>) {
        %p = tt.get_program_id x : i32
        %one = arith.constant 1.0 : f32
        %ap = tt.addptr %a, %p : !tt.ptr<f32>, i32
        tt.store %ap, %one : !tt.ptr<f32>
        %xp = tt.addptr %x, %p : !tt.ptr<f32>, i32
        %value = tt.load %xp : !tt.ptr<f32>
        %yp = tt.addptr %y, %p : !tt.ptr<f32>, i32
        tt.store %yp, %value : !tt.ptr<f32>
        tt.return
      }
    }'''
    if same_root:
        source = source.replace('tt.addptr %a, %p', 'tt.addptr %x, %p')
    mod = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert 'hivm.' not in str(mod)
    assert 'batch_effect_order_verified' not in str(mod)
    guard = json.loads(report['guard_json'])
    args = [1 << 32, 1 << 40, 1 << 48]
    if not same_root:
        assert module.check_guard(guard, args, (7, 1, 1), 3)[0]
    args[1] = args[0]
    assert module.check_guard(guard, args, (7, 1, 1), 3) == (False, 'WRITE_ALIAS')
    assert mod.verify()


def test_loop_carried_device_index_with_output_is_rejected(tmp_path):
    from test_block_batching_shapes import parse
    from triton._C.libtriton import ascend
    text = '''module {
      tt.func public @chain(%ids: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
        %p = tt.get_program_id x : i32
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %c3 = arith.constant 3 : i32
        %last = scf.for %i = %c0 to %c3 step %c1 iter_args(%index = %p) -> (i32) : i32 {
          %ptr = tt.addptr %ids, %index : !tt.ptr<i32>, i32
          %next = tt.load %ptr : !tt.ptr<i32>
          scf.yield %next : i32
        }
        %target = tt.addptr %out, %p : !tt.ptr<i32>, i32
        tt.store %target, %last : !tt.ptr<i32>
        tt.return
      }
    }'''
    mod = parse(tmp_path, text)
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 3, 0)
    assert report['compile_status'] == 'REJECTED', report
    assert report['reason'] == 'UNPROVEN_MEMORY_NONINTERFERENCE'
    assert str(mod) == before and mod.verify()


def test_opaque_read_with_multiple_outputs_is_rejected(tmp_path):
    from test_block_batching_layout_pointwise import column_source
    from test_block_batching_shapes import parse
    from triton._C.libtriton import ascend
    source = column_source()
    source = source.replace('%k: i32)', '%k: i32, %ids: !tt.ptr<i32>, %out: !tt.ptr<f32>)')
    extra = """
    %ip = tt.addptr %ids, %p : !tt.ptr<i32>, i32
    %index = tt.load %ip : !tt.ptr<i32>
    %address = tt.addptr %x, %index : !tt.ptr<f32>, i32
    %extra = tt.load %address : !tt.ptr<f32>
"""
    source = source.replace('    %value = tt.load', extra + '    %value = tt.load')
    source = source.replace(
        'tt.return', """
    %target = tt.addptr %out, %p : !tt.ptr<f32>, i32
    tt.store %target, %extra : !tt.ptr<f32>
    tt.return""")
    mod = parse(tmp_path, source)
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 3, 0)
    assert report['compile_status'] == 'REJECTED', report
    assert report['reason'] == 'UNPROVEN_MEMORY_NONINTERFERENCE'
    assert str(mod) == before and mod.verify()


def test_proven_shared_read_after_store_does_not_request_fences(tmp_path):
    from test_block_batching_shapes import parse
    from triton._C.libtriton import ascend
    text = """module {
      tt.func public @shared_after_store(%x: !tt.ptr<f32>, %w: !tt.ptr<f32>, %s: !tt.ptr<f32>, %y: !tt.ptr<f32>) {
        %p = tt.get_program_id x : i32
        %xp = tt.addptr %x, %p : !tt.ptr<f32>, i32
        %v = tt.load %xp : !tt.ptr<f32>
        %sp = tt.addptr %s, %p : !tt.ptr<f32>, i32
        tt.store %sp, %v : !tt.ptr<f32>
        %weight = tt.load %w : !tt.ptr<f32>
        %result = arith.mulf %v, %weight : f32
        %yp = tt.addptr %y, %p : !tt.ptr<f32>, i32
        tt.store %yp, %result : !tt.ptr<f32>
        tt.return
      }
    }"""
    mod = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    guard = json.loads(report['guard_json'])
    assert guard['memory_policy'] == 'disjoint_or_terminal_store'
    assert 'ordered_effects' not in guard
    assert guard['obligations'][2]['rule'] == 'ShareInvariantLoad'
    assert 'hivm.' not in str(mod)
    args = [1 << 32, 1 << 40, 1 << 44, 1 << 48]
    assert module.check_guard(guard, args, (7, 1, 1), 3)[0]
    args[1] = args[2]
    assert module.check_guard(guard, args, (7, 1, 1), 3) == (False, 'WRITE_ALIAS')
