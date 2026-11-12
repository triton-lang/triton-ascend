import json
from pathlib import Path
import shutil

import pytest

import backend_cost_model as backend
from vv_contention_model import ContentionProfile, schedule
from vv_mixed_model import DATA, GATHER, MixedModel, MixedSchedule
from vv_skill_model import InstructionModel, analyze_candidate
from run_vv_skill_selector import select


@pytest.fixture
def mixed():
    p = json.loads((DATA / 'provenance.json').read_text())
    evidence = dict(
        compiler={'sha256': p['compiler_sha256']}, aiv_meta=dict(path='/compiler/lib/meta_op.aiv.c310.O0.bc',
                                                                 sha256=p['meta_variants']['meta_op.aiv.c310.O0.bc']))
    return MixedModel(InstructionModel(), evidence)


def ir():
    return '''module attributes {hivm.aiv_bitcode = #hivm.aiv_bitcode<"/compiler/lib/meta_op.aiv.c310.O0.bc">,
      spec = ["VECTOR_CORE_COUNT", 56 : i32]} {
func.func @kernel() attributes {hacc.entry, hivm.vf_mode<MIX>, hivm.func_core_type<AIV>} {}
}'''


def gather_args(n):
    return [None, None, 0, 0, 0] + [None, None, 0, n, 1] * 4


def test_hash_bound_library_and_wrapper_identity(mixed):
    wrappers = (DATA / 'wrappers.ll').read_text()
    mixed.validate(wrappers, ir())
    broken = wrappers.replace('ret void', '%changed = add i64 1, 2\n  ret void', 1)
    with pytest.raises(backend.Unsupported, match='wrapper changed'):
        mixed.validate(broken, ir())
    mixed.evidence['aiv_meta']['sha256'] = 'changed'
    with pytest.raises(backend.Unsupported, match='meta-library evidence'):
        mixed.validate(wrappers, ir())


def test_changed_reference_and_unknown_compiler_rejected(mixed, tmp_path):
    mixed.evidence['compiler']['sha256'] = 'changed'
    with pytest.raises(backend.Unsupported, match='pinned compiler'):
        mixed.validate('', ir())
    shutil.copytree(DATA, tmp_path / 'data')
    with (tmp_path / 'data/meta-reference.ll').open('a') as f:
        f.write('\n; changed')
    with pytest.raises(backend.Unsupported, match='asset changed'):
        MixedModel(InstructionModel(), {}, tmp_path / 'data')


def test_gather_unknown_shapes_strides_and_abi_fail_closed(mixed):
    for args in (gather_args(None), gather_args(-1), gather_args(2**31), gather_args(2)[:-1]):
        with pytest.raises(backend.Unsupported):
            mixed.gather(args)
    args = gather_args(2)
    args[14] = 2
    with pytest.raises(backend.Unsupported, match='contiguous'):
        mixed.gather(args)
    assert mixed.gather(gather_args(4))['transaction_bytes_upper'] == 256
    assert mixed.gather(gather_args(4))['instruction_steps'] > mixed.gather(gather_args(2))['instruction_steps']


def test_simt_waits_for_simd_and_dma_and_consumer_waits_for_simt(mixed):
    s = MixedSchedule(backend.ASSUMPTIONS, mixed)
    s.vf(dict(cycles=100, counts={}))
    vector = s.pipe_refs[1][0]
    s.dma('load', 1024)
    dma = s.pipe_refs[4][0]
    assert s.extra_call(GATHER, gather_args(4))
    simt = len(s.events) - 1
    assert {vector, dma}.issubset(dict(s.events[simt]['dependencies']))
    s.vf(dict(cycles=10, counts={}))
    assert simt in dict(s.events[-1]['dependencies'])
    result = schedule([s.result()], 1, ContentionProfile())
    assert result['simt_gm_transaction_bytes_upper'] == 256
    assert result['dma_requests'] == 1  # SIMT is not an MTE command.
    assert s.counts['simt_gather_calls'] == 1
    assert not s.extra_call('unknown_simt_kernel', gather_args(4))


def test_library_pointer_alias_dependency_and_predicate_result(mixed):
    v = mixed.vector_factory({}, backend.ASSUMPTIONS)
    v.add('llvm.hivm.vmul.s.x.v64f32', '', '%input')
    before = v.finish
    assert v.extra_call('_mlir_ciface_vcmp_lt_int64_t', [None] * 3, '', '%predicate')
    predicate_ready = v.ready['%predicate']
    assert predicate_ready > before
    v.add('llvm.hivm.vsel.v64i32', '%predicate', '%result')
    assert v.ready['%result'] > predicate_ready
    assert v.counts['llvm.hivm.psel'] == 1


def test_interpreter_keeps_helper_result_ready_and_unknown_branch_unknown(mixed):
    text = '''define internal ptc_simdvf void @vf() {
  %p = call <256 x i1> @_mlir_ciface_vcmp_lt_int64_t(ptr undef, ptr undef, <256 x i1> undef)
  %v = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> undef, <64 x i32> undef, <256 x i1> %p)
  ret void
}'''
    result = backend.Analyzer(text, {}, vector_factory=mixed.vector_factory).trace('vf', [], 0)
    assert result['cycles'] > 0
    with pytest.raises(backend.Unsupported, match='unmodelled call'):
        backend.Analyzer(text, {}).trace('vf', [], 0)
    branch = text.replace('  ret void', '  br i1 %p, label %yes, label %no\nyes:\n  ret void\nno:\n  ret void')
    with pytest.raises(backend.Unsupported, match='data-dependent branch'):
        backend.Analyzer(branch, {}, vector_factory=mixed.vector_factory).trace('vf', [], 0)


def test_backend_mixed_requires_explicit_adapter():
    with pytest.raises(backend.Unsupported, match='verified mixed-path adapter'):
        backend.analyze_candidate('', '', ir(), {'mix_mode': 'aiv', 'parallel_mode': 'mix_simd_simt'}, {}, -1, 1, {})


def test_missing_wrapper_and_shadowed_library_rejected(mixed):
    with pytest.raises(backend.Unsupported, match='wrapper definition is missing'):
        mixed.validate('call void @' + GATHER + '()', ir())
    with pytest.raises(backend.Unsupported, match='shadows'):
        mixed.validate('define void @_mlir_ciface_vbr_int64_t() {\n  ret void\n}', ir())


def test_cost_analysis_preserves_three_candidate_scope_and_rejects_aic(mixed):
    metadata = dict(target={'arch': 'Ascend950PR_9579'}, parallel_mode='mix_simd_simt', mix_mode='aic')
    with pytest.raises(backend.Unsupported):
        analyze_candidate('', '', ir(), metadata, {}, -1, 1, {}, mixed_evidence=mixed.evidence)
    with pytest.raises(backend.Unsupported, match='unmerged/2/4'):
        analyze_candidate('', '', ir(), metadata, {}, 0, 8, {}, mixed_evidence=mixed.evidence)


def test_experimental_mixed_scores_do_not_replace_default_baseline():
    baseline = dict(status='STATIC_COST_ESTIMATED', resource_checked=True, binary={'sha256': 'baseline'}, score=100,
                    factor=1)
    merged = dict(baseline, factor=2, score=1, experimental_mixed_model=True)
    candidates = {'unmerged': baseline, 'a0_f2': merged}
    default = select(candidates)
    assert default['selected'] == 'unmerged'
    assert default['static_best'] == 'a0_f2'
    assert default['analysis_complete'] and not default['model_complete']
    assert default['selection_reason'] == 'BASELINE_FALLBACK_EXPERIMENTAL_MIX'
    assert select(candidates, experimental_mixed_ranking=True)['selected'] == 'a0_f2'
    baseline['resource_checked'] = False
    assert select(candidates)['selected'] is None
