import builtins
import json
from pathlib import Path
import pytest

from backend_cost_model import Analyzer, Unsupported, choose
from cv_cost_model import ASSUMPTIONS, CVSchedule, EventGraph


def test_both_vector_subblocks_wait_for_cube():
    graph = EventGraph()
    cube = CVSchedule(graph, 'aic', ASSUMPTIONS)
    cube.op(10, 'result_ready', 100)
    cube.cross_flag('llvm.hivm.SET.INTRA.BLOCKI.mode', [10, 0])
    cube.cross_flag('llvm.hivm.SET.INTRA.BLOCKI.mode', [10, 16])
    for subblock in (0, 1):
        vector = CVSchedule(graph, f'aiv{subblock}', ASSUMPTIONS)
        vector.cross_flag('llvm.hivm.WAIT.INTRA.BLOCKI.mode', [1, 0])
        vector.vf({'cycles': 20, 'counts': {}})
    result = graph.finish(True)
    # Second subblock's set follows the first on FIX, then waits and computes.
    assert result['cycles'] == 100 + 4 + 4 + 4 + 20 + 8
    assert result['cross_flag_pairs'] == 2
    for event in result['events']:
        if event['label'] == 'vector_function':
            assert event['start'] >= 108


def test_cross_flags_match_fifo_and_reject_unmatched_wait():
    graph = EventGraph()
    cube = CVSchedule(graph, 'aic', ASSUMPTIONS)
    vector = CVSchedule(graph, 'aiv0', ASSUMPTIONS)
    for _ in range(2):
        cube.op(10, 'produce', 50)
        cube.cross_flag('llvm.hivm.SET.INTRA.BLOCKI.mode', [10, 0])
        vector.cross_flag('llvm.hivm.WAIT.INTRA.BLOCKI.mode', [1, 0])
        vector.vf({'cycles': 80, 'counts': {}})
    result = graph.finish()
    assert result['cycles'] == 54 + 2 * (4 + 80 + 8)
    graph = EventGraph()
    CVSchedule(graph, 'aiv0', ASSUMPTIONS).cross_flag('llvm.hivm.WAIT.INTRA.BLOCKI.mode', [1, 0])
    with pytest.raises(Unsupported, match='unmatched cross-core'):
        graph.finish()


def test_cyclic_cross_core_wait_is_not_silently_zero_cost():
    graph = EventGraph()
    cube = CVSchedule(graph, 'aic', ASSUMPTIONS)
    vector = CVSchedule(graph, 'aiv0', ASSUMPTIONS)
    cube.cross_flag('llvm.hivm.WAIT.INTRA.BLOCKI.mode', [10, 1])
    cube.cross_flag('llvm.hivm.SET.INTRA.BLOCKI.mode', [10, 0])
    vector.cross_flag('llvm.hivm.WAIT.INTRA.BLOCKI.mode', [1, 0])
    vector.cross_flag('llvm.hivm.SET.INTRA.BLOCKI.mode', [1, 1])
    with pytest.raises(Unsupported, match='cyclic'):
        graph.finish()


def test_mma_uses_m_k_n_and_releases_pingpong_event():
    graph = EventGraph()
    cube = CVSchedule(graph, 'aic', ASSUMPTIONS)
    cube.flag('llvm.hivm.SET.FLAG.IMM', [2, 3, 0])
    args = [None] * 46
    args[23:26] = [16, 32, 48]
    args[37:46] = [-1, -1, -1, -1, -1, -1, -1, 0, 0]
    cube.mma('mma_tile_float_to_float', args)
    cube.flag('llvm.hivm.WAIT.FLAG.IMM', [2, 3, 0])
    result = cube.result()
    assert result['counts']['useful_flops'] == 2 * 16 * 32 * 48
    assert result['counts']['l1_to_l0_bytes'] == (16 * 32 + 32 * 48) * 4
    assert graph.finish()['cycles'] > 0
    args[23:26] = [2048, 128, 128]
    with pytest.raises(Unsupported, match='single-buffer'):
        cube.mma('mma_tile_float_to_float', args)


@pytest.mark.parametrize('name',
                         ['mma_tile_float_to_float', 'mma_tile_float_to_float_ta', 'mma_tile_float_to_float_tb'])
@pytest.mark.parametrize('m,k,n,slices,padded_k', [
    (128, 128, 32, [64, 64], 128),
    (32, 128, 128, [64, 64], 128),
    (128, 132, 32, [64, 64, 4], 136),
    (128, 64, 32, [64], 64),
])
def test_l0_k_slices_preserve_flops_and_l1_event_lifetime(name, m, k, n, slices, padded_k):
    graph = EventGraph()
    cube = CVSchedule(graph, 'aic', ASSUMPTIONS)
    for event in (0, 1):
        cube.flag('llvm.hivm.SET.FLAG.IMM', [2, 3, event])
    for event in (4, 5):
        cube.flag('llvm.hivm.SET.FLAG.IMM', [4, 3, event])
    args = [None] * 46
    args[23:26] = [m, k, n]
    args[37:46] = [4, 5, 6, 7, -1, -1, -1, 0, 0]
    cube.mma(name, args)
    for event in (6, 7):
        cube.flag('llvm.hivm.WAIT.FLAG.IMM', [3, 4, event])
    for event in (0, 1):
        cube.flag('llvm.hivm.WAIT.FLAG.IMM', [2, 3, event])
    counts = cube.result()['counts']
    assert counts['mma_calls'] == len(slices)
    assert counts['useful_flops'] == 2 * m * k * n
    assert counts['padded_flops'] == 2 * m * padded_k * n
    assert counts['l1_to_l0_bytes'] == 4 * (m + n) * padded_k
    events = graph.finish(True)['events']
    matmuls = [e for e in events if e['label'] == name]
    assert len(matmuls) == len(slices)
    assert sum(e['label'] == 'wait (4, 3, 4)' for e in events) == 1
    assert sum(e['label'] == 'wait (4, 3, 5)' for e in events) == 1
    assert sum(e['label'] == 'set (3, 4, 6)' for e in events) == 1
    assert sum(e['label'] == 'set (3, 4, 7)' for e in events) == 1
    assert cube.l0_pingpong == len(slices) % 2


def test_nd2nz_padding_and_dual_fixpipe_not_double_counted_as_gm_write():
    graph = EventGraph()
    cube = CVSchedule(graph, 'aic', ASSUMPTIONS)
    nd = [None] * 19
    nd[3:5], nd[10:14] = [17, 32], [4, 2, 16, 8]
    assert cube.extra_call('nd2nz_float', nd)
    fix = [None] * 26
    fix[14:16], fix[23:26] = [16, 32], [0, 0, 1]
    cube.fixpipe('fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf', fix)
    result = cube.result()
    assert result['dma_read_bytes'] == 17 * 32 * 4
    assert result['counts']['nd2nz_padded_bytes'] == 32 * 32 * 4
    assert result['counts']['fixpipe_bytes'] == 32 * 32 * 4
    assert result['dma_write_bytes'] == 0


def test_subblock_identity_selects_each_half_without_device_query():
    text = '''
define void @main() {
  %1 = call i64 @llvm.hivm.GET.SUBBLOCKID()
  %2 = icmp eq i64 %1, 0
  br i1 %2, label %3, label %4
3:
  %5 = add i64 1, 2
  br label %4
4:
  ret void
}
'''
    for subblock in (0, 1):
        graph = EventGraph()
        schedule = CVSchedule(graph, f'aiv{subblock}', ASSUMPTIONS)
        result = Analyzer(text, {}, kernel_factory=lambda _: schedule, subblock_id=subblock).trace('main', [], 0)
        assert ('3' in result['block_visits']) == (subblock == 0)


def test_static_entry_does_not_import_or_call_device_runtime(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'torch_npu', 'triton', 'pyacl', 'acl'):
            raise AssertionError('unexpected runtime import ' + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    import run_cv_spec_experiment as runner
    assert callable(runner.checked_candidate)
    assert choose({'a0_f2': {'status': 'STATIC_COST_ESTIMATED', 'resource_checked': False, 'score': 0}}) is None


def test_proof_status_cannot_bypass_binary_hash(tmp_path):
    from run_cv_spec_experiment import checked_candidate, digest
    (tmp_path / 'binary').write_text('changed binary')
    item = {'files': {'binary': {'path': str(tmp_path / 'binary'), 'sha256': 'old hash'}}}
    with pytest.raises(ValueError, match='changed compiler artifact'):
        checked_candidate(item)
