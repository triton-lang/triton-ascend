import pytest

from backend_cost_model import Unsupported
from cv_cost_model import ASSUMPTIONS, CVSchedule, EventGraph


def cube():
    return CVSchedule(EventGraph(), 'aic', ASSUMPTIONS)


@pytest.mark.parametrize('stride,commands', [(1024, 1), (16383, 1), (16384, 4)])
def test_nd2nz_byte_stride_limit_changes_descriptors_not_bytes(stride, commands):
    schedule = cube()
    args = [None] * 23
    args[3:9] = [4, 32, 32, stride, 32, 1]
    args[12:18] = [4, 4, 2, 16, 8, 1024]
    schedule.extra_call('nd2nz_batch_float', args)
    assert schedule.counts['nd2nz_calls'] == commands
    assert schedule.dma_read_bytes == 4 * 32 * 32 * 4
    assert schedule.counts['nd2nz_padded_bytes'] == 4 * 32 * 32 * 4


@pytest.mark.parametrize('batch,m,k,n,chunks', [(2, 32, 32, 32, 1), (4, 32, 128, 32, 1), (4, 64, 128, 64, 2)])
def test_batch_mma_keeps_each_matrix_and_l0_chunk_dependencies(batch, m, k, n, chunks):
    schedule = cube()
    for src, dst, event in ((2, 3, 0), (4, 3, 4), (4, 3, 5)):
        schedule.flag('llvm.hivm.SET.FLAG.IMM', [src, dst, event])
    args = [None] * 52
    args[3] = args[16] = args[33] = batch
    args[27:30] = [m, k, n]
    args[43:52] = [4, 5, 6, 7, -1, -1, -1, 0, 0]
    schedule.extra_call('batch_mma_tile_float_to_float', args)
    for src, dst, event in ((2, 3, 0), (3, 4, 6), (3, 4, 7)):
        schedule.flag('llvm.hivm.WAIT.FLAG.IMM', [src, dst, event])
    counts = schedule.result()['counts']
    assert counts['mma_calls'] == batch
    assert counts['useful_flops'] == counts['padded_flops'] == 2 * batch * m * k * n
    assert counts['l1_to_l0_bytes'] == 4 * batch * k * (m + n)
    assert counts['batch_l0_chunks'] == chunks
    events = schedule.graph.finish(True)['events']
    assert sum(e['label'] == 'wait (4, 3, 4)' for e in events) == 1
    assert sum(e['label'] == 'set (3, 4, 6)' for e in events) == 1
    matmuls = [e for e in events if e['label'] == 'batch_mma_tile_float_to_float']
    assert len(matmuls) == batch
    assert all(e['start'] >= next(t['end'] for t in events if t['label'] == 'l1_to_l0') for e in matmuls)
    args[50] = 3
    with pytest.raises(Unsupported, match='unit-flag'):
        schedule.extra_call('batch_mma_tile_float_to_float', args)


@pytest.mark.parametrize('to_ub,stride,commands', [(False, 1024, 1), (False, 65535, 1), (False, 65536, 4),
                                                   (True, 1024, 1)])
def test_batch_fixpipe_has_one_descriptor_unless_stride_overflows(to_ub, stride, commands):
    schedule = cube()
    args = [None] * (30 if to_ub else 29)
    args[16:19] = [4, 32, 32]
    args[8], args[11], args[19], args[20], args[27] = 1024, 16, stride, 32, 0
    name = 'fixpipe_nz2nd_float_to_float_5d_to_3d_' + ('ubuf' if to_ub else 'gm')
    schedule.extra_call(name, args)
    assert schedule.counts['fixpipe_calls'] == commands
    assert schedule.counts['fixpipe_bytes'] == 4 * 32 * 32 * 4
    assert schedule.dma_write_bytes == (0 if to_ub else 4 * 32 * 32 * 4)
    if to_ub:
        args[19] = 65536
        with pytest.raises(Unsupported, match='field limits'):
            schedule.extra_call(name, args)


def test_unknown_or_partial_batch_information_cannot_be_zero_cost():
    with pytest.raises(Unsupported, match='unknown'):
        cube().extra_call('nd2nz_batch_float', [None] * 23)
    with pytest.raises(Unsupported, match='dimensions'):
        args = [None] * 52
        args[3], args[16], args[33], args[27:30] = 2, 4, 2, [32, 32, 32]
        cube().extra_call('batch_mma_tile_float_to_float', args)
