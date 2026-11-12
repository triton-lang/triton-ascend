import pytest

from backend_cost_model import (Analyzer, KernelSchedule, ASSUMPTIONS, Unsupported, choose, launch_arguments, split_top)


def test_loop_counts_and_loop_carried_dependency():
    template = '''
define internal ptc_simdvf void @vf(ptr addrspace(6) %0) {
  %2 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.0, <256 x i1> undef, i32 0)
  br label %3
3:
  %4 = phi i32 [ 0, %1 ], [ %9, %6 ]
  %5 = phi <64 x float> [ %2, %1 ], [ %8, %6 ]
  %7 = icmp slt i32 %4, TRIPS
  br i1 %7, label %6, label %10
6:
  %8 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %5, <64 x float> %5, <256 x i1> undef)
  %9 = add i32 %4, 1
  br label %3
10:
  ret void
}
'''
    costs = {('VFMulOp', 'F32', 'F32'): (1., 5., 2.)}
    one = Analyzer(template.replace('TRIPS', '1'), costs).trace('vf', [None], 0)
    ten = Analyzer(template.replace('TRIPS', '10'), costs).trace('vf', [None], 0)
    assert ten['counts']['llvm.hivm.vmul.s.x.v64f32'] == 10
    assert ten['cycles'] - one['cycles'] == 45


def test_data_dependent_loop_does_not_get_an_invented_trip_count():
    text = '''
define void @main(ptr addrspace(1) %0) {
  %2 = load i32, ptr addrspace(1) %0
  %3 = icmp sgt i32 %2, 0
  br i1 %3, label %4, label %5
4:
  ret void
5:
  ret void
}
'''
    with pytest.raises(Unsupported, match='data-dependent branch'):
        Analyzer(text, {}).trace('main', [None], 0)


def test_original_and_batch_grid_are_both_preserved():
    ttir = 'tt.func public @k(%n: i32, %__hacc_original_grid_x: i32, %__hacc_original_grid_y: i32, %__hacc_original_grid_z: i32) {}'
    args, grid = launch_arguments(ttir, {'grid': [11, 3, 1], 'arguments': {'n': {'kind': 'scalar', 'value': 57}}}, 0, 4)
    assert grid == [3, 3, 1]
    assert args == [None, None, 57, 11, 3, 1, 3, 3, 1]


def test_flags_enforce_dma_to_vector_dependency():
    s = KernelSchedule(ASSUMPTIONS)
    s.dma('load', 3200)
    s.flag('llvm.hivm.SET.FLAG.IMM', [4, 1, 0])
    s.flag('llvm.hivm.WAIT.FLAG.IMM', [4, 1, 0])
    s.vf({'cycles': 100, 'counts': {}})
    assert s.pipes[1] == 32 + 100 + 1 + 100 + 8
    with pytest.raises(Unsupported, match='without statically observed set'):
        s.flag('llvm.hivm.WAIT.FLAG.IMM', [4, 1, 0])


def test_only_resource_checked_static_candidates_can_be_selected():
    candidates = {
        'unmerged': {'resource_checked': True, 'status': 'STATIC_COST_ESTIMATED', 'score':
                     50}, 'a0_f2': {'resource_checked': False, 'status': 'STATIC_COST_ESTIMATED', 'score': 1}, 'a0_f4':
        {'resource_checked': True, 'status': 'STATIC_ANALYSIS_UNSUPPORTED', 'score': 0}
    }
    assert choose(candidates) == 'unmerged'
    for v in candidates.values():
        v['historical_us'] = 1e-9
    assert choose(candidates) == 'unmerged'


def test_nested_operands_split():
    assert split_top('i1 %0, { i64, [2 x i32] } { i64 2, [2 x i32] [i32 1, i32 2] }, i32 7') == [
        'i1 %0', '{ i64, [2 x i32] } { i64 2, [2 x i32] [i32 1, i32 2] }', 'i32 7'
    ]


def test_scalar_grid_stride_mapping_uses_concrete_block_id():
    text = '''
define void @main(i32 %0) {
  %2 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %3 = trunc i64 %2 to i32
  br label %4
4:
  %5 = phi i32 [ %3, %1 ], [ %8, %7 ]
  %6 = icmp slt i32 %5, %0
  br i1 %6, label %7, label %9
7:
  %8 = add nsw i32 %5, 4
  br label %4
9:
  ret void
}
'''
    a = Analyzer(text, {})
    assert a.trace('main', [10], 0)['block_visits']['7'] == 3
    assert a.trace('main', [10], 3)['block_visits']['7'] == 2
