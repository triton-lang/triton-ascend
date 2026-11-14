# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Automatic row/column choice must preserve complete batching groups."""
import pytest
from triton._C.libtriton import ascend
from block_batching_contract_ir import parse, symbolic_columns_ir

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


@pytest.mark.parametrize('axis,policy,selected', [(-1, 'auto', 1), (0, 'auto', 0), (1, 'auto', 1),
                                                  (-1, 'semantic_only', 0)])
def test_auto_axis_uses_proven_column_region_and_preserves_explicit_axis(tmp_path, axis, policy, selected):
    module = parse(tmp_path, symbolic_columns_ir())
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, axis, policy)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['selected_axis'] == selected
    assert module.verify()


def test_auto_column_selection_respects_singleton_grid_hint(tmp_path):
    source = symbolic_columns_ir().replace(
        'module {', 'module attributes {hacc.block_batching_grid = [128 : i32, 1 : i32, 1 : i32]} {')
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, -1)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['selected_axis'] == 0


@pytest.mark.parametrize('grid,factor,axis,selected',
                         [((128, 4, 1), 8, -1, 0),  # Avoid an inactive half-group in every column tile.
                          ((128, 4, 1), 6, -1, 0),  # The complete-row preference also covers nonpowers of two.
                          ((128, 4, 1), 4, -1, 1),  # Keep the established contiguous column path.
                          ((2, 4, 1), 8, -1, 1),  # Partial groups remain supported when neither axis fits F.
                          ((128, 4, 1), 8, 1, 1),  # Explicit requests are unchanged.
                          ])
def test_auto_column_preference_preserves_available_full_groups(tmp_path, grid, factor, axis, selected):
    hint = ', '.join(f'{extent} : i32' for extent in grid)
    source = symbolic_columns_ir().replace('module {', f'module attributes {{hacc.block_batching_grid = [{hint}]}} {{')
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, axis)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['selected_axis'] == selected
    assert module.verify()


@pytest.mark.parametrize('calculation', [
    '%pf = arith.sitofp %column_p : i32 to f32\n'
    '    %pv = tt.splat %pf : f32 -> tensor<2x4xf32>\n'
    '    %out = arith.addf %v, %pv : tensor<2x4xf32>',
    '%cast = arith.fptosi %v : tensor<2x4xf32> to tensor<2x4xi32>\n'
    '    %out = arith.sitofp %cast : tensor<2x4xi32> to tensor<2x4xf32>',
])
def test_auto_column_selection_retains_old_axis_for_unbridged_dataflow(tmp_path, calculation):
    source = symbolic_columns_ir().replace('%out = arith.addf %v, %one : tensor<2x4xf32>', calculation)
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, -1)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['selected_axis'] == 0
    assert module.verify()
