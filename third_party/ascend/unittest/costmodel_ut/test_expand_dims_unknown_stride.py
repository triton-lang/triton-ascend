"""Regression for affine strides after expanding a non-affine loaded index."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
PROFILE = ROOT / 'third_party/ascend/costmodel/profiles/simd_simt/ascend_950pr_simd_simt_v1.json'


@pytest.mark.parametrize('unknown_axis, expected_strides, contiguous', [
    (1, [None, 1], 64),
    (0, [1, None], None),
])
def test_expand_dims_preserves_only_provable_stride(tmp_path, unknown_axis, expected_strides, contiguous):
    # The loaded remainder has unknown affine stride. The newly inserted axis
    # is invariant; broadcasting it must not lose that fact or invent another.
    unknown_shape = '64x1' if unknown_axis == 1 else '1x64'
    known_axis = 1 - unknown_axis
    known_shape = '1x64' if known_axis == 0 else '64x1'
    source = f'''module {{
      tt.func public @main(%index: !tt.ptr<i32>, %data: !tt.ptr<f16>, %out: !tt.ptr<f16>) {{
        %r = tt.make_range {{start = 0 : i32, end = 64 : i32}} : tensor<64xi32>
        %ip = tt.splat %index : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
        %ips = tt.addptr %ip, %r : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
        %loaded = tt.load %ips : tensor<64x!tt.ptr<i32>>
        %c16 = arith.constant dense<16> : tensor<64xi32>
        %unknown = arith.remsi %loaded, %c16 : tensor<64xi32>
        %u = tt.expand_dims %unknown {{axis = {unknown_axis} : i32}} : tensor<64xi32> -> tensor<{unknown_shape}xi32>
        %ub = tt.broadcast %u : tensor<{unknown_shape}xi32> -> tensor<64x64xi32>
        %k = tt.expand_dims %r {{axis = {known_axis} : i32}} : tensor<64xi32> -> tensor<{known_shape}xi32>
        %kb = tt.broadcast %k : tensor<{known_shape}xi32> -> tensor<64x64xi32>
        %c64 = arith.constant dense<64> : tensor<64x64xi32>
        %scaled = arith.muli %ub, %c64 : tensor<64x64xi32>
        %offset = arith.addi %scaled, %kb : tensor<64x64xi32>
        %base = tt.splat %data : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
        %ptr = tt.addptr %base, %offset : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
        %values = tt.load %ptr : tensor<64x64x!tt.ptr<f16>>
        %outbase = tt.splat %out : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
        %outptr = tt.addptr %outbase, %offset : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
        tt.store %outptr, %values : tensor<64x64x!tt.ptr<f16>>
        tt.return
      }}
    }}'''
    from triton._C.libtriton import ascend, ir
    context = ir.context()
    ir.load_dialects(context)
    ascend.load_dialects(context)
    module_path = tmp_path / 'input.mlir'
    module_path.write_text(source)
    module = ir.parse_mlir_module(str(module_path), context)
    report = tmp_path / 'report.jsonl'
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_select_simd_simt_costmodel(pm, 'report', str(PROFILE), 'Ascend950PR_9579', 32, True, False,
                                                      False, 1, 1, 1, '', '{}', str(report))
    pm.run(module, 'expand_dims_unknown_stride')
    result = json.loads(report.read_text().splitlines()[-1])
    loads = [
        access for stage in result['stage_model']['logical_stages']
        for access in stage.get('workload', {}).get('memory_accesses_per_iteration', [])
        if access.get('address_depends_on_loaded_index') and not access.get('store')
    ]
    assert len(loads) == 1
    assert loads[0]['strides'] == expected_strides
    if contiguous is not None:
        assert loads[0]['proven_contiguous_elements'] == contiguous
