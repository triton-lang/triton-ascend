"""Keep independent batched scans one-dimensional through backend lowering."""
import re

import pytest
from triton._C.libtriton import ascend, ir

pytestmark = pytest.mark.backend("native")


def parse(tmp_path, text):
    context = ir.context()
    ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    path = tmp_path / "scan.mlir"
    # In normal compilation the preceding lowering loads func and tensor.
    # An empty owned module lets the same API load its required dialects here.
    path.write_text('module attributes {hacc.block_mapping_owner = "block_batching"} {}')
    bootstrap = ir.parse_mlir_module(str(path), context)
    assert ascend.passes.ttir.lower_block_batching_scans(bootstrap, 2) == 0
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    assert module.verify()
    return module


def fixture(shape="2x512", dtype="f32", axis=1, reverse=False, owner="block_batching", calls=1, mode="aiv"):
    ty = f"tensor<{shape}x{dtype}>"
    attr = f'attributes {{hacc.block_mapping_owner = "{owner}"}}' if owner else ''
    body = '\n'.join(
        f'%r{i} = func.call @triton_cumsum(%x, %axis, %reverse) : ({ty}, i32, i1) -> {ty}' for i in range(calls))
    return f'''module {attr} {{
      func.func private @triton_cumsum({ty}, i32, i1) -> {ty}
      func.func @kernel(%x: {ty}) -> {ty} attributes {{mix_mode = "{mode}", parallel_mode = "simd"}} {{
        %axis = arith.constant {axis} : i32
        %reverse = arith.constant {str(reverse).lower()}
        {body}
        return %r{calls-1} : {ty}
      }}
    }}'''


@pytest.mark.parametrize("factor", [2, 3, 6, 8, 16, 32, 64])
@pytest.mark.parametrize("dtype", ["f32", "i32"])
@pytest.mark.parametrize("reverse", [False, True])
def test_lane_scan_preserves_axis_direction_and_type(tmp_path, factor, dtype, reverse):
    module = parse(tmp_path, fixture(shape=f"{factor}x512", dtype=dtype, reverse=reverse))
    assert ascend.passes.ttir.lower_block_batching_scans(module, factor) == 1
    assert module.verify()
    text = str(module)
    assert text.count('tensor.extract_slice') == factor
    assert text.count('tensor.insert_slice') == factor
    callees = re.findall(r'call @(triton_cumsum[^ (]*)', text)
    assert len(callees) == len(set(callees)) == factor
    assert f'tensor<512x{dtype}>' in text
    assert 'parallel_mode = "mix_simd_simt"' in text
    assert '#hivm.vf_mode<MIX>' in text
    for lane in range(factor):
        assert f'[{lane}, 0] [1, 512] [1, 1]' in text
    assert f'arith.constant {str(reverse).lower()}' in text
    # Applying twice must not add lanes or change already lowered calls.
    assert ascend.passes.ttir.lower_block_batching_scans(module, factor) == 0
    assert str(module) == text


@pytest.mark.parametrize("change", [
    dict(owner=None),
    dict(owner="legacy"),
    dict(shape="512", axis=0),
    dict(shape="2x8x64", axis=2),
    dict(shape="3x512"),
    dict(shape="2x?"),
    dict(axis=0),
    dict(dtype="f16"),
    dict(mode="mix")
])
def test_nonmatching_scan_is_unchanged(tmp_path, change):
    module = parse(tmp_path, fixture(**change))
    before = str(module)
    assert ascend.passes.ttir.lower_block_batching_scans(module, 2) == 0
    assert str(module) == before


@pytest.mark.parametrize("factor", [0, 1, 65])
def test_disabled_or_out_of_capacity_factor_is_unchanged(tmp_path, factor):
    module = parse(tmp_path, fixture())
    before = str(module)
    assert ascend.passes.ttir.lower_block_batching_scans(module, factor) == 0
    assert str(module) == before


def test_each_call_has_its_own_declaration(tmp_path):
    module = parse(tmp_path, fixture(calls=2))
    assert ascend.passes.ttir.lower_block_batching_scans(module, 2) == 2
    assert module.verify()
    text = str(module)
    callees = re.findall(r'call @(triton_cumsum[^ (]*)', text)
    assert len(callees) == len(set(callees)) == 4
    declarations = re.findall(r'func.func private @(triton_cumsum[^ (]*)', text)
    assert sorted(declarations) == sorted(callees)


@pytest.mark.parametrize('shape,axis', [('2x1x512', 2), ('2x512x1', 1), ('2x1x512x1', 2)])
def test_singleton_scan_uses_the_same_independent_lane_lowering(tmp_path, shape, axis):
    module = parse(tmp_path, fixture(shape=shape, axis=axis))
    assert ascend.passes.ttir.lower_block_batching_scans(module, 2) == 1
    assert module.verify()
    text = str(module)
    assert text.count('tensor.extract_slice') == 2
    assert text.count('tensor.insert_slice') == 2
    assert 'tensor<512xf32>' in text
    assert '#hivm.vf_mode<MIX>' in text
