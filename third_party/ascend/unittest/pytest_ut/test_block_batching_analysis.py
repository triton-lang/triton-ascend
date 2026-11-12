# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
from triton._C.libtriton import ascend, ir
from triton.backends.ascend.compiler import NPUOptions, make_ttir
from triton.backends.ascend.block_batching import BatchingRejected

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def parse(tmp_path, body="tt.return"):
    context = ir.context()
    ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    path = tmp_path / "input.mlir"
    path.write_text("module { tt.func public @kernel(%arg: i32) { " + body + " } }")
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


@pytest.mark.parametrize("factor", [2, 3, 6, 16])
@pytest.mark.parametrize("axis", [-1, 0, 1, 2])
def test_preflight_never_changes_ir(tmp_path, factor, axis):
    module = parse(tmp_path)
    before = str(module)
    report = ascend.passes.ttir.analyze_block_batching(module, factor, axis)
    assert report["compile_status"] == "REJECTED"
    assert report["reason"] == ("NO_PROGRAM_AXIS" if axis == -1 else "UNUSED_BATCH_AXIS")
    assert report["operations"]["tt.return"] == 1
    assert report["requested_factor"] == factor
    assert report["requested_axis"] == axis
    assert report["applied_factor"] == 1
    assert str(module) == before
    assert module.verify()


@pytest.mark.parametrize("factor,axis", [(0, 0), (-1, 0), (True, 0), (2, False), (2, 3), (2, -2), (2.0, 0)])
def test_cpp_checks_request_independently(tmp_path, factor, axis):
    with pytest.raises(ValueError):
        ascend.passes.ttir.analyze_block_batching(parse(tmp_path), factor, axis)


def test_loop_without_a_program_axis_has_diagnostic(tmp_path):
    module = parse(
        tmp_path, """
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      scf.for %iv = %zero to %arg step %one : i32 { scf.yield }
      tt.return
    """)
    report = ascend.passes.ttir.analyze_block_batching(module, 4, 0)
    assert report["reason"] == "UNUSED_BATCH_AXIS"
    assert report["location"]


@pytest.mark.parametrize("graph", [True, False])
def test_rejection_keeps_baseline_ir_and_options(tmp_path, graph):
    baseline = parse(tmp_path, "%pid = tt.get_program_id x : i32\n tt.return")
    requested = parse(tmp_path, "%pid = tt.get_program_id x : i32\n tt.return")
    opts = NPUOptions(arch="Ascend950", enable_graph_optimize=graph)
    batch_opts = NPUOptions(arch="Ascend950", enable_graph_optimize=graph, block_batching_factor=3)
    metadata = {}
    with pytest.raises(BatchingRejected, match="NO_PROGRAM_AXIS"):
        make_ttir(requested, metadata, batch_opts)
    result = make_ttir(baseline, {}, opts)
    assert batch_opts.enable_graph_optimize == graph
    assert "__hacc_original_grid" not in str(result)
    assert "hacc.block_mapping_owner" not in str(result)
