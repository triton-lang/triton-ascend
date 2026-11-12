"""Native policy preservation, boundary lowering and transactional rejection."""
import json
import pytest
from triton.backends.ascend.block_batching import IMPLEMENTATION_VERSION

extension = pytest.importorskip("triton._C.libtriton")
from test_block_batching_rewrite import parse, pointwise

ascend = extension.ascend
pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


@pytest.mark.parametrize("layout", ["semantic_only", "auto"])
@pytest.mark.parametrize("load_cache,store_cache", [("ca", "wb"), ("cg", "cs"), ("cv", "wt")])
@pytest.mark.parametrize("evict", ["evict_normal", "evict_first", "evict_last"])
@pytest.mark.parametrize("volatile", [False, True])
def test_memory_policies_survive_semantic_and_layout_rewrite(tmp_path, layout, load_cache, store_cache, evict,
                                                             volatile):
    flags = " {isVolatile = true}" if volatile else ""
    source = pointwise(0).replace(
        "tt.load %xa, %mask, %zero :",
        f"tt.load %xa, %mask, %zero cacheModifier = {load_cache} evictionPolicy = {evict}{flags} :",
    ).replace("tt.store %ya, %result, %mask :",
              f"tt.store %ya, %result, %mask cacheModifier = {store_cache} evictionPolicy = {evict} :")
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0, layout)
    assert report["compile_status"] == "REWRITTEN", report
    assert report["implementation_version"] == IMPLEMENTATION_VERSION
    assert module.verify()
    text = str(module)
    loads = [line for line in text.splitlines() if " = tt.load " in line]
    stores = [line for line in text.splitlines() if "tt.store " in line]
    assert len(loads) == len(stores) == 1
    assert f"cacheModifier = {load_cache}" in loads[0]
    assert f"cacheModifier = {store_cache}" in stores[0]
    if evict != "evict_normal":
        assert all(f"evictionPolicy = {evict}" in line for line in loads + stores)
    assert ("isVolatile = true" in loads[0]) == volatile
    assert json.loads(report["guard_json"])["accesses"][0]["volatile"] == volatile
    if layout == "auto":
        assert json.loads(report["layout_json"])["accesses"][0]["rewritten"]


def test_shared_volatile_result_is_lifted_instead_of_reused(tmp_path):
    source = pointwise(0).replace(
        "%v = tt.load %xa, %mask, %zero : tensor<4x!tt.ptr<f32>>",
        "%shared = tt.load %x {isVolatile = true} : !tt.ptr<f32>\n"
        "    %v = tt.splat %shared : f32 -> tensor<4xf32>")
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    line = next(line for line in str(module).splitlines() if " = tt.load " in line)
    assert "tensor<3x!tt.ptr<f32>>" in line and "isVolatile = true" in line
    assert json.loads(report["guard_json"])["accesses"][0]["volatile"]
    assert module.verify()


@pytest.mark.parametrize("hint", ["tt.contiguity", "tt.divisibility", "tt.constancy"])
def test_axis_hints_do_not_leak_into_lifted_shapes(tmp_path, hint):
    source = pointwise(0).replace("arith.addi %basev, %range :",
                                  f"arith.addi %basev, %range {{{hint} = dense<4> : tensor<1xi32>}} :")
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert hint not in str(module)
    assert module.verify()


@pytest.mark.parametrize("attr,reason", [
    ("tt.contiguity = dense<4> : tensor<2xi32>", "INVALID_AXIS_HINT"),
    ("tt.contiguity = dense<0> : tensor<1xi32>", "INVALID_AXIS_HINT"),
    ("unmodeled_semantics = true", "UNSUPPORTED_OPERATION_ATTRIBUTE"),
])
def test_invalid_hints_and_unknown_semantic_attributes_are_transactional(tmp_path, attr, reason):
    source = pointwise(0).replace("arith.addi %basev, %range :", f"arith.addi %basev, %range {{{attr}}} :")
    module = parse(tmp_path, source)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["reason"] == reason, report
    assert str(module) == before


def block_pointer_ir(padding, advance=False):
    step = "%bp2 = tt.advance %bp, [%zero] : !tt.ptr<tensor<4xf32>>" if advance else ""
    ptr = "%bp2" if advance else "%bp"
    pad = f", padding = {padding} : i32" if padding else ""
    return f'''module {{
  tt.func public @boundary(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i64) {{
    %p = tt.get_program_id x : i32
    %four = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %offset = arith.muli %p, %four : i32
    %bp = tt.make_tensor_ptr %x, [%n], [%one], [%offset] {{order = array<i32: 0>}} : !tt.ptr<tensor<4xf32>>
    {step}
    %out = tt.make_tensor_ptr %y, [%n], [%one], [%offset] {{order = array<i32: 0>}} : !tt.ptr<tensor<4xf32>>
    %v = tt.load {ptr} cacheModifier = ca {{boundaryCheck = array<i32: 0>{pad}}} : !tt.ptr<tensor<4xf32>>
    tt.store %out, %v cacheModifier = wb {{boundaryCheck = array<i32: 0>}} : !tt.ptr<tensor<4xf32>>
    tt.return
  }}
}}'''


@pytest.mark.parametrize("padding", [None, 1, 2])
@pytest.mark.parametrize("advance", [False, True])
def test_block_pointer_bounds_and_padding_become_explicit(tmp_path, padding, advance):
    module = parse(tmp_path, block_pointer_ir(padding, advance))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert module.verify()
    text = str(module)
    assert "tt.make_tensor_ptr" not in text and "tt.advance" not in text
    assert "boundaryCheck" not in text and "padding =" not in text
    assert "arith.cmpi sge" in text and "arith.cmpi slt" in text
    assert "cacheModifier = ca" in text and "cacheModifier = wb" in text
    if padding == 2:
        assert "0x7FC00000" in text or "nan" in text.lower()
    guard = json.loads(report["guard_json"])
    assert len(guard["accesses"]) == 2
    assert all(access["mask"] is not None for access in guard["accesses"])


def test_unknown_block_pointer_attribute_is_not_lost_in_normalization(tmp_path):
    source = block_pointer_ir(1).replace("{order = array<i32: 0>}",
                                         "{order = array<i32: 0>, unmodeled_semantics = true}", 1)
    module = parse(tmp_path, source)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["reason"] == "UNSUPPORTED_OPERATION_ATTRIBUTE", report
    assert str(module) == before


def test_block_pointer_without_boundary_has_no_spurious_other(tmp_path):
    source = block_pointer_ir(None).replace(" {boundaryCheck = array<i32: 0>}", "")
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert module.verify()


def test_two_dimensional_block_pointer_keeps_both_boundary_axes(tmp_path):
    source = block_pointer_ir(1).replace("tensor<4xf32>", "tensor<2x4xf32>")
    source = source.replace("%four = arith.constant 4 : i32",
                            "%four = arith.constant 2 : i32\n    %columns = arith.constant 4 : i64")
    source = source.replace("[%n], [%one], [%offset] {order = array<i32: 0>}",
                            "[%n, %columns], [%columns, %one], [%offset, %zero] {order = array<i32: 1, 0>}")
    source = source.replace("boundaryCheck = array<i32: 0>", "boundaryCheck = array<i32: 0, 1>")
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["compile_status"] == "REWRITTEN", report
    assert module.verify()
    assert "tensor<3x2x4xf32>" in str(module)
    assert all(access["shape"] == [2, 4] for access in json.loads(report["guard_json"])["accesses"])


def test_disabled_batching_does_not_normalize_original_block_pointers(tmp_path):
    module = parse(tmp_path, block_pointer_ir(2))
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 1, 0)
    assert report["compile_status"] == "DISABLED", report
    assert str(module) == before


def test_selected_block_pointer_is_rejected_without_entering_stock_pass(tmp_path):
    source = block_pointer_ir(1).replace(
        "%v = tt.load %bp", "%yes = arith.constant true\n"
        "    %selected = arith.select %yes, %bp, %out : !tt.ptr<tensor<4xf32>>\n"
        "    %v = tt.load %selected")
    module = parse(tmp_path, source)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report["reason"] == "UNSUPPORTED_BLOCK_POINTER_FLOW", report
    assert str(module) == before
