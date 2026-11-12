"""Candidate-only DMA inlining; no change to masks, layouts or F1 policy."""
from types import SimpleNamespace

import pytest
from triton._C.libtriton import ir, ascend, buffer_ir
from triton.backends.ascend import block_batching as batching

pytestmark = pytest.mark.backend("native")


def parse(tmp_path, text):
    context = ir.context()
    ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    ascend.load_dialects(context)
    buffer_ir.load_dialects(context)
    path = tmp_path / "memory_lowering.mlir"
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    assert module.verify()
    return module


@pytest.mark.parametrize("shape,src_layout,dst_layout,expected", [
    ("?x64", "strided<[64, 1], offset: ?>", "strided<[64, 1]>", True),
    ("?x8x16", "strided<[128, 16, 1]>", "strided<[128, 16, 1]>", True),
    ("?", "strided<[1], offset: ?>", "strided<[1]>", True),
    ("?x64", "strided<[128, 1]>", "strided<[64, 1]>", False),
    ("?x64", "strided<[64, 1]>", "strided<[128, 1]>", False),
    ("?x64", "strided<[128, 2]>", "strided<[64, 1]>", False),
    ("?x64", "strided<[?, 1]>", "strided<[64, 1]>", False),
    ("2x?", "strided<[64, 1]>", "strided<[64, 1]>", False),
    ("?x?", "strided<[64, 1]>", "strided<[64, 1]>", False),
    ("2x64", "strided<[64, 1]>", "strided<[64, 1]>", False),
])
def test_copy_layout_analysis_is_read_only(tmp_path, shape, src_layout, dst_layout, expected):
    source = f"memref<{shape}xf32, {src_layout}>"
    dest = f"memref<{shape}xf32, {dst_layout}>"
    module = parse(
        tmp_path, f'''module attributes {{hacc.block_mapping_owner = "block_batching"}} {{
      tt.func public @copy(%src: {source}, %dst: {dest}) {{
        memref.copy %src, %dst : {source} to {dest}
        tt.return
      }}
    }}''')
    original = str(module)
    assert ascend.passes.ttir.block_batching_has_dynamic_contiguous_copy(module) == expected
    assert str(module) == original


@pytest.mark.parametrize("owner,expected", [("block_batching", True), ("legacy", False), (None, False)])
def test_store_analysis_requires_batch_owner(tmp_path, owner, expected):
    attr = f'attributes {{hacc.block_mapping_owner = "{owner}"}}' if owner else ''
    module = parse(
        tmp_path, f'''module {attr} {{
      tt.func public @store(%src: tensor<?x64xf32>, %dst: memref<?x64xf32>) {{
        bufferization.materialize_in_destination %src in writable %dst :
          (tensor<?x64xf32>, memref<?x64xf32>) -> ()
        tt.return
      }}
    }}''')
    original = str(module)
    assert ascend.passes.ttir.block_batching_has_dynamic_contiguous_copy(module) == expected
    assert str(module) == original


def metadata():
    return dict(
        block_batching_factor=2, target=SimpleNamespace(arch="Ascend950PR_9579"), mix_mode="aiv", parallel_mode="simd",
        block_batching=dict(owner="block_batching", cv_mode="disabled",
                            memory_lowering=dict(dynamic_contiguous_copy=True)))


@pytest.mark.parametrize("change",
                         ["none", "baseline", "no_report", "wrong_owner", "cv", "no_copy", "arch", "mix", "simt"])
def test_compile_policy_is_candidate_only(change):
    value = metadata()
    if change == "baseline":
        value["block_batching_factor"] = 1
    if change == "no_report":
        del value["block_batching"]
    if change == "wrong_owner":
        value["block_batching"]["owner"] = "legacy"
    if change == "cv":
        value["block_batching"]["cv_mode"] = "batch"
    if change == "no_copy":
        value["block_batching"]["memory_lowering"]["dynamic_contiguous_copy"] = False
    if change == "arch":
        value["target"].arch = "Ascend910B4"
    if change == "mix":
        value["mix_mode"] = "mix"
    if change == "simt":
        value["parallel_mode"] = "simt"
    calls = []

    def supported(path):
        calls.append(path)
        return True

    options = batching.memory_compile_options(value, "/compiler", supported)
    assert options == (["--enable-lib-call-no-inline=false"] if change == "none" else [])
    assert calls == (["/compiler"] if change == "none" else [])


def test_unsupported_compiler_keeps_candidate_policy_default():
    value = metadata()
    assert batching.memory_compile_options(value, "/compiler", lambda _: False) == []
    assert value["block_batching_factor"] == 2
    assert value["block_batching"]["memory_lowering"]["libcall_inlining"] is False
