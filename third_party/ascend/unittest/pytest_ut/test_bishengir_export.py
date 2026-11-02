"""Python serialization controls; op printing is tested in LLVM20Compat/OpCompat."""

from types import SimpleNamespace

import pytest
import triton.backends.ascend.compiler as compiler

from triton._C.libtriton import ir

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    # Override the parent's fixture: these tests never execute on a device.
    yield


def test_serialization_does_not_mutate_module(tmp_path):
    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path / "input.mlir"
    path.write_text("""
module {
  tt.func @entry(%arg: i32) -> i8 {
    %r = "arith.trunci"(%arg) {overflowFlags = #arith.overflow<nsw>} : (i32) -> i8
    tt.return %r : i8
  }
}
""")
    module = ir.parse_mlir_module(str(path), context)
    original = str(module)
    assert "overflow" not in original
    assert compiler._serialize_module(module, SimpleNamespace(debug=True)) == original
    assert str(module) == original
    assert module.verify()


@pytest.mark.parametrize("debug,disable_line_info,msdebug,expect_locations", [
    (False, True, False, False),
    (True, True, False, True),
    (False, False, False, True),
    (False, True, True, True),
])
def test_export_respects_debug_line_controls(tmp_path, monkeypatch, debug, disable_line_info, msdebug,
                                             expect_locations):
    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path / "debug.mlir"
    path.write_text('module { tt.func @entry() { tt.return loc("kernel.py":3:1) } }')
    module = ir.parse_mlir_module(str(path), context)
    monkeypatch.setattr(compiler, "_is_debug_line_info_disabled", lambda: disable_line_info)
    monkeypatch.setattr(compiler, "_enable_msdebug", lambda: msdebug)
    text = compiler._serialize_module(module, SimpleNamespace(debug=debug))
    assert ('"kernel.py"' in text) == expect_locations
    assert ("loc(" in text) == expect_locations
    assert '"kernel.py"' in str(module), "export must not erase locations from the module"
