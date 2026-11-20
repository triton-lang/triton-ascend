import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend


@triton.jit
def overflow_location_kernel(out, index, stride):
    block_start = index * stride
    tl.store(out, block_start)


def test_overflow_check_does_not_inherit_assignment_name(monkeypatch):
    pytest.importorskip("triton.backends.ascend.compiler")
    monkeypatch.setenv("TRITON_DISABLE_LINE_INFO", "0")
    target = GPUTarget("npu", "Ascend950", 0)
    backend = make_backend(target)
    options = backend.parse_options({"compile_mode": "simt_only", "debug": True, "sanitize_overflow": True})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(overflow_location_kernel, {"out": "*i32", "index": "i32", "stride": "i32"})
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)

    multiply_locations = {}
    assertion_locations = []

    def collect(op):
        if op.get_name() == "arith.muli":
            value = op.get_result(0)
            multiply_locations[str(value.get_type())] = value.get_loc()
        if op.get_name() == "tt.assert":
            assertion_locations.append(str(op.get_operand(0).get_loc()))

    module.walk(collect)
    assert '"block_start"' in str(multiply_locations["i32"])
    assert '"block_start"' not in str(multiply_locations["i64"])
    source_location = str(multiply_locations["i32"].without_name())
    assert str(multiply_locations["i64"]) == source_location
    assert assertion_locations == [source_location]
    assert "tt.auto_overflow_assert" in str(module)
