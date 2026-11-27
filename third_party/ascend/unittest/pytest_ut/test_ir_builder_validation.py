from types import SimpleNamespace

import pytest
import triton.language as tl

from triton._C.libtriton import ascend, ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.language.extra.cann.extension import custom_op

pytestmark = pytest.mark.backend("native")


@pytest.fixture
def builder_and_module():
    ctx = ir.context()
    ir.load_dialects(ctx)
    ascend_ir.load_dialects(ctx)
    builder = ascend_ir.ascendnpu_ir_builder(ctx)
    module = builder.create_module()
    fn = builder.get_or_insert_function(module, "test", builder.get_function_ty([], []), "public", False)
    module.push_back(fn)
    builder.set_insertion_point_to_start(fn.add_entry_block())
    yield builder, module


def test_index_select_requires_pointer(builder_and_module):
    builder, _ = builder_and_module
    scalar = builder.get_int32(0)
    with pytest.raises(TypeError, match="src must be pointer type"):
        builder.create_index_select_simd(scalar, scalar, 0, [], [], [], [])


@pytest.mark.parametrize(("stride", "padding", "dilation", "field"), [
    ([1], 0, 1, "stride"),
    (1, [0, 0, 0], 1, "padding"),
    (1, 0, [1, 1, 1, 1], "dilation"),
])
def test_conv2d_parameter_lengths(builder_and_module, stride, padding, dilation, field):
    builder, _ = builder_and_module
    scalar = builder.get_int32(0)
    with pytest.raises(ValueError, match=field):
        builder.create_conv2d(scalar, scalar, None, stride, padding, dilation, 1, builder.get_int32_ty())


def test_conv2d_ir_padding(tmp_path):
    ctx = ir.context()
    ir.load_dialects(ctx)
    ascend_ir.load_dialects(ctx)
    ascend.load_dialects(ctx)
    source = tmp_path / "conv2d.mlir"
    module = """
        module {
          tt.func public @kernel(%x: tensor<1x4x4xf16>, %w: tensor<1x1x1x1xf16>) {
            %result = ascend.conv2d(%x, %w) {dilation = 1 : i64, groups = 1 : i64,
                padding = PADDING, stride = 1 : i64} :
                (tensor<1x4x4xf16>, tensor<1x1x1x1xf16>) -> tensor<1x4x4xf16>
            tt.return
          }
        }
    """
    for padding in ("0 : i64", "array<i32: 0, 0>", "array<i32: 0, 0, 0, 0>"):
        source.write_text(module.replace("PADDING", padding))
        assert ir.parse_mlir_module(str(source), ctx).verify()
    source.write_text(module.replace("PADDING", "array<i32: 0, 0, 0>"))
    with pytest.raises(RuntimeError, match="Parse MLIR file failed"):
        ir.parse_mlir_module(str(source), ctx)


@pytest.mark.parametrize("method", ["create_custom_op", "create_custom_macro_op"])
def test_custom_attributes_for_omitted_operand(builder_and_module, method):
    builder, module = builder_and_module
    scalar = builder.get_int32(1)
    getattr(builder, method)("example", {}, [scalar], [], [{"present": builder.get_unit_attr()}, {}])
    builder.ret([])
    assert "arg_attrs = [{present}]" in str(module)


def test_custom_attributes_follow_flattened_operands(builder_and_module, monkeypatch):
    builder, _ = builder_and_module
    monkeypatch.setattr(custom_op, "_custom_op_registry", {})

    @custom_op.register_custom_op
    class OptionalInputs:
        name = "__builtin_test_optional_inputs"
        core = custom_op.core.CORE.CUBE
        pipe = custom_op.core.PIPE.PIPE_M
        data_layout = {"inputs": custom_op.core.DataLayout.Fractal}
        address_space = {"out": custom_op.core.AddressSpace.L1}

        def __init__(self, inputs, optional=None, out=None):
            pass

    def check_operands(name, attrs, inputs, outputs, arg_attrs):
        assert len(inputs) == 2 and len(outputs) == 1
        assert [set(a) for a in arg_attrs] == [{"data_layout"}, {"data_layout"}, {"address_space"}]
        return []

    semantic = SimpleNamespace(builder=SimpleNamespace(_ascend_builder=builder, create_custom_op=check_operands))
    value = tl.tensor(builder.get_int32(1), tl.int32)
    custom_op.custom_semantic(OptionalInputs.name, (value, value), out=value, _semantic=semantic)
