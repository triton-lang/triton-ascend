"""TLA TensorView types, helper inlining and the native adapter boundary."""
import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline, lower_offline


@tla.jit
def read_tile(view):
    return tla.load(view, index=(0, ))


@tla.jit
def view_copy(src, dst, size, stride, KIND: tl.constexpr):
    if KIND == "partition":
        source = tla.make_partition_view(src, [size], [stride], [8])
    elif KIND == "strided":
        source = tla.make_strided_view(src, [size], [stride], [8], [4])
    else:
        source = tla.make_gather_scatter_view(src, [size], [stride], [8], [0])
    target = tla.make_partition_view(dst, [8], [1], [8])
    if KIND == "gather":
        values = tla.load(source, index=(tl.arange(0, 8), ))
    else:
        values = read_tile(source)
    tla.store(target, values, index=(0, ))


@pytest.mark.parametrize("kind", ["partition", "strided", "gather"])
def test_view_ir(kind):
    module, ctx = compile_offline(view_copy, {"src": "*fp32", "dst": "*fp32", "size": "i32", "stride": "i32"},
                                  {"KIND": kind})
    text = str(module)
    assert "tv.make_tensor_view" in text and "tv.view_load" in text and "tv.view_store" in text
    text = str(lower_offline(module, ctx))
    assert "tv.view_load" in text and "tv.view_store" in text
    assert "!tt.ptr" not in text and "tt.call" not in text
    assert "strides=[?]" in text
    assert "hivm.func_core_type<AIV>" in text
