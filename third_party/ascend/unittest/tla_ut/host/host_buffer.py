# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Host-side ``tla.buffer`` kernel parameters.

The annotation is validated against the tensor passed at launch, serialized
into the specialization signature, decoded by the TLA frontend and mapped onto
a pointer argument by the launcher.  These tests cover the host contract and
the offline IR; the device run needs an NPU.
"""
import pytest
import triton.language as tl
from triton.language.extra import tla
from triton.language.extra.tla.host import (buffer_spec, is_buffer_annotation, is_buffer_spec, parse_buffer_spec,
                                            resolve_buffer_arg)
from triton.language.extra.tla.jit import TLAJITFunction

from utils.tla_test_utils import compile_offline, lower_offline


class FakeTensor:
    """Minimal tensor-like value: dtype, shape, row-major strides, data pointer."""

    def __init__(self, dtype, shape, strides=None):
        self.dtype = dtype
        self.shape = tuple(shape)
        if strides is None:
            strides, stride = [], 1
            for size in reversed(self.shape):
                strides.insert(0, stride)
                stride *= size
        self._strides = tuple(strides)

    def stride(self):
        return self._strides

    def data_ptr(self):
        return 0x1000


CASE = tla.buffer_type((2, 32), tl.float32)


@pytest.mark.parametrize("ty", [
    tla.buffer_type((64, ), tl.float32),
    tla.buffer_type((2, 32), tl.bfloat16),
    tla.buffer_type((3, 5, 7), tl.int8, strides=(35, 7, 1)),
    tla.buffer_type((16, ), tl.uint64, offset=0),
])
def test_spec_roundtrip(ty):
    spec = buffer_spec(ty)
    assert is_buffer_spec(spec)
    assert parse_buffer_spec(spec) == ty


def test_spec_layout():
    assert buffer_spec(tla.buffer_type((2, 32), tl.float32)) == "tlabuf:gm:fp32:2x32:32x1:0"
    assert not is_buffer_spec("*fp32")
    with pytest.raises(ValueError, match="not a TLA buffer signature"):
        parse_buffer_spec("*fp32")


def test_annotation_detection():
    assert is_buffer_annotation(CASE)
    assert is_buffer_annotation(tla.buffer)
    assert not is_buffer_annotation(tl.float32)
    assert not is_buffer_annotation(None)


def test_resolve_matches_annotation():
    value = FakeTensor("fp32", (2, 32))
    assert resolve_buffer_arg(value, CASE, "x") == CASE
    assert resolve_buffer_arg(FakeTensor(tl.float32, (2, 32)), CASE, "x") == CASE


def test_resolve_derives_bare_buffer():
    ty = resolve_buffer_arg(FakeTensor("bfloat16", (4, 8)), tla.buffer, "x")
    assert ty == tla.buffer_type((4, 8), tl.bfloat16, "gm")


def test_resolve_rejects_mismatch():
    with pytest.raises(TypeError, match="expects fp32, got int32"):
        resolve_buffer_arg(FakeTensor("int32", (2, 32)), CASE, "x")
    with pytest.raises(ValueError, match="expects shape"):
        resolve_buffer_arg(FakeTensor("fp32", (2, 16)), CASE, "x")
    with pytest.raises(ValueError, match="must be contiguous"):
        resolve_buffer_arg(FakeTensor("fp32", (2, 32), strides=(1, 2)), CASE, "x")
    with pytest.raises(ValueError, match="'gm' space"):
        resolve_buffer_arg(FakeTensor("fp32", (2, 32)), tla.buffer_type((2, 32), tl.float32, "ub"), "x")
    with pytest.raises(ValueError, match="view offset"):
        resolve_buffer_arg(FakeTensor("fp32", (2, 32)), tla.buffer_type((2, 32), tl.float32, offset=None), "x")


def test_binder_rewrites_buffer_specialization():

    def kernel(x: tla.buffer_type((2, 32), tl.float32)):
        pass

    fn = TLAJITFunction(kernel)
    annotations = fn._buffer_annotations()
    assert annotations == {"x": kernel.__annotations__["x"]}
    value = FakeTensor("fp32", (2, 32))
    binder = fn._wrap_binder(lambda *args, **kwargs: (dict(x=args[0]), [("*fp32", "D")], "options"), annotations)
    bound, specialization, options = binder(value)
    assert bound["x"] is value
    assert specialization == [("tlabuf:gm:fp32:2x32:32x1:0", None)]
    assert options == "options"


def test_launcher_expands_buffer_specs():
    from triton.backends.ascend.launcher import argument_types

    spec = buffer_spec(tla.buffer_type((64, ), tl.int32))
    assert argument_types({"x": spec}) == argument_types({"x": "*i32"})


@TLAJITFunction
def _host_copy(src: tla.buffer_type((2, 32), tl.float32), dst: tla.buffer_type((2, 32), tl.float32)):
    ub = tla.alloc((2, 32), tl.float32)
    tla.load_gm_to_ub(src, ub, tla.LoadGM2UBParams())
    tla.store_ub_to_gm(ub, dst, tla.StoreUB2GMParams())


@tla.jit
def _first_kernel(x):
    pass


@tla.jit
def _second_kernel(x):
    pass


def test_jit_decorator_survives_lazy_import():
    assert isinstance(_first_kernel, TLAJITFunction)
    assert isinstance(_second_kernel, TLAJITFunction)


def test_ir_from_spec_signature(output_dir=None):
    signature = {
        "src": buffer_spec(tla.buffer_type((2, 32), tl.float32)),
        "dst": buffer_spec(tla.buffer_type((2, 32), tl.float32)),
    }
    module, ctx = compile_offline(_host_copy, signature, {}, output_dir=output_dir)
    text = str(module)
    assert "tt.func public @_host_copy(%src: memref<2x32xf32, #hivm.address_space<gm>>" in text
    assert "%dst: memref<2x32xf32, #hivm.address_space<gm>>" in text
    assert "tt.ptr" not in text
    lowered = str(lower_offline(module, ctx, output_dir=output_dir))
    assert lowered.count("hivm.hir.load ") == 1 and lowered.count("hivm.hir.store ") == 1
