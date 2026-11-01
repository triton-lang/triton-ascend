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
"""Host-side integration of ``tla.buffer`` kernel parameters.

A ``@tla.jit`` kernel may declare a parameter as a TLA buffer instead of a raw
pointer::

    @tla.jit
    def kernel(x: tla.buffer_type((64, 128), tla.float32), ...):
        tile = tla.alloc_buffer((64, 128), tla.float32)
        tla.copy(x, tile)

The host still passes a torch tensor.  ``@tla.jit`` validates it against the
annotation and replaces the parameter's specialization entry with
``tlabuf:<space>:<dtype>:<shape>:<strides>:<offset>``.  The compiler decodes
that entry back into a :class:`tla.buffer_type`, and the launcher maps it onto
a pointer argument, so the device ABI is the same as for ``*dtype`` pointer
parameters.  Declaring the buffer statically also puts shape, strides, and
dtype into the specialization key, so differently shaped tensors compile
separate kernels.

Only GM buffers can come from the host: on-chip spaces are allocated inside
the kernel with ``tla.alloc_buffer``.
"""
from triton.language import core as tl

from .core import buffer, buffer_type, _row_major_strides

SPEC_PREFIX = "tlabuf:"
HOST_SPACE = "gm"

# Value/library dtype spellings -> canonical Triton dtype names.
_DTYPE_ALIASES = {
    "bfloat16": "bf16",
    "bool": "int1",
    "double": "fp64",
    "float": "fp32",
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "half": "fp16",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
}


def is_buffer_annotation(annotation):
    """True when a kernel parameter annotation asks for a TLA buffer."""
    return isinstance(annotation, buffer_type) or annotation is buffer


def is_buffer_spec(value):
    """True when ``value`` is a serialized TLA buffer signature entry."""
    return isinstance(value, str) and value.startswith(SPEC_PREFIX)


def buffer_spec(ty):
    """Serialize ``ty`` into the specialization signature entry."""
    if not isinstance(ty, buffer_type):
        raise TypeError(f"expected a tla.buffer_type, got {type(ty).__name__}")
    if ty.offset is None:
        raise ValueError("a host tla.buffer_type must have a static offset")
    shape = "x".join(str(size) for size in ty.shape)
    strides = "x".join(str(stride) for stride in ty.strides)
    return f"{SPEC_PREFIX}{ty.space}:{ty.element_ty.name}:{shape}:{strides}:{ty.offset}"


def parse_buffer_spec(spec):
    """Rebuild the :class:`tla.buffer_type` encoded by :func:`buffer_spec`."""
    if not is_buffer_spec(spec):
        raise ValueError(f"not a TLA buffer signature: {spec!r}")
    space, dtype_name, shape, strides, offset = spec[len(SPEC_PREFIX):].split(":")
    shape = [int(size) for size in shape.split("x")]
    strides = [int(stride) for stride in strides.split("x")]
    offset = int(offset)
    return buffer_type(shape, tl.dtype(dtype_name), space, strides, offset)


def _dtype_of(value, name):
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise TypeError(f"tla buffer argument {name!r} has no dtype; pass a tensor-like value")
    if isinstance(dtype, tl.dtype):
        return dtype
    text = dtype if isinstance(dtype, str) else getattr(dtype, "name", None) or str(dtype)
    text = _DTYPE_ALIASES.get(text.rsplit(".", 1)[-1], text.rsplit(".", 1)[-1])
    if not tl.dtype.is_dtype(text):
        raise TypeError(f"tla buffer argument {name!r} has unsupported dtype {dtype!r}")
    return tl.dtype(text)


def _shape_of(value, name):
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"tla buffer argument {name!r} has no shape; pass a tensor-like value")
    shape = tuple(int(size) for size in shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"tla buffer argument {name!r} must be non-empty, got shape {shape}")
    return shape


def _strides_of(value, name, shape):
    stride = getattr(value, "stride", None)
    if stride is None:
        return _row_major_strides(shape)
    stride = stride() if callable(stride) else stride
    return tuple(int(item) for item in stride)


def resolve_buffer_arg(value, annotation, name):
    """Validate a host value against a ``tla.buffer`` annotation.

    Returns the :class:`tla.buffer_type` the kernel is compiled with.  A
    ``tla.buffer_type`` annotation is checked against the value; a bare
    ``tla.buffer`` annotation derives the type from the value.
    """
    dtype = _dtype_of(value, name)
    shape = _shape_of(value, name)
    if isinstance(annotation, buffer_type):
        ty = annotation
        if ty.space != HOST_SPACE:
            raise ValueError(f"tla buffer argument {name!r} must use the {HOST_SPACE!r} space; "
                             f"on-chip buffers are allocated with tla.alloc_buffer, got {ty.space!r}")
        if ty.offset is None or ty.offset != 0:
            raise ValueError(f"tla buffer argument {name!r} cannot carry a view offset; "
                             f"pass the base tensor instead")
        if ty.element_ty != dtype:
            raise TypeError(f"tla buffer argument {name!r} expects {ty.element_ty}, got {dtype}")
        if ty.shape != shape:
            raise ValueError(f"tla buffer argument {name!r} expects shape {ty.shape}, got {shape}")
    else:
        ty = buffer_type(shape, dtype, HOST_SPACE)
    strides = _strides_of(value, name, shape)
    if strides != ty.strides:
        raise ValueError(f"tla buffer argument {name!r} must be contiguous with row-major strides "
                         f"{ty.strides}, got {strides}")
    return ty
