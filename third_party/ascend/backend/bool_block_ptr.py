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

"""Ascend block pointers retain bool values while using byte-addressed storage.

Installed through the backend module map, never by patching shared language
functions. All other language attributes delegate to the ordinary frontend.
"""

from triton.language import core


def __getattr__(name):
    import triton.language as tl
    return getattr(tl, name) if hasattr(tl, name) else getattr(core, name)


class _BoolBlockPointerType(core.pointer_type):
    """Byte block pointer carrying bool provenance through loops/JIT calls.

    Keep provenance in the type rather than on a tensor instance. The physical
    element type remains int8 so unmapped core builtins retain their old ABI.
    """

    def mangle(self):
        return "AscendBool" + super().mangle()

    def __eq__(self, other):
        return isinstance(other, _BoolBlockPointerType) and super().__eq__(other)

    def storage_type(self):
        return core.pointer_type(self.element_ty, self.address_space, self.const)


def _is_bool_block_ptr(pointer):
    return isinstance(pointer.type, _BoolBlockPointerType)


def _storage_pointer(pointer):
    return core.tensor(pointer.handle, pointer.type.storage_type())


@core.builtin
def make_block_ptr(base, shape, strides, offsets, block_shape, order, _semantic=None):
    pointer = core.make_block_ptr(base, shape, strides, offsets, block_shape, order, _semantic=_semantic)
    if base.type.element_ty == core.int1:
        ty = pointer.type
        logical_ty = _BoolBlockPointerType(ty.element_ty, ty.address_space, ty.const)
        pointer = core.tensor(pointer.handle, logical_ty)
    return pointer


@core.builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, _semantic=None):
    is_bool = _is_bool_block_ptr(pointer)
    if is_bool:
        pointer = _storage_pointer(pointer)
    value = core.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                      volatile, _semantic=_semantic)
    # Use nonzero comparison, not truncation: e.g. byte 2 also represents true.
    return _semantic.cast(value, core.int1) if is_bool else value


@core.builtin
def store(pointer, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="", _semantic=None):
    if _is_bool_block_ptr(pointer):
        pointer = _storage_pointer(pointer)
        value = _semantic.cast(_semantic.cast(_semantic.to_tensor(value), core.int1), core.int8)
    return core.store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy, _semantic=_semantic)
