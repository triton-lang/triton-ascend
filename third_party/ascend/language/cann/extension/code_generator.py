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

"""
Ascend-specific code generation helpers.
"""

__all__ = ["mangle_ty"]


def mangle_ty(ty):
    """
    Replacement implementation for triton.compiler.code_generator.mangle_ty.
    """
    # Lazy imports to avoid circular dependencies at module import time.
    from triton import language
    from triton.extension.buffer.language import core as bl

    # Buffer types are Python-side dtypes; handle them first.
    if isinstance(ty, bl.buffer_type):
        elt = mangle_ty(ty.element_ty)
        shape = "_".join(map(str, ty.shape))
        return f"B{elt}S{shape}S"

    if ty.is_ptr():
        return "P" + mangle_ty(ty.element_ty)
    if ty.is_int():
        SIGNED = language.dtype.SIGNEDNESS.SIGNED
        prefix = "i" if ty.int_signedness == SIGNED else "u"
        return prefix + str(ty.int_bitwidth)
    if ty.is_floating():
        return str(ty)
    if ty.is_block():
        elt = mangle_ty(ty.scalar)
        shape = "_".join(map(str, ty.shape))
        return f"{elt}S{shape}S"
    if ty.is_void():
        return "V"
    raise TypeError(f"Unsupported type {ty}")

