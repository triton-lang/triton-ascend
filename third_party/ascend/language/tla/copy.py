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
"""Buffer-to-buffer DMA. No tensor materialization or bind_buffer annotation."""
from triton.language.core import builtin
from .core import buffer


@builtin
def copy(src, dst, _semantic=None):
    if not isinstance(src, buffer) or not isinstance(dst, buffer):
        raise TypeError("tla.copy expects two tla.buffer operands; pointer/tensor arguments are unsupported")
    if src.shape != dst.shape or src.dtype != dst.dtype:
        raise ValueError("DMA source and destination must have identical shape and element type")
    if (src.space, dst.space) == ("gm", "ub"):
        _semantic.builder.create_hivm_load(src.handle, dst.handle)
    elif (src.space, dst.space) == ("ub", "gm"):
        _semantic.builder.create_hivm_store(src.handle, dst.handle)
    else:
        raise ValueError("tla.copy currently supports only GM->UB and UB->GM")
