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
"""Compatibility dispatcher; new code can use load/store/copy route APIs."""
from triton.language.core import builtin
from . import dma

_COPY_ROUTES = {
    ("gm", "ub"): dma.load_gm_to_ub,
    ("ub", "gm"): dma.store_ub_to_gm,
    ("gm", "l1"): dma.load_gm_to_l1,
    ("ub", "l1"): dma.copy_ub_to_l1,
    ("l1", "ub"): dma.copy_l1_to_ub,
    ("l1", "bt"): dma.copy_l1_to_bt,
    ("l1", "fb"): dma.copy_l1_to_fb,
    ("l1", "l0a"): dma.copy_l1_to_l0,
    ("l1", "l0b"): dma.copy_l1_to_l0,
    ("l0c", "gm"): dma.store_l0c_to_gm,
    ("l0c", "ub"): dma.copy_l0c_to_ub,
    ("l0c", "l1"): dma.copy_l0c_to_l1,
}
_FIXPIPE_PARAMS = {
    "gm": dma.FixpipeL0C2GMParams,
    "ub": dma.FixpipeL0C2UBParams,
    "l1": dma.FixpipeL0C2L1Params,
}


@builtin
def copy(src, dst, params=None, _semantic=None):
    dma._buffers(src, dst)
    route = _COPY_ROUTES.get((src.space, dst.space))
    if route is None:
        raise ValueError(f"tla.copy unsupported space pair {(src.space, dst.space)}")
    if src.space == "l0c" and dma._unwrap_if_constexpr(params) is None:
        mode = dma.FixpipeDMAMode.NZ2ND if len(dst.shape) == 2 else dma.FixpipeDMAMode.NZ2NZ
        params = _FIXPIPE_PARAMS[dst.space](dma_mode=mode)
    route(src, dst, params, _semantic=_semantic)
