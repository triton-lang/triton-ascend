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


def _fixpipe_dma_mode(dst):
    # Match InsertLoadStoreForMixCV::getInsertedFixpipeDmaMode: a 2D ND dst
    # uses NZ2ND (drop NZ to ND); a 4D fractal dst keeps NZ (NZ2NZ).
    # TODO: Replace it with explicit fixpipe API with mode.
    return "nz2nd" if len(dst.shape) == 2 else "normal"


def _is_zn_fractal(nz, nd):
    """Whether 4-D ``nz`` is the zN-fractal expansion of 2-D ``nd``.

    Only the layout-transform DMAs need this relation: ``nd2nz`` (gm->l1)
    writes the fractal of an ND source and ``fixpipe`` (l0c->gm) reads it back
    into ND; plain load/store require identical shapes instead.  npuir gives
    ND2NZOp/NZ2NDOp no verifier and FixpipeOp::verify only checks the dma_mode
    and pre-quant contract, so the fractal relation is validated here.
    """
    if len(nz) != 4 or len(nd) != 2:
        return False
    b0, b1 = nz[2], nz[3]
    if b0 <= 0 or b1 <= 0:
        return False
    return nd[0] % b0 == 0 and nd[1] % b1 == 0 and \
        nz[0] == nd[1] // b1 and nz[1] == nd[0] // b0


@builtin
def copy(src, dst, _semantic=None):
    if not isinstance(src, buffer) or not isinstance(dst, buffer):
        raise TypeError("tla.copy expects two tla.buffer operands; pointer/tensor arguments are unsupported")
    # Only the nd2nz / fixpipe paths may pair ND with its zN fractal; other
    # DMAs need identical shapes.
    nd2nz = ((src.space, dst.space) == ("gm", "l1") and _is_zn_fractal(dst.shape, src.shape))
    nz2nd = (src.space == "l0c" and _is_zn_fractal(src.shape, dst.shape))
    if (src.shape != dst.shape and not (nd2nz or nz2nd)) or src.dtype != dst.dtype:
        raise ValueError("DMA source and destination must have matching shape and element type")
    if (src.space, dst.space) == ("gm", "ub"):
        _semantic.builder.create_hivm_load(src.handle, dst.handle)
    elif (src.space, dst.space) == ("ub", "gm"):
        _semantic.builder.create_hivm_store(src.handle, dst.handle)
    elif (src.space, dst.space) == ("gm", "l1"):
        _semantic.builder.create_hivm_nd2nz(src.handle, dst.handle)
    elif src.space == "l0c":
        _semantic.builder.create_hivm_fixpipe(src.handle, dst.handle, _fixpipe_dma_mode(dst))
    else:
        raise ValueError(f"tla.copy unsupported space pair {(src.space, dst.space)}")
