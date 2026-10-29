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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
``tla.device_assert``: runtime diagnostic (DBG-01, restricted debug profile).

Semantics follow ``tl.device_assert`` (``triton/language/semantic.py``): the
assert is only emitted when the compiler ``options.debug`` flag is set (the
kernel fails on a false condition at runtime and the message survives); under
``debug=False`` the call is a no-op and is not emitted at all.

Pure-buffer constraint (see ``ai/spec/debug.md`` and the topk_gate migration
record): the AVE dialect exposes no scalar-extraction op, so a data-dependent
*i1 scalar* cannot be produced from vector-register results inside the pure
``tla.*`` interface set.  The condition accepted here must therefore be a
host-side scalar i1 (``tl.program_id(0) >= 0``-style or a constant); vector
conditions are rejected explicitly to keep the ``tensor<``-free IR contract.
"""

from triton.language.core import builtin, _unwrap_if_constexpr

__all__ = ["device_assert"]


@builtin
def device_assert(cond, msg="", _semantic=None):
    """
    Assert ``cond`` at runtime; terminate the kernel with ``msg`` on failure.

    :param cond: scalar i1 (0-d) condition; a data-dependent vector-register
        condition is not expressible in the pure-buffer interface set and is
        rejected (see module docstring)
    :param msg: literal diagnostic message
    """
    msg = _unwrap_if_constexpr(msg)
    if not isinstance(msg, str):
        raise TypeError(f"device_assert message must be a string, got {msg!r}")
    cond_t = _semantic.to_tensor(cond)
    if len(cond_t.shape) != 0:
        raise ValueError("tla.device_assert requires a scalar i1 condition; "
                         f"got shape {tuple(cond_t.shape)} "
                         "(the pure-buffer interface set cannot produce a data-dependent "
                         "scalar from AVE vector registers)")
    return _semantic.device_assert(cond_t, msg, None)
