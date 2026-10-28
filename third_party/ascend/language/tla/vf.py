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
"""Core-mode scopes for TLA (``with tla.scope.simd():``).

A scope marks a block of register ops (``simd.*`` on the vector core) that the
compiler emits as an inline ``scope.scope`` carrying ``hivm.vector_function``
and the per-argument buffer ``hivm.memory_effect`` summary (read/write on the
UB buffers); the bishengir ``outline-scope`` pass extracts the marked scope
into the ``func.func`` + ``func.call`` that the HIVM stage consumes.

``tla.scope.simd()`` and ``tla.scope.cube()`` are the entry points; each is a
context-manager class (the core mode is fixed on the class), so the compiler
dispatch table keys off the class object itself.
"""

__all__ = ["scope", "simd", "cube"]


class scope:
    """Base marker for a core-mode region.

    Subclasses fix the core type (:class:`simd` -> vector core,
    :class:`cube` -> cube core). The Triton code generator routes ``with``
    statements whose context expression evaluates to one of these classes to
    ``code_generator.handle_scope_with`` via ``TLA_WITH_DISPATCH``, so the
    classes are never instantiated.
    """

    core_mode = "vector"


class simd(scope):
    """``with tla.scope.simd():`` -- SIMD vector-function region (vector core)."""

    core_mode = "vector"


class cube(scope):
    """``with tla.scope.cube():`` -- cube-core region (reserved; cube ops not yet exposed)."""

    core_mode = "cube"


# ``tla.scope`` resolves as the namespace holding the fixed-core-mode classes.
scope.simd = simd
scope.cube = cube
