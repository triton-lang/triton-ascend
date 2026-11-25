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

A scope marks a block of ops (``simd.*`` on the vector core) that the compiler
emits as an inline ``scope.scope`` carrying the core-mode markers; the
bishengir ``outline-scope`` pass extracts marked SIMD scopes into the
``func.func`` + ``func.call`` that the HIVM stage consumes.

``tla.scope.vector()``, ``tla.scope.simd()`` and ``tla.scope.cube()`` are the
entry points; each is a context-manager class (the core mode is fixed on the
class), so the compiler dispatch table keys off the class object itself. Each
class defines ``mlir_attrs``, the attribute set stamped on the emitted
``scope.scope``.
"""

from dataclasses import dataclass

from triton.language.core import _unwrap_if_constexpr, builtin

from ._decorator import experimental

__all__ = ["scope", "vector", "simd", "cube", "pipeline"]


def _check_not_in_simd(name, semantic):
    """Reject operations that must execute outside a SIMD VF."""
    context = getattr(getattr(semantic, "builder", None), "_tla_context", None)
    if context is not None and context.vf_mode == "simd":
        raise ValueError(f"{name} cannot be used inside SIMD VF")


class scope:
    """Base marker for a core-mode region.

    Subclasses fix the core type (:class:`simd` -> vector core,
    :class:`cube` -> cube core) and the attribute set for the emitted
    ``scope.scope`` (``mlir_attrs``). The Triton code generator routes ``with``
    statements whose context expression evaluates to one of these classes to
    ``code_generator.handle_scope_with`` via ``TLA_WITH_DISPATCH``, so the
    classes are never instantiated.
    """

    core_mode = "vector"
    vf_mode = None

    @classmethod
    def mlir_attrs(cls, builder):
        """Attributes stamped on the emitted ``scope.scope`` op."""
        return {"hivm.tcore_type": builder.parse_attr("#hivm.tcore_type<VECTOR>")}


@dataclass(frozen=True)
class _TaskScope:
    kind: type
    name: str
    stage: int | None


class _TaskScopeKind(scope):

    @classmethod
    def task(cls, name, *, stage=None):
        """Create a named core scope, optionally scheduled within a pipeline."""
        name = _unwrap_if_constexpr(name)
        if not isinstance(name, str) or not name:
            raise ValueError("task name must be a nonempty compile-time string")
        stage = _unwrap_if_constexpr(stage)
        if stage is not None and (type(stage) is not int or not 0 <= stage < 64):
            raise ValueError("task stage must be a compile-time integer in [0, 64)")
        return _TaskScope(cls, name, stage)


class vector(_TaskScopeKind):
    """``with tla.scope.vector():`` -- vector-core region for non-SIMD ops.

    A vector scope is the vector core's inline region: it hosts non-SIMD
    vector-core ops (``tla.load_gm_to_ub``, ``tla.store_ub_to_gm``, control flow) and can nest
    :class:`simd` scopes for the register work. Only ``hivm.tcore_type<VECTOR>``
    is stamped (no VF markers), so ``outline-scope`` never extracts it;
    ``InlineScope(forceInline=true)`` folds it back at the end of the HIVM
    pipeline.
    """

    core_mode = "vector"


class simd(scope):
    """``with tla.scope.simd():`` -- SIMD vector-function region (vector core).

    Emitted as a manual vector function: the bishengir ``outline-scope`` pass
    extracts this marked scope into ``func.func`` + ``func.call``. See
    :meth:`mlir_attrs` for the role of each stamped attribute.
    Pure VV kernels may omit the explicit Vector scope; Vector ownership remains.
    """

    core_mode = "vector"
    vf_mode = "simd"

    @classmethod
    def mlir_attrs(cls, builder):
        attrs = super().mlir_attrs(builder)
        attrs.update({
            "no_inline": builder.get_unit_attr(),
            "outline": builder.get_unit_attr(),
            "vector_mode": builder.get_string_attr("vec"),
            "hivm.func_core_type": builder.parse_attr("#hivm.func_core_type<AIV>"),
            "hivm.vf_mode": builder.parse_attr("#hivm.vf_mode<SIMD>"),
            "hivm.vector_function": builder.get_unit_attr(),
        })
        return attrs


class cube(_TaskScopeKind):
    """Selects the Cube core (AIC) for a region in a ``@tla.jit`` kernel.

    Use ``with tla.scope.cube():`` without arguments or an ``as`` binding.
    The region emits ``scope.scope`` with ``hivm.tcore_type<CUBE>``. Allocate
    Cube-local buffers and emit Cube operations such as GM-to-L1 ND2NZ,
    ``tla.cube.mmad`` and L0C Fixpipe transfers inside this region.

    .. note::
        This is a compile-time region marker, not a host context manager or
        a synchronization operation. Explicitly synchronize data passed to
        or from a Vector scope, and respect the destination AIV sub-blocks.
        Put SIMD instructions inside ``tla.scope.simd()`` in a Vector region.
        Cube regions are not outlined as AIV vector functions.

    Example inside a ``@tla.jit`` kernel::

        with tla.scope.cube():
            tile = tla.alloc((2, 1, 16, 16), tl.float16, space=tla.AddressSpace.L1)
            tla.load_gm_to_l1(tla.view(X, (16, 32)), tile)
    """

    core_mode = "cube"

    @classmethod
    def mlir_attrs(cls, builder):
        return {"hivm.tcore_type": builder.parse_attr("#hivm.tcore_type<CUBE>")}


# ``tla.scope`` resolves as the namespace holding the fixed-core-mode classes.
scope.vector = vector
scope.simd = simd
scope.cube = cube


@dataclass(frozen=True)
class _PipelineSpec:
    num_iters: object


@experimental
@builtin
def pipeline(num_iters, _semantic=None):
    """Schedule staged Vector and Cube tasks in a @tla.jit kernel entry loop.

    Iteration starts at zero with a unit step. Tasks specify nondecreasing
    stages. Stage s executes logical iteration n-s at tick n. The compiler
    manages synchronization and versions scratch allocated in the loop.
    Pass `npuir_extra_options=["--enable-tla-cv-pipelining=false"]` at
    launch to run the same tasks without native preload.
    """
    return _PipelineSpec(num_iters)
