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
"""Core-mode scopes and compile-time task/stage annotations for TLA.

Tasks specify fixed stages of an explicitly scheduled pipeline. The backend
materializes the schedule, storage versions and synchronization.
"""
from dataclasses import dataclass
from triton.language.core import _unwrap_if_constexpr

__all__ = ["scope", "vector", "simd", "cube", "pipeline"]


@dataclass(frozen=True)
class _TaskScope:
    kind: type
    name: str
    stage: int

    def __post_init__(self):
        if self.kind not in (vector, cube):
            raise ValueError("task() is only supported on scope.vector() and scope.cube()")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("task name must be a nonempty compile-time string")
        if type(self.stage) is not int or not 0 <= self.stage < 64:
            raise ValueError("task stage must be a compile-time integer in [0, 64)")

    def mlir_attrs(self, builder):
        attrs = self.kind.mlir_attrs(builder)
        attrs["hivm.task_name"] = builder.get_string_attr(self.name)
        attrs["hivm.pipeline_stage"] = builder.parse_attr(f"{self.stage} : i32")
        return attrs


class pipeline:
    """Loop marker: ``for i in tla.pipeline(num_iters)`` starts a pipeline domain.

    Uses a zero lower bound and unit step. Every direct vector/cube scope must
    specify task(name, stage=...). Stages are nondecreasing in source order;
    stage s executes logical iteration n-s at tick n. There is no order hint.

    May be inside ordinary for loops, but not another pipeline, a while loop
    or a core scope. Conditional placement is not validated here. Tasks remain direct children;
    computation loops (e.g. SIMD reductions) belong inside a task.

    Allocate iteration-local scratch directly in the pipeline: its stage span
    determines buffer versions and each iteration must write before reading.
    Local allocations outside the pipeline retain their enclosing lifetime.
    The current frontend does not prove cross-iteration dependencies for
    external state. Such dependencies must be valid under the chosen schedule;
    scratch versioning and synchronization do not repair an invalid schedule.
    Initialize state before entry; consume it after drain. The pipeline itself
    cannot carry SSA values; ordinary enclosing loops may do so.

    The caller guarantees initialization and GM iteration independence,
    including through aliases. The compiler manages scratch reuse and all
    asynchronous synchronization, including pipeline entry/exit dependencies.
    """

    def __init__(self, num_iters):
        self.num_iters = num_iters

    def __iter__(self):
        raise RuntimeError("tla.pipeline can only be used in a @tla.jit for loop")


class scope:
    """Base marker for a core-mode region.

    Subclasses fix the core type (:class:`simd` -> vector core,
    :class:`cube` -> cube core) and the attribute set for the emitted
    ``scope.scope`` (``mlir_attrs``). The Triton code generator routes ``with``
    statements whose context expression evaluates to one of these classes to
    ``code_generator.handle_scope_with`` via ``TLA_WITH_DISPATCH``, so the
    plain and task-annotated scopes share the same core-mode attributes.
    """

    core_mode = "vector"

    @classmethod
    def mlir_attrs(cls, builder):
        """Attributes stamped on the emitted ``scope.scope`` op."""
        return {"hivm.tcore_type": builder.parse_attr("#hivm.tcore_type<VECTOR>")}


class _TaskScopeKind(scope):
    """Execution-core scopes that can own a pipeline task."""

    @classmethod
    def task(cls, name, *, stage):
        """Name a task and specify its fixed compile-time stage in [0, 64).

        A task may contain several pipes; the scope itself is not a barrier.
        Tasks at the same stage retain source order. SIMD work belongs in an
        inner scope.simd(); this outer task retains the scheduling boundary.
        """
        return _TaskScope(cls, _unwrap_if_constexpr(name), _unwrap_if_constexpr(stage))


class vector(_TaskScopeKind):
    """``with tla.scope.vector():`` -- vector-core region for non-SIMD ops.

    A vector scope is the vector core's inline region: it hosts non-SIMD
    vector-core ops (``tla.copy`` GM<->UB, control flow) and can nest
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
    :meth:`mlir_attrs` for the role of each stamped attribute. SIMD scopes do
    not own tasks; nest them inside ``scope.vector().task(...)`` instead.
    """

    core_mode = "vector"

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
            tile = tla.alloc((2, 1, 16, 16), tl.float16, scope=tla.AddressSpace.L1)
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
