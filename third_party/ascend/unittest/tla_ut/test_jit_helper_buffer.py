"""A @tla.jit helper may take a ``tla.buffer`` parameter and still compile.

Outlining a helper makes it a nested JIT function. Two things must both hold for
its signature to be built:

  * ``call_JitFunction`` must generate the callee with the caller's generator
    class (``type(self)``), not a hardcoded base ``CodeGenerator``;
  * that class must mount the TLA builder in ``__init__``, because the buffer
    parameter type is produced by ``buffer_type.to_ir`` ->
    ``builder.create_tla_buffer_type``, a TLA-builder method.

With only one of them the callee is built by a generator whose builder lacks the
TLA type constructors, and compilation fails with
``AttributeError: 'ir.builder' object has no attribute 'create_tla_buffer_type'``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton.language as tl  # noqa: E402
import triton.language.extra.tla as tla  # noqa: E402
from utils.tla_test_utils import compile_offline  # noqa: E402

SIGNATURE = {"X": "*u32"}


@tla.jit
def _store_zero(buf, n: tl.constexpr):
    """Callee whose parameter is a tla.buffer (memref) value."""
    tla.store_scalar(buf, 0)


@tla.jit
def _kernel(X, N: tl.constexpr):
    with tla.scope.vector():
        buf = tla.alloc((1, ), tl.uint32)
        _store_zero(buf, N)
        tla.store_ub_to_gm(buf, tla.view(X, (1, )))


def test_helper_with_buffer_parameter_compiles():
    module, _ctx = compile_offline(_kernel, SIGNATURE, {"N": 1})
    ttir = str(module)

    # The helper is outlined, not pasted into the kernel ...
    assert ttir.count("tt.func private") == 1, ttir
    assert ttir.count("tt.call") == 1, ttir

    # ... and its signature really carries the buffer parameter, i.e. the TLA
    # builder produced a memref type for it inside the nested generator.
    callee = next(ln for ln in ttir.splitlines() if "tt.func private" in ln)
    assert "_store_zero" in callee, callee
    assert "memref<" in callee, callee
