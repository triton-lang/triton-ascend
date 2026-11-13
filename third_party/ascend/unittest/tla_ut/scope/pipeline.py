"""Pipeline scope placement and native stage grouping."""
import pytest
import triton.language as tl
import triton.language.extra.tla as tla
from triton.compiler.errors import CompilationError

from utils.tla_test_utils import Options, compile_offline, lower_offline


@tla.jit
def staged_copy(X, Y, count, NESTED: tl.constexpr, CARRY: tl.constexpr):
    index = tl.program_id(0)
    for i in tla.pipeline(count):
        scratch = tla.alloc((2, 64), tl.float32, space="ub")
        for row in tl.static_range(2):
            with tla.scope.vector().task(f"load_{row}", stage=0):
                tile = tla.subview(scratch, (row, 0), (1, 64))
                tla.copy(tla.view(X + i * 128 + row * 64, (1, 64)), tile)
        if NESTED:
            for j in tl.range(count):
                with tla.scope.vector().task("nested", stage=1):
                    tla.copy(scratch, tla.view(Y + j * 128, (2, 64)))
        with tla.scope.vector().task("store", stage=1):
            tla.copy(scratch, tla.view(Y + (index + i) * 128, (2, 64)))
        if CARRY:
            index += 1


SIGNATURE = {"X": "*fp32", "Y": "*fp32", "count": "i32"}


@pytest.mark.parametrize("count,enabled,marked", [(None, None, True), (None, False, False), (1, None, False)])
def test_ir(count, enabled, marked):
    constants = {"NESTED": False, "CARRY": False}
    if count is not None:
        constants["count"] = count
    options = Options()
    options.enable_tla_cv_pipelining = enabled
    module, ctx = compile_offline(staged_copy, SIGNATURE, constants, options=options)
    text = str(lower_offline(module, ctx))
    assert text.count("hivm.cv_pipelined_loop") == int(marked)
    assert text.count("hivm.preload_num") == (2 if marked else 0)


@pytest.mark.parametrize("nested,carry,error", [
    (True, False, "direct children"),
    (False, True, "loop-carried values"),
])
def test_invalid_pipeline(nested, carry, error):
    with pytest.raises(CompilationError, match=error):
        compile_offline(staged_copy, SIGNATURE, {"NESTED": nested, "CARRY": carry})
