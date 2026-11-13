"""Named tasks use the regular Cube and Vector execution scopes."""
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline, lower_offline


@tla.jit
def named_scopes(X, Y):
    l1 = tla.alloc((2, 1, 16, 16), tl.float16, space="l1")
    ub = tla.alloc((16, 32), tl.float16, space="ub")
    with tla.scope.cube().task("load_tile"):
        tla.load_gm_to_l1(tla.view(X, (16, 32)), l1)
    with tla.scope.vector().task("copy_tile"):
        tla.copy(tla.view(X, (16, 32)), ub)
        tla.copy(ub, tla.view(Y, (16, 32)))


def test_ir():
    module, ctx = compile_offline(named_scopes, {"X": "*fp16", "Y": "*fp16"}, {})
    text = str(lower_offline(module, ctx))
    assert "#hivm.tcore_type<CUBE>" in text
    assert "#hivm.tcore_type<VECTOR>" in text
    assert "hivm.hir.nd2nz" in text and "hivm.hir.store" in text
    assert "hivm.preload_num" not in text
