"""TLA scope: GM->L1 ``tla.copy`` (hivm.hir.nd2nz) inside ``tla.scope.vector()``.

Motivating case: a non-SIMD DMA whose destination is L1 sits in the vector
scope.  IR: no load/store, one ``hivm.hir.nd2nz``, one multi-buffer (L1).
Fragment kernel (no GM output), IR-only.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import get_stages, compile_offline, lower_offline, check_lowered_ir

SIGNATURE = {"A_ptr": "*fp16"}
M = 64
K = 64


@tla.jit
def gm2l1(A_ptr, M: tl.constexpr, K: tl.constexpr):
    # 4-D NZ physical shape for the L1 tile (fractal of the 2-D ND view).
    a_l1 = tla.alloc_buffer((K // 16, M // 16, 16, 16), tl.float16, space="l1")
    a_blk = tla.view(A_ptr, (M, K), strides=(K, 1))
    with tla.scope.vector():
        tla.copy(a_blk, a_l1)  # hivm.hir.nd2nz (GM->L1) in a vector scope


def constexprs():
    return {"M": M, "K": K}


def test_ir(output_dir=None):
    module, ctx = compile_offline(gm2l1, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "hivm.tcore_type<VECTOR>" in text, "vector scope must stamp tcore_type<VECTOR>"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=0, stores=0, buffers=None,
                     extra=("hivm.tcore_type<VECTOR>", "hivm.hir.nd2nz", "hivm.func_core_type<AIV>"))
    assert text.count("hivm.hir.nd2nz") == 1, "one GM->L1 nd2nz in the vector scope"
    assert text.count("hivm.vector_function") == 0, "no simd scope, no VF"
    assert 'mix_mode = "aiv"' in text
    print("IR PASS: gm2l1 (GM->L1 nd2nz in vector scope)")


def main():
    if get_stages()[0]:
        test_ir()
    print("ALL PASS: gm2l1 ir")


if __name__ == "__main__":
    main()
