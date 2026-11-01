"""TLA scope: ``tla.scope.vector()`` hosting only GM<->UB copies (no simd).

IR: 1 load / 1 store / 1 multi-buffer, AIV entry, no ``hivm.vector_function``.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SIGNATURE = {"X_ptr": "*fp32", "O_ptr": "*fp32"}
N = 64


@tla.jit
def copy_only(X_ptr, O_ptr, N: tl.constexpr):
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    x_blk = tla.view(X_ptr, (N, ), strides=(1, ))
    o_blk = tla.view(O_ptr, (N, ), strides=(1, ))
    with tla.scope.vector():
        tla.copy(x_blk, x_ub)  # hivm.hir.load
        tla.copy(x_ub, o_blk)  # hivm.hir.store


def constexprs():
    return {"N": N}


def test_ir(output_dir=None):
    module, ctx = compile_offline(copy_only, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "hivm.tcore_type<VECTOR>" in text, "vector scope must stamp tcore_type<VECTOR>"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=1, stores=1, buffers=None,
                     extra=("hivm.tcore_type<VECTOR>", "hivm.func_core_type<AIV>"))
    assert text.count("hivm.vector_function") == 0, "vector scope is not a VF"
    assert 'mix_mode = "aiv"' in text, "pure-vector kernel is aiv"
    print("IR PASS: copy_only (DMA-only vector scope)")


def test_compile():
    X = torch.randn(N, dtype=torch.float32, device="npu")
    O = torch.empty(N, dtype=torch.float32, device="npu")
    copy_only[(1, )](X, O, **constexprs())
    torch.npu.synchronize()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: copy_only artifacts", len(artifacts))


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    print("ALL PASS: copy_only", "ir" if do_ir else "", "compile" if do_compile else "")


if __name__ == "__main__":
    main()
