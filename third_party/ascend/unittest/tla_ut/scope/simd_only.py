"""TLA scope regression: standalone ``tla.scope.simd()`` (no vector wrapper).

IR: 1 load / 1 store / 2 multi-buffers, one outlined ``hivm.vector_function``.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SIGNATURE = {"X_ptr": "*fp32", "O_ptr": "*fp32"}
N = 64


@triton.jit
def simd_only(X_ptr, O_ptr, N: tl.constexpr):
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    x_blk = tla.view(X_ptr, (N, ), strides=(1, ))
    o_blk = tla.view(O_ptr, (N, ), strides=(1, ))
    tla.copy(x_blk, x_ub)
    with tla.scope.simd():  # standalone: not nested in a vector scope
        v = tla.simd.vlds(x_ub[0])
        tla.simd.vsts(o_ub[0], tla.simd.vadd(v, v))
    tla.copy(o_ub, o_blk)


def constexprs():
    return {"N": N}


def test_ir(output_dir=None):
    module, ctx = compile_offline(simd_only, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "hivm.tcore_type<VECTOR>" in text
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=1, stores=1, buffers=2,
                     extra=("hivm.vector_function", "hivm.vf_mode<SIMD>", "hivm.func_core_type<AIV>"))
    assert text.count("hivm.vector_function") == 1, "standalone simd scope is one VF"
    assert 'mix_mode = "aiv"' in text
    print("IR PASS: simd_only (standalone simd scope)")


def test_compile():
    X = torch.randn(N, dtype=torch.float32, device="npu")
    O = torch.empty(N, dtype=torch.float32, device="npu")
    simd_only[(1, )](X, O, enable_tla_mode=True, **constexprs())
    torch.npu.synchronize()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: simd_only artifacts", len(artifacts))


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    print("ALL PASS: simd_only", "ir" if do_ir else "", "compile" if do_compile else "")


if __name__ == "__main__":
    main()
