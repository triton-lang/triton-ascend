"""TLA scope: two sibling ``tla.scope.simd()`` scopes inside one vector scope.

Nesting robustness: the vector scope hosts DMA + multiple independent SIMD
VFs, each outlined separately.  IR: 2 loads / 2 stores / 4 multi-buffers,
exactly two ``hivm.vector_function`` VFs.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SIGNATURE = {"X_ptr": "*fp32", "W_ptr": "*fp32", "O1_ptr": "*fp32", "O2_ptr": "*fp32"}
N = 64


@triton.jit
def multi_vf(X_ptr, W_ptr, O1_ptr, O2_ptr, N: tl.constexpr):
    x_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    w_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o1_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o2_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    x_blk = tla.view(X_ptr, (N, ), strides=(1, ))
    w_blk = tla.view(W_ptr, (N, ), strides=(1, ))
    o1_blk = tla.view(O1_ptr, (N, ), strides=(1, ))
    o2_blk = tla.view(O2_ptr, (N, ), strides=(1, ))
    with tla.scope.vector():
        tla.copy(x_blk, x_ub)
        tla.copy(w_blk, w_ub)
        with tla.scope.simd():  # VF 1: o1 = x + x
            x = tla.simd.vlds(x_ub[0])
            tla.simd.vsts(o1_ub[0], tla.simd.vadd(x, x))
        with tla.scope.simd():  # VF 2: o2 = w * w
            w = tla.simd.vlds(w_ub[0])
            tla.simd.vsts(o2_ub[0], tla.simd.vmul(w, w))
        tla.copy(o1_ub, o1_blk)
        tla.copy(o2_ub, o2_blk)


def constexprs():
    return {"N": N}


def test_ir(output_dir=None):
    module, ctx = compile_offline(multi_vf, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "hivm.tcore_type<VECTOR>" in text
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=2, stores=2, buffers=4, extra=("hivm.tcore_type<VECTOR>", "hivm.vector_function"))
    assert text.count("hivm.vector_function") == 2, "two sibling simd scopes -> two VFs"
    assert 'mix_mode = "aiv"' in text
    print("IR PASS: multi_vf (two nested SIMD VFs in one vector scope)")


def test_compile():
    X = torch.randn(N, dtype=torch.float32, device="npu")
    W = torch.randn(N, dtype=torch.float32, device="npu")
    O1 = torch.empty(N, dtype=torch.float32, device="npu")
    O2 = torch.empty(N, dtype=torch.float32, device="npu")
    multi_vf[(1, )](X, W, O1, O2, enable_tla_mode=True, **constexprs())
    torch.npu.synchronize()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: multi_vf artifacts", len(artifacts))


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    print("ALL PASS: multi_vf", "ir" if do_ir else "", "compile" if do_compile else "")


if __name__ == "__main__":
    main()
