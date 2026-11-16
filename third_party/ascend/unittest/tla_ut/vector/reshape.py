"""TLA 3-in-1 test: buffer reshape (memref.reshape).

Semantics: rank-only reinterpretation of a contiguous row-major buffer, element
count preserved.  Covers flatten (2D->1D), unflatten (1D->2D), merge+split
(2D->2D) and split (2D->3D); strided sources and element-count mismatches are
rejected at trace time.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM IR assert (2 collapse / 3 expand / 4 load / 4 store /
            4 multibuffer / 6 pointer entries)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from triton.compiler.errors import CompilationError
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# H=4, D=8 -> 32 elements; every target shape below keeps the element count.
SHAPE = dict(H=4, D=8)

SIGNATURE = {"X_ptr": "*bf16", "O_ptr": "*bf16"}


def constexprs():
    return dict(SHAPE)


@tla.jit
def reshape_fwd(X_ptr, O_ptr, H: tl.constexpr, D: tl.constexpr):
    with tla.scope.vector():
        # Branches A/C/D share the same (H, D) GM tile; B reads the flat (H*D,) tile.
        x_tile = tla.view(X_ptr, (H, D), strides=(D, 1), offsets=(0, 0))
        x_flat = tla.view(X_ptr, (H * D, ), strides=(1, ), offsets=(0, ))

        # A: (H, D) -> (H*D,)  flatten
        ub_a = tla.alloc_buffer((H, D), tl.bfloat16, space="ub")
        tla.copy(x_tile, ub_a)
        a = tla.reshape(ub_a, (H * D, ))
        tla.copy(a, tla.view(O_ptr, (H * D, ), strides=(1, ), offsets=(0, )))

        # B: (H*D,) -> (H, D)  unflatten
        ub_b = tla.alloc_buffer((H * D, ), tl.bfloat16, space="ub")
        tla.copy(x_flat, ub_b)
        b = tla.reshape(ub_b, (H, D))
        tla.copy(b, tla.view(O_ptr, (H, D), strides=(D, 1), offsets=(0, 0)))

        # C: (H, D) -> (H//2, 2*D)  split dim0 + merge dim1
        ub_c = tla.alloc_buffer((H, D), tl.bfloat16, space="ub")
        tla.copy(x_tile, ub_c)
        c = tla.reshape(ub_c, (H // 2, 2 * D))
        tla.copy(c, tla.view(O_ptr, (H // 2, 2 * D), strides=(2 * D, 1), offsets=(0, 0)))

        # D: (H, D) -> (H//2, 2, D)  split dim0 into two
        ub_d = tla.alloc_buffer((H, D), tl.bfloat16, space="ub")
        tla.copy(x_tile, ub_d)
        d = tla.reshape(ub_d, (H // 2, 2, D))
        tla.copy(d, tla.view(O_ptr, (H // 2, 2, D), strides=(2 * D, D, 1), offsets=(0, 0, 0)))


@tla.jit
def reshape_strided(X_ptr, O_ptr, H: tl.constexpr, D: tl.constexpr):
    x_ub = tla.alloc_buffer((H, D), tl.bfloat16, space="ub")
    x = tla.view(X_ptr, (H, D), strides=(D, 1), offsets=(0, 0))
    tla.copy(x, x_ub)
    sub = tla.view(x_ub, (H, D // 2), strides=(1, 2), offsets=(0, 0))  # strided tile
    tla.reshape(sub, (H * D // 2, ))


@tla.jit
def reshape_count_mismatch(X_ptr, O_ptr, H: tl.constexpr, D: tl.constexpr):
    x_ub = tla.alloc_buffer((H, D), tl.bfloat16, space="ub")
    x = tla.view(X_ptr, (H, D), strides=(D, 1), offsets=(0, 0))
    tla.copy(x, x_ub)
    tla.reshape(x_ub, (H, 2 * D))  # 32 -> 64


def test_ir(output_dir=None):
    module, ctx = compile_offline(reshape_fwd, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=4, stores=4, buffers=None)
    assert text.count("memref.reshape") == 0, "reshape must decompose to expand/collapse"
    # A (4,8)->(32,): pure collapse; B (32,)->(4,8): pure expand;
    # C (4,8)->(2,16): collapse-to-1D then expand; D (4,8)->(2,2,8) is a pure
    # expand (4 -> 2x2) so triton_to_linalg's canonicalizer folds its
    # collapse+expand pair into a single expand.
    assert text.count("memref.collapse_shape") == 2, "collapses: A, C"
    assert text.count("memref.expand_shape") == 3, "expands: B, C, D"
    # Result-type fingerprints pin each reshape to exactly one op: A/C collapse
    # into 1-D, B expands back to (4,8), C to (2,16), D straight to (2,2,8).
    assert text.count("into memref<32xbf16") == 2, "flatten results: A, C"
    assert text.count("into memref<4x8xbf16") == 1, "unflatten result: B"
    assert text.count("into memref<2x16xbf16") == 1, "merge+split result: C"
    assert text.count("into memref<2x2x8xbf16") == 1, "split result: D"
    # 2 shared in-tiles + 4 out views.
    assert text.count("hivm.hir.pointer_cast") == 6, "tla.view(ptr) x6"
    assert text.count("memref.reinterpret_cast") == 6, "tla.view(ptr) x6"
    print("IR PASS: reshape HIVM structure ok (2 collapse / 3 expand / 4 load / 4 store / 4 multibuffer)")
    # Negative cases: strided source and element-count mismatch must be
    # rejected at trace time (the generator wraps the frontend ValueError in
    # CompilationError, keeping the message).
    rejects = {
        reshape_strided: "contiguous row-major",
        reshape_count_mismatch: "element count mismatch",
    }
    for fn, reason in rejects.items():
        try:
            compile_offline(fn, SIGNATURE, constexprs())
            raise AssertionError(f"{fn.__name__} should raise ValueError")
        except CompilationError as e:
            assert reason in str(e), f"{fn.__name__} rejected for wrong reason: {e}"
    print("NEG PASS: strided / count-mismatch reshape rejected")


def _launch():
    H, D = SHAPE["H"], SHAPE["D"]
    X = torch.arange(H * D, dtype=torch.bfloat16, device="npu").reshape(H, D)
    O = torch.full((H, D), 0.0, dtype=torch.bfloat16, device="npu")
    reshape_fwd[(1, )](X, O, **constexprs())
    torch.npu.synchronize()
    return X, O


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: reshape artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    X, O = _launch()
    ox, oo = X.cpu(), O.cpu()
    ok = torch.equal(oo, ox)
    print("RESHAPE", "OK" if ok else "MISMATCH")
    assert ok, "reshape e2e mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: reshape", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
