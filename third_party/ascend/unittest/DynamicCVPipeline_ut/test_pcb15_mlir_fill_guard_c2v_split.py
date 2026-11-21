"""
Test Case: PCB15 - masked-load fill guard duplicated on the VECTOR side of a
CUBE_AND_VECTOR buffer-chain split (GEN-1571)

[MLIR Validation]

Description:
  A boundary-masked tl.load lowers to
      %alloc = memref.alloc()
      scf.if %cond { linalg.fill ... outs(%alloc) }   // zero-pad guard
      memref.copy %src, %subview(%alloc)
      %t = bufferization.to_tensor %alloc
  When %t feeds both a CUBE consumer (tl.dot, directly and via tl.trans) and a
  VECTOR consumer (tl.sum), OpClassifier marks the buffer chain
  CUBE_AND_VECTOR and handleCubeAndVector splits it so each core computes into
  its own buffer copy. The guard scf.if (result-less) and its fill used to
  stay CUBE_ONLY, so the VECTOR-side buffer copy was never zero-filled and its
  out-of-bounds region read uninitialized memory (GEN-1571: 2D float32
  unaligned shape (31, 63), mismatch=2/31).

  This test verifies the fix: after the DynamicCVPipeline pass the VECTOR
  scope contains a copy of the fill guard (scf.if wrapping a linalg.fill on
  the duplicated alloc).

Kernel structure notes (each element is required to keep the DynamicCV
pipeline from falling back before the split happens):
  - for loop over K: gives the loop a main_loop tag (via the in-loop fixpipe)
  - masked tl.load: produces the alloc + guard fill + copy + to_tensor chain
  - tl.trans(x) feeds the matmul RHS: CUBE consumer of the load buffer, which
    is what triggers the CUBE_AND_VECTOR split of the buffer chain
  - tl.where-filtered x feeds the matmul LHS: VECTOR->CUBE interaction
  - tl.sum(dot_out): CUBE->VECTOR interaction (both directions needed, or
    AnalyzeScope falls back with "One-way CV interaction")
  - tl.sum(x): the VECTOR consumer of the same load buffer (the bug pattern)

Test Cases:
  - PCB15-TC01: float32, BLOCK=16, M=31, K=63 (unaligned boundary shape)
"""

import os
import re
import subprocess
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, make_ttir, ttir_to_linalg, min_dot_size
from triton.backends.ascend import _apply_ascend_patch
import pytest

# Apply Ascend patch to inject hacc.target attribute into MLIR modules.
# Required by bishengir FixpipeOp::verify() which checks isAscend950(moduleOp)
# via the hacc.target attribute; without it, dst=UB fixpipe ops are rejected.
_apply_ascend_patch()


# ============================================================================
# Compile helper: compile Triton Kernel to MLIR (linalg dialect)
# Reference: compile_kernel implementation in test_custom.py
# ============================================================================
def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel function to MLIR in linalg dialect.

    Compile a Triton Kernel to MLIR in linalg dialect for subsequent content validation.

    Args:
        kernel: Triton JIT-compiled kernel function
        signature: argument type signature dict, e.g. {"x_ptr": "*fp32", "n": "i32"}
        constants: constexpr argument dict, e.g. {"BLOCK": 256}

    Returns:
        str: compiled MLIR code string; returns None on compilation failure
    """
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    try:
        options = NPUOptions(arch="Ascend910_9589", enable_dynamic_cv_pipeline=True)
        # Register codegen_fns, including min_dot_size required by tl.dot.
        codegen_fns = {"min_dot_size": min_dot_size(None)}
        ttir = ast_to_ttir(kernel, src, context, options, codegen_fns, {})
        metadata = {
            **options.__dict__,
        }
        # Call make_ttir for TTIR optimization (consistent with the normal compilation path),
        # including key optimization passes such as inliner/canonicalizer/cse/licm/loop_unroll.
        # Without this step, complex kernels (e.g. while loops) will
        # raise RuntimeError: PassManager::run failed during ttir_to_linalg lowering
        ttir = make_ttir(ttir, metadata, options)
        linalg = ttir_to_linalg(ttir, metadata, options, named_ops=True)
        return str(linalg)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        print(ex.stderr.decode())
        print("failed")
        return None


# ============================================================================
# MLIR output configuration
# ============================================================================
# MLIR output directory: mlir_output subdirectory alongside this test file
MLIR_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlir_output")


def _write_mlir_to_file(mlir, filename):
    os.makedirs(MLIR_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(MLIR_OUTPUT_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mlir)


# ============================================================================
# Kernel under test
# ============================================================================
@triton.jit
def pcb15_tc01_kernel(
    x_ptr, out_ptr,
    M, K,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    racc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        offs_k = k0 + tl.arange(0, BLOCK)
        # Boundary-masked load: lowers to alloc + guard scf.if(fill) + copy
        # + to_tensor. With unaligned M/K the guard fill is the only writer
        # of the out-of-bounds rows.
        x = tl.load(x_ptr + offs[:, None] * K + offs_k[None, :],
                    mask=(offs[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0)
        # CUBE consumer of the load buffer (also via the transpose below);
        # this is what makes the buffer chain CUBE_AND_VECTOR.
        rhs = tl.trans(x)
        # VECTOR op whose result feeds the matmul (VECTOR->CUBE interaction).
        xs = tl.where(x > 0.0, x, x * 0.5)
        # CUBE computation; dot_out feeds a reduce below (CUBE->VECTOR).
        d = tl.dot(xs, rhs, input_precision="ieee")
        racc += tl.sum(d, axis=1)
        # VECTOR consumer of the same load buffer: without the fix, the
        # split leaves this side's buffer copy unguarded.
        racc += tl.sum(x, axis=1)
    tl.store(out_ptr + offs, racc)


def _build_pcb15_signature():
    return {
        "x_ptr": "*fp32",
        "out_ptr": "*fp32",
        "M": "i32",
        "K": "i32",
    }


# ============================================================================
# Scope-region extraction helper
# ============================================================================
_VECTOR_SCOPE_RE = re.compile(
    r"scope\.scope : \(\) -> \(\) \{(.*?)\n    \} "
    r"\{hivm\.matmul_limited_in_cube, hivm\.tcore_type = #hivm\.tcore_type<VECTOR>\}",
    re.S,
)

# A guard scf.if whose body is exactly one linalg.fill into a memref alloc
# (the duplicated zero-pad guard this test looks for).
_VECTOR_GUARD_FILL_RE = re.compile(
    r"scf\.if \%\w+ \{\s*\n\s*linalg\.fill ins\([^)]*\) outs\(\%\w+ : "
    r"memref<\d+x\d+xf32>\)",
)


def test_pcb15_tc01():
    """PCB15-TC01: fill guard is duplicated into the VECTOR scope.

    Test steps:
      1. Compile pcb15_tc01_kernel (masked load feeding both tl.dot and
         tl.sum inside a K loop) to MLIR
      2. Verify the DynamicCVPipeline produced VECTOR and CUBE scopes
      3. Verify the VECTOR scope contains the duplicated fill guard
         (scf.if wrapping a linalg.fill on the split buffer's alloc),
         i.e. the zero-fill protecting the out-of-bounds rows of the
         unaligned tile runs on the VECTOR side as well
    """
    signature = _build_pcb15_signature()
    constants = {"BLOCK": 16}

    mlir = compile_kernel(pcb15_tc01_kernel, signature, constants)
    _write_mlir_to_file(mlir, "pcb15_tc01_kernel.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @pcb15_tc01_kernel(" in mlir, \
        "Kernel function definition not found in MLIR code"

    # The pipeline must not have fallen back: both scopes exist.
    assert "hivm.tcore_type = #hivm.tcore_type<VECTOR>" in mlir, \
        "VECTOR scope not found: DynamicCVPipeline fell back or skipped splitting"
    assert "hivm.tcore_type = #hivm.tcore_type<CUBE>" in mlir, \
        "CUBE scope not found: DynamicCVPipeline fell back or skipped splitting"

    vec_match = _VECTOR_SCOPE_RE.search(mlir)
    assert vec_match, "Could not isolate the VECTOR scope region in MLIR code"
    vec_region = vec_match.group(1)

    # The bug: the VECTOR-side copy of the load buffer was never zero-filled.
    # The fix: the guard scf.if (and its fill) is duplicated into the VECTOR
    # scope, so the out-of-bounds region reads zeros instead of uninitialized
    # memory.
    assert _VECTOR_GUARD_FILL_RE.search(vec_region), (
        "VECTOR scope is missing the duplicated fill guard (scf.if wrapping "
        "a linalg.fill on the split buffer): the out-of-bounds rows of the "
        "VECTOR-side buffer copy would read uninitialized memory")
