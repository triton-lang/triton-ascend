// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=IR
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// The loop has both CUBE and VECTOR work, so it reaches transfer discovery,
// but the independent VECTOR chain consumes no matmul result. Stage 8 rejects
// it after unrolling and scheduling; the module transaction must return the
// original loop rather than any partially transformed variant.

// DIAG: [cv-split] === Stage 8: cross-scope transfers ===
// DIAG: [cv-split] Candidate failed; restoring function and trying next function
// DIAG: [cv-split] No candidate transformed; keeping original IR

// IR-NOT: ssbuffer.core_type
// IR-NOT: hivm.disable_auto_tile_and_bind_subblock
// IR-LABEL: func.func @missing_cross_scope_transfer
// IR: %[[STEP:.*]] = arith.constant 1 : index
// IR: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[STEP]] {
// IR-NEXT: %{{.*}} = linalg.matmul
// IR-NEXT: %{{.*}} = math.exp
// IR-NEXT: }
// IR-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @missing_cross_scope_transfer(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>, %vector_input: tensor<32x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c1 {
      %matmul = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %vector = math.exp %vector_input : tensor<32x16xf32>
    }
    return
  }
}
