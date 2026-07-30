// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=IR
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// Candidate 1 reaches Stage 8 but has no supported cross-scope transfer.
// Candidate 2 is valid and must still be attempted and committed.

// DIAG-LABEL: [cv-split] Function: first_candidate_fails
// DIAG-LABEL: [cv-split] Function: second_candidate_succeeds
// DIAG: [cv-split] Candidate failed; restoring function and trying next function
// DIAG: [cv-split] Stage 9 complete

// IR-LABEL: func.func @first_candidate_fails
// IR: %[[FIRST_STEP:.*]] = arith.constant 1 : index
// IR: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[FIRST_STEP]] {
// IR-NEXT: %{{.*}} = linalg.matmul
// IR-NEXT: %{{.*}} = math.exp
// IR-NEXT: }
// IR-NOT: scope.scope

// IR-LABEL: func.func @second_candidate_succeeds
// IR: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @first_candidate_fails(
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

  func.func @second_candidate_succeeds(
      %lhs_src: memref<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %lhs_buffer = memref.alloc() : memref<32x16xf16>
    memref.copy %lhs_src, %lhs_buffer : memref<32x16xf16> to memref<32x16xf16>
    %lhs = bufferization.to_tensor %lhs_buffer restrict writable :
        memref<32x16xf16>
    scf.for %iv = %c0 to %c16 step %c1 {
      %matmul = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %vector = math.exp %matmul : tensor<32x16xf32>
    }
    return
  }
}
