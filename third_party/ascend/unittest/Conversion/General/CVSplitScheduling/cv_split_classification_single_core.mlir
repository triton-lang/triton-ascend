// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=IR
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// The pass prepares every function before the module-level classifier runs.
// A vector-only loop has no matmul seed, so precheck rejects it before
// classification. The cube-only candidate reaches classification but is then
// rejected because a CV split requires both engines.
// DIAG-LABEL: [cv-split] Function: cube_only
// DIAG: [cv-split] Pre-check accepted candidate loop
// DIAG: [cv-split] Function: vector_only
// DIAG: [cv-split] Pre-check rejected function, skip
// DIAG: [cv-split] Classification: 4C 0V
// DIAG: [cv-split] Loop must contain both CUBE and VECTOR ops, skip
// DIAG: [cv-split] Candidate failed; restoring function and trying next function
// DIAG: [cv-split] No candidate transformed; keeping original IR

// IR-NOT: ssbuffer.core_type
// IR-LABEL: func.func @cube_only
// IR: %[[CUBE_STEP:.*]] = arith.constant 1 : index
// IR: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[CUBE_STEP]] {
// IR-NEXT: %{{.*}} = linalg.matmul
// IR-NEXT: }
// IR-NOT: scope.scope
// IR-LABEL: func.func @vector_only
// IR: %[[VECTOR_STEP:.*]] = arith.constant 1 : index
// IR: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[VECTOR_STEP]] {
// IR-NEXT: %{{.*}} = arith.addf
// IR-NEXT: }
// IR-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @cube_only(%lhs: tensor<32x16xf16>,
                       %rhs: tensor<16x16xf16>,
                       %init: tensor<32x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c1 {
      %result = linalg.matmul ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
    }
    return
  }

  func.func @vector_only(%lhs: tensor<16x16xf32>,
                         %rhs: tensor<16x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c1 {
      %result = arith.addf %lhs, %rhs : tensor<16x16xf32>
    }
    return
  }
}
