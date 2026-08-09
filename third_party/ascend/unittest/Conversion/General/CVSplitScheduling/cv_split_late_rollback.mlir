// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=IR
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// All candidates pass classification and Stage 8. The first is rejected by
// Stage 9 because ROW_SPLIT cannot retile a non-splat shaped constant. The
// second reaches the post-transformation verifier, which rejects the stale
// static sizes on a generically retiled tensor.extract_slice. The third is a
// valid candidate and must still be transformed after both earlier candidates
// are restored. triton-opt must exit successfully.

// DIAG: [cv-split] Function: stage9_retiling_failure
// DIAG: [cv-split] Function: verifier_rejects_retile
// DIAG: [cv-split] === Stage 9: scope separation ===
// DIAG: error: VECTOR retiling only supports shaped splat constants
// DIAG: [cv-split] Candidate failed; restoring function and trying next function
// DIAG: [cv-split] Stage 9 complete
// DIAG: error: expected type to be 'tensor<32x16xf32>'
// DIAG: [cv-split] Candidate failed; restoring function and trying next function
// DIAG: [cv-split] Stage 9 complete
// DIAG: [cv-split] Function attributes set on missing_q_staging_candidate

// IR: module attributes {{.*}}hivm.disable_auto_tile_and_bind_subblock
// IR-LABEL: func.func @stage9_retiling_failure
// IR: %[[FIRST_STEP:.*]] = arith.constant 1 : index
// IR: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[FIRST_STEP]] {
// IR-NEXT: %{{.*}} = linalg.matmul
// IR-NEXT: %{{.*}} = math.exp
// IR-NEXT: %{{.*}} = arith.constant dense<
// IR-NEXT: }
// IR-NOT: scope.scope
// IR-LABEL: func.func @verifier_rejects_retile
// IR: %[[SECOND_STEP:.*]] = arith.constant 1 : index
// IR: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[SECOND_STEP]] {
// IR-NEXT: %{{.*}} = linalg.matmul
// IR-NEXT: %{{.*}} = math.exp
// IR-NEXT: %{{.*}} = tensor.extract_slice
// IR-NEXT: }
// IR-NOT: scope.scope
// IR-LABEL: func.func @missing_q_staging_candidate
// IR: %[[THIRD_STEP:.*]] = arith.constant 4 : index
// IR: scope.scope
// IR: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[THIRD_STEP]] {
// IR-NEXT: %{{.*}} = linalg.matmul
// IR: } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}
// IR: scope.scope
// IR: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[THIRD_STEP]] {
// IR: %{{.*}} = math.exp
// IR: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @stage9_retiling_failure(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c1 {
      %matmul = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %vector = math.exp %matmul : tensor<32x16xf32>
      %non_splat = arith.constant dense<[
          0, 1, 2, 3, 4, 5, 6, 7,
          8, 9, 10, 11, 12, 13, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23,
          24, 25, 26, 27, 28, 29, 30, 31
        ]> : tensor<32xi32>
    }
    return
  }

  func.func @verifier_rejects_retile(
      %lhs_src: memref<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>, %wide: tensor<64x16xf32>) {
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
      %slice = tensor.extract_slice %wide[0, 0] [32, 16] [1, 1]
          : tensor<64x16xf32> to tensor<32x16xf32>
    }
    return
  }

  func.func @missing_q_staging_candidate(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c1 {
      %matmul = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %vector = math.exp %matmul : tensor<32x16xf32>
    }
    return
  }

}
