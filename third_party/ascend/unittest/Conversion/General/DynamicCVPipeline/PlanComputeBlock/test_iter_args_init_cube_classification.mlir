// RUN: triton-opt --op-classifier %s | FileCheck %s

// Regression test for `OpClassifierPass`:
//   When a scf.for loop has its iter_args init consumed as the `outs` of a
//   CUBE linalg.matmul inside the body, but the loop's yielded value flows
//   into a VECTOR arith op outside the loop, the init must NOT be wrongly
//   classified as VECTOR_AND_CUBE.
//
//   Before the fix, `propagateVectorUpstream` propagated VECTOR onto the init
//   through `scf.for` operands; the init then split into a CUBE original
//   and a VECTOR clone -- and a faulty `getForInitCoreType` bound the
//   iter_args to the VECTOR clone while the body still needed the CUBE
//   original (which became dead code).
//
//   After the fix:
//     1. `propagateVectorUpstream` skips iter_args init defining ops.
//     2. `getForInitCoreType` classifies the init by its body-side compute
//        consumers (here: a CUBE linalg.matmul), so the init mirrors the
//        body and stays CUBE-only. The outside VECTOR arith consumer does
//        not leak back into the init.
//
//   Mirrors syy.mlir's %3 pattern: a `linalg.fill` init consumed as the
//   outs of a CUBE matmul inside the scf.for body, with the yielded value
//   feeding a VECTOR `arith.mulf` after the loop.

module {
  // CHECK-LABEL: func.func @iter_args_init_stays_cube_with_outer_vector_consumer
  // The init's `tensor.empty` must be classified CUBE-only -- matches the
  // body-side CUBE consumer, NOT split with VECTOR.
  // CHECK: tensor.empty() {ssbuffer.core_type = "CUBE"}
  // The init's `linalg.fill` must be classified CUBE-only.
  // CHECK: linalg.fill {{.*}}{ssbuffer.core_type = "CUBE"}
  // The body-side matmul must stay CUBE-only.
  // CHECK: linalg.matmul {{.*}}{ssbuffer.core_type = "CUBE"}
  // The outside-loop VECTOR consumer must stay VECTOR-only (does NOT pollute
  // the iter_args init).
  // CHECK: arith.mulf {{.*}}{ssbuffer.core_type = "VECTOR"}
  // Negative guard: no clone of the init should have been produced for the
  // VECTOR consumer. Before the fix the init was marked CUBE_AND_VECTOR and
  // got cloned; the VECTOR clone routed the iter_args off the CUBE original.
  // CHECK-NOT: linalg.fill {{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-NOT: linalg.fill {{.*}}{ssbuffer.core_type = "VECTOR_AND_CUBE"}
  // CHECK-NOT: tensor.empty() {{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-NOT: tensor.empty() {{.*}}{ssbuffer.core_type = "VECTOR_AND_CUBE"}
  func.func @iter_args_init_stays_cube_with_outer_vector_consumer(
      %a: tensor<16x64xf16>,
      %b: tensor<64x32xf16>,
      %extra: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<16x32xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<16x32xf32>) -> tensor<16x32xf32>

    %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %init) -> (tensor<16x32xf32>) {
      // Body-side CUBE consumer: matmul writes `outs(%acc)`.
      %mm = linalg.matmul ins(%a, %b : tensor<16x64xf16>, tensor<64x32xf16>)
          outs(%acc : tensor<16x32xf32>) -> tensor<16x32xf32>
      scf.yield %mm : tensor<16x32xf32>
    }

    // Outside-loop VECTOR consumer: must NOT propagate onto `%init`.
    // (Mirrors syy.mlir's `%86 = arith.mulf %85, %53`.)
    %outside = arith.mulf %result, %extra : tensor<16x32xf32>
    return %outside : tensor<16x32xf32>
  }
}
