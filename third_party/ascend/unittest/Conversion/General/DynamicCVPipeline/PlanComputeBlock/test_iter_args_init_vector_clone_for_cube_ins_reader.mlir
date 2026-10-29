// RUN: triton-opt --op-classifier %s | FileCheck %s

// Regression test for `OpClassifierPass`:
//   A tensor iter_arg whose only body-side compute consumer is a CUBE
//   linalg.matmul reading it as an `ins` operand, while the value yielded
//   back (the recurrence's producer) is a genuine VECTOR compute op
//   (arith.addf recomputing the state every iteration).
//
//   The carried state is VECTOR-owned: CUBE merely reads it cross-core, so
//   the loop must carry the VECTOR version of the init. The init chain is
//   split (CUBE original + VECTOR clone) and the iter_args must bind the
//   VECTOR clone — otherwise the loop is pinned to the CUBE init, the
//   cross-core multi_buffer recurrence is lost and the state degenerates
//   into the initial (zero) placeholder.
//
//   Mirrors chunk_generalized_iplr_delta_rule_fwd_kernel_h's h-state loop:
//     %b_h_51 = scf.for ... iter_args(%b_h_53 = %b_h_49) {
//       %v = linalg.matmul ins(%d, %b_h_53) outs(...)   // CUBE reads h
//       %h_new = arith.addf %loaded, %delta             // VECTOR produces h
//       scf.yield %h_new
//     }
//
//   Distinct from syy.mlir's pattern (test_iter_args_init_cube_classification
//   .mlir): there the CUBE matmul *writes* the iter_arg as `outs` and the
//   yield's def is a region container (scf.if) that the default sweep marks
//   VECTOR even though its body is CUBE — such inits stay CUBE.

module {
  func.func @tensor_iter_arg_vector_yield_cube_ins_reader(
      %d: tensor<16x64xf32>,
      %base: tensor<64x16xf32>,
      %delta: tensor<64x16xf32>) -> tensor<64x16xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32
    %buf = memref.alloc() : memref<64x16xf32>
    %h0 = bufferization.to_tensor %buf restrict writable : memref<64x16xf32> to tensor<64x16xf32>

    // The init chain is split into a CUBE original and a VECTOR clone, and
    // the iter_args must bind the VECTOR clone (NOT the CUBE original,
    // which is left for the cross-core CUBE reads).
    // CHECK: %[[VEC:.*]] = bufferization.to_tensor {{.*}}{ssbuffer.core_type = "VECTOR"}
    // CHECK: bufferization.to_tensor {{.*}}{ssbuffer.core_type = "CUBE"}
    // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[VEC]])
    // The body-side matmul (reading the state as `ins`) stays CUBE.
    // CHECK: linalg.matmul {{.*}}{ssbuffer.core_type = "CUBE"}
    // The recurrence producer stays VECTOR.
    // CHECK: arith.addf {{.*}}{ssbuffer.core_type = "VECTOR"}
    // CHECK: } {ssbuffer.core_type = "VECTOR"}
    // Negative guard: no CUBE_AND_VECTOR residue.
    // CHECK-NOT: {{.*}}{ssbuffer.core_type = "VECTOR_AND_CUBE"}
    %h = scf.for %i = %c0 to %c8 step %c1 iter_args(%h_ = %h0) -> (tensor<64x16xf32>) : i32 {
      %empty = tensor.empty() : tensor<16x16xf32>
      %v = linalg.matmul ins(%d, %h_ : tensor<16x64xf32>, tensor<64x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %h_new = arith.addf %base, %delta : tensor<64x16xf32>
      scf.yield %h_new : tensor<64x16xf32>
    }
    return %h : tensor<64x16xf32>
  }
}
