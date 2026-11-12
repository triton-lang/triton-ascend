// RUN: triton-opt --canonicalize --normalize-outer-main-loop %s | FileCheck %s

// Canonicalization removes the one-iteration inner loop before normalization.
// There is no remaining mixed CV inner loop, so use the ordinary pipeline.
// CHECK-LABEL: func.func @folded_single_iteration
// CHECK: scf.for
// CHECK-NOT: scf.for
// CHECK: linalg.matmul
// CHECK: arith.negf
// CHECK-NOT: scf.for
// CHECK-NOT: ssbuffer.
// CHECK: return
func.func @folded_single_iteration(%count: index, %a: tensor<4x4xf32>,
    %b: tensor<4x4xf32>, %initial: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %count step %c1
      iter_args(%acc = %initial) -> tensor<4x4xf32> {
    %inner = scf.for %j = %c0 to %c1 step %c1
        iter_args(%value = %acc) -> tensor<4x4xf32> {
      %next = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
                            outs(%value : tensor<4x4xf32>) -> tensor<4x4xf32>
      %post = arith.negf %next : tensor<4x4xf32>
      scf.yield %post : tensor<4x4xf32>
    }
    scf.yield %inner : tensor<4x4xf32>
  } {ssbuffer.outer_main_loop}
  return %result : tensor<4x4xf32>
}
