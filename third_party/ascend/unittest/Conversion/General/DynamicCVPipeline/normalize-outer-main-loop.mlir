// RUN: triton-opt --normalize-outer-main-loop %s | FileCheck %s

// One mixed CV inner loop is a single event domain without partial/full joining.
// Its original seed, tensor recurrence and final store remain guarded per task.
// CHECK-LABEL: func.func @single_cv_loop
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.maxui
// CHECK: arith.minui
// CHECK: arith.cmpi ule
// CHECK: scf.if
// CHECK: scf.for {{.*}} : i64
// CHECK: linalg.matmul
// CHECK: arith.addf
// CHECK: bufferization.materialize_in_destination
// CHECK: ssbuffer.explicit_main_loop
// CHECK-SAME: ssbuffer.outer_event_control_count = 4 : i32
// CHECK: } else {
// CHECK: scf.for
// CHECK: scf.for
// CHECK: ssbuffer.outer_event_fallback
func.func @single_cv_loop(%end: index, %counts: memref<?xi32>,
                          %a: tensor<4x4xf32>, %b: tensor<4x4xf32>,
                          %seed: tensor<4x4xf32>, %out: memref<4x4xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %zero_i32 = arith.constant 0 : i32
  %one_i32 = arith.constant 1 : i32
  scf.for %task = %zero to %end step %one {
    %count = memref.load %counts[%task] : memref<?xi32>
    %result = scf.for %j = %zero_i32 to %count step %one_i32
        iter_args(%state = %seed) -> tensor<4x4xf32> : i32 {
      %product = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
          outs(%seed : tensor<4x4xf32>) -> tensor<4x4xf32>
      %next = arith.addf %state, %product : tensor<4x4xf32>
      scf.yield %next : tensor<4x4xf32>
    }
    bufferization.materialize_in_destination %result in writable %out
        : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
  } {ssbuffer.outer_main_loop}
  return
}
