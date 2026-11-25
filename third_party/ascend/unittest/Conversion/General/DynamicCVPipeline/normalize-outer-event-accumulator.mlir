// RUN: triton-opt --normalize-outer-main-loop %s | FileCheck %s

// CHECK: module attributes {{.*}}ssbuffer.inter_core_buf_count = 1 : i32
// CHECK-LABEL: func.func @join_matmul_recurrence
// CHECK: linalg.matmul
// CHECK-NOT: arith.negf
// CHECK: linalg.matmul
// CHECK: arith.negf
// CHECK: linalg.matmul
// CHECK: ssbuffer.explicit_main_loop
// CHECK-SAME: ssbuffer.outer_event_control_count = 1 : i32
// CHECK: %[[MASK:.*]] = tensor.splat {{.*}} : tensor<4x4xi1>
// CHECK: arith.select %[[MASK]],
// CHECK: ssbuffer.outer_event_fallback
func.func @join_matmul_recurrence(%rows: index, %partial: memref<?xi32>,
                                 %full: memref<?xi32>, %a: tensor<4x4xf32>,
                                 %seed: tensor<4x4xf32>, %out: memref<4x4xf32>) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %zero_f32 = arith.constant 0.0 : f32
  %begin = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %row = %begin to %rows step %step {
    %np = memref.load %partial[%row] : memref<?xi32>
    %p = scf.for %i = %zero to %np step %one iter_args(%acc = %seed) -> tensor<4x4xf32> : i32 {
      %x = linalg.matmul ins(%a, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%seed : tensor<4x4xf32>) -> tensor<4x4xf32>
      %neg = arith.negf %x : tensor<4x4xf32>
      %y = linalg.matmul ins(%a, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%seed : tensor<4x4xf32>) -> tensor<4x4xf32>
      %z = arith.addf %neg, %y : tensor<4x4xf32>
      %next = linalg.matmul ins(%z, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
      scf.yield %next : tensor<4x4xf32>
    } {may_not_exec = true}
    %nf = memref.load %full[%row] : memref<?xi32>
    %f = scf.for %i = %zero to %nf step %one iter_args(%acc = %p) -> tensor<4x4xf32> : i32 {
      %x = linalg.matmul ins(%a, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%seed : tensor<4x4xf32>) -> tensor<4x4xf32>
      %neg = arith.negf %x : tensor<4x4xf32>
      %y = linalg.matmul ins(%a, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%seed : tensor<4x4xf32>) -> tensor<4x4xf32>
      %z = arith.subf %neg, %y : tensor<4x4xf32>
      %next = linalg.matmul ins(%z, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
      scf.yield %next : tensor<4x4xf32>
    } {hivm.matmul_limited_in_cube}
    %tiles = arith.addi %np, %nf : i32
    %nonempty = arith.cmpi sgt, %tiles, %zero : i32
    %empty = linalg.fill ins(%zero_f32 : f32) outs(%f : tensor<4x4xf32>) -> tensor<4x4xf32>
    %result = arith.select %nonempty, %f, %empty : tensor<4x4xf32>
    bufferization.materialize_in_destination %result in writable %out : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
  } {ssbuffer.outer_main_loop}
  return
}
