// RUN: triton-opt --op-classifier --plan-cube-block %s | FileCheck %s

// A packed scope is an atomic cube unit: PlanCubeBlockPass must assign it a
// block id and must NOT descend into its body to assign ids to the inner ops.
module {
  // CHECK-LABEL: func.func @cube_scope_is_atomic(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<4x4xf16> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<4x4xf16>
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK:   linalg.fill ins(%[[CST]] : f16) outs(%[[ALLOC]] : memref<4x4xf16>)
  // CHECK:   %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4x4xf16>
  // CHECK:   scope.return %[[T]] : tensor<4x4xf16>
  // CHECK: } {ssbuffer.block_id = [[ID:[0-9]+]] : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.block_id = [[ID]] : i32, ssbuffer.core_type = "CUBE"} ins(%[[SCOPE]], %arg0
  func.func @cube_scope_is_atomic(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = scope.scope : () -> tensor<4x4xf16> {
      %alloc = memref.alloc() : memref<4x4xf16>
      %fill_val = arith.constant 0.0 : f16
      linalg.fill ins(%fill_val : f16) outs(%alloc : memref<4x4xf16>)
      %t = bufferization.to_tensor %alloc restrict writable : memref<4x4xf16> to tensor<4x4xf16>
      scope.return %t : tensor<4x4xf16>
    } {ssbuffer.core_type = "CUBE", ssbuffer.load}
    %1 = linalg.matmul ins(%0, %arg0 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%arg1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}
