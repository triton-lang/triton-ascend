// RUN: triton-opt --plan-compute-block %s | FileCheck %s

// Packed scopes are atomic compute units: the planner assigns a block id to the
// scope itself and leaves every op inside its body untouched.
module {
  // CHECK-LABEL: func.func @cube_scope_is_atomic(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<4x4xf16> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<4x4xf16>
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK:   linalg.fill ins(%[[CST]] : f16) outs(%[[ALLOC]] : memref<4x4xf16>)
  // CHECK:   %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4x4xf16>
  // CHECK:   scope.return %[[T]] : tensor<4x4xf16>
  // CHECK: } {ssbuffer.block_id = [[CID:[0-9]+]] : i32, ssbuffer.core_type = "CUBE", ssbuffer.load}
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.block_id = [[CID]] : i32, ssbuffer.core_type = "CUBE"} ins(%[[SCOPE]], %arg0
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

  // CHECK-LABEL: func.func @vector_scope_is_atomic(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<64xf32> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<64xf32>
  // CHECK:   %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<64xf32>
  // CHECK:   scope.return %[[T]] : tensor<64xf32>
  // CHECK: } {ssbuffer.block_id = [[VID:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CHECK: %[[EXP:.*]] = tensor.expand_shape %[[SCOPE]] {{\[\[}}0, 1{{\]\]}} output_shape [8, 8] {ssbuffer.block_id = [[VID]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32> into tensor<8x8xf32>
  func.func @vector_scope_is_atomic(%arg0: tensor<64xf32>) -> tensor<8x8xf32> {
    %0 = scope.scope : () -> tensor<64xf32> {
      %alloc = memref.alloc() : memref<64xf32>
      %t = bufferization.to_tensor %alloc restrict writable : memref<64xf32> to tensor<64xf32>
      scope.return %t : tensor<64xf32>
    } {ssbuffer.core_type = "VECTOR", ssbuffer.load}
    %expanded = tensor.expand_shape %0 [[0, 1]] output_shape [8, 8] : tensor<64xf32> into tensor<8x8xf32>
    return %expanded : tensor<8x8xf32>
  }
}
