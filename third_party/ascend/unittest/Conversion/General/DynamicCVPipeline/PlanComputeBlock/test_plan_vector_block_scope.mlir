// RUN: triton-opt --op-classifier --plan-vector-block %s | FileCheck %s

// A packed scope is an atomic vector unit: PlanVectorBlockPass must assign it a
// block id and must NOT descend into its body to assign ids to the inner ops.
module {
  // CHECK-LABEL: func.func @vector_scope_is_atomic(
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<64xf32> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<64xf32>
  // CHECK:   %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<64xf32>
  // CHECK:   scope.return %[[T]] : tensor<64xf32>
  // CHECK: } {ssbuffer.block_id = [[ID:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR", ssbuffer.load}
  // CHECK: %[[EXP:.*]] = tensor.expand_shape %[[SCOPE]] {{\[\[}}0, 1{{\]\]}} output_shape [8, 8] {ssbuffer.block_id = [[ID]] : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32> into tensor<8x8xf32>
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
