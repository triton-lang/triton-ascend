// RUN: triton-opt %s | FileCheck %s

// CHECK-LABEL: func.func @bufferization_misc(
// CHECK-NEXT: %[[ALLOC:.*]] = bufferization.alloc_tensor() : tensor<4xf32>
// CHECK-NEXT: %[[TT:.*]] = bufferization.to_tensor %arg1 restrict writable : memref<4xf32>
// CHECK-NEXT: bufferization.materialize_in_destination %arg0 in writable %arg1 : (tensor<4xf32>, memref<4xf32>) -> ()
// CHECK-NEXT: return %[[ALLOC]] : tensor<4xf32>
func.func @bufferization_misc(%src: tensor<4xf32>, %dst: memref<4xf32>) -> tensor<4xf32> {
  %alloc = bufferization.alloc_tensor() : tensor<4xf32>
  %t = bufferization.to_tensor %dst restrict writable : memref<4xf32> to tensor<4xf32>
  bufferization.materialize_in_destination %src in writable %dst : (tensor<4xf32>, memref<4xf32>) -> ()
  return %alloc : tensor<4xf32>
}
