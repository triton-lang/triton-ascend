// RUN: triton-opt %s | FileCheck %s
// Input: LLVM 23 parse (requires `to tensor<...>`). Output: LLVM 20 print (no `to type`).
// CHECK-LABEL: func.func @to_tensor
// CHECK: %{{.*}} = bufferization.to_tensor %{{.*}} restrict writable : memref<4xf32>
// CHECK: return %{{.*}} : tensor<4xf32>
// CHECK-NOT: to tensor<

func.func @to_tensor(%m: memref<4xf32>) -> tensor<4xf32> {
  %0 = bufferization.to_tensor %m restrict writable : memref<4xf32> to tensor<4xf32>
  return %0 : tensor<4xf32>
}
