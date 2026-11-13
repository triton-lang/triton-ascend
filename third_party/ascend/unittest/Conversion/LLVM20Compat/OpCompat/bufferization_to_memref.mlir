// RUN: triton-opt %s | FileCheck %s
// Patch restores to_memref (23 upstream renamed to to_buffer).
// CHECK-LABEL: func.func @to_memref
// CHECK: %{{.*}} = bufferization.to_memref %{{.*}} : memref<4xf32>
// CHECK: return %{{.*}} : memref<4xf32>
// CHECK-NOT: bufferization.to_buffer

func.func @to_memref(%t: tensor<4xf32>) -> memref<4xf32> {
  %0 = bufferization.to_memref %t : memref<4xf32>
  return %0 : memref<4xf32>
}
