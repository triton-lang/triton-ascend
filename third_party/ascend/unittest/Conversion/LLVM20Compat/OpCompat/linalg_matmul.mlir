// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @matmul
// CHECK: %{{.*}} = linalg.matmul ins(%{{.*}} : tensor<4x8xf32>, tensor<8x4xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: return %{{.*}} : tensor<4x4xf32>
// CHECK-NOT: linalg.matmul{{[[:space:]]+}}indexing_maps{{[[:space:]]}}*=

func.func @matmul(%A: tensor<4x8xf32>, %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x4xf32>)
                     outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
