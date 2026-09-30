// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @batch_matmul
// CHECK: %{{.*}} = linalg.batch_matmul ins(%{{.*}} : tensor<2x4x8xf32>, tensor<2x8x4xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK: return %{{.*}} : tensor<2x4x4xf32>
// CHECK-NOT: linalg.batch_matmul{{[[:space:]]+}}indexing_maps{{[[:space:]]}}*=

func.func @batch_matmul(%A: tensor<2x4x8xf32>, %B: tensor<2x8x4xf32>, %C: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
  %0 = linalg.batch_matmul ins(%A, %B : tensor<2x4x8xf32>, tensor<2x8x4xf32>)
                           outs(%C : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %0 : tensor<2x4x4xf32>
}
