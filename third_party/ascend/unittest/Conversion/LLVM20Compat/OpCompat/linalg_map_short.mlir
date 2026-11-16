// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @map_short
// CHECK: %{{.*}} = linalg.map { arith.addf }
// CHECK-SAME: ins(%{{.*}} : tensor<64xf32>, tensor<64xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<64xf32>)
// CHECK: return %{{.*}} : tensor<64xf32>

func.func @map_short(%lhs: tensor<64xf32>, %rhs: tensor<64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
  %0 = linalg.map { arith.addf }
      ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
      outs(%init : tensor<64xf32>)
  return %0 : tensor<64xf32>
}
