// RUN: triton-opt %s | FileCheck %s

// CHECK-LABEL: func.func @linalg_misc(
// CHECK: %[[EMPTY2D:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[EMPTY2D]] : tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK: %[[BROADCAST:.*]] = linalg.broadcast ins(%arg0 : tensor<2xf32>) outs(%[[FILL]] : tensor<2x4xf32>) dimensions = [1]
// CHECK: %[[EMPTYTR:.*]] = tensor.empty() : tensor<4x2xf32>
// CHECK: %[[TRANSPOSE:.*]] = linalg.transpose ins(%[[BROADCAST]] : tensor<2x4xf32>) outs(%[[EMPTYTR]] : tensor<4x2xf32>) permutation = [1, 0]
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: ins(%[[TRANSPOSE]], %[[TRANSPOSE]] : tensor<4x2xf32>, tensor<4x2xf32>)
// CHECK: %[[IDX:.*]] = linalg.index 0 : index
// CHECK: linalg.yield %{{.*}} : f32
// CHECK: %[[EMPTY1D:.*]] = tensor.empty() : tensor<2xf32>
// CHECK: %[[FILL1D:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[EMPTY1D]] : tensor<2xf32>) -> tensor<2xf32>
// CHECK: %[[REDUCE:.*]] = linalg.reduce ins(%[[GENERIC]] : tensor<4x2xf32>) outs(%[[FILL1D]] : tensor<2xf32>) dimensions = [0]
// CHECK: linalg.yield %{{.*}} : f32
// CHECK: return %[[REDUCE]] : tensor<2xf32>
func.func @linalg_misc(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty2d = tensor.empty() : tensor<2x4xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%empty2d : tensor<2x4xf32>) -> tensor<2x4xf32>
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<2xf32>) outs(%filled : tensor<2x4xf32>) dimensions = [1]
  %empty_tr = tensor.empty() : tensor<4x2xf32>
  %transposed = linalg.transpose ins(%broadcasted : tensor<2x4xf32>) outs(%empty_tr : tensor<4x2xf32>) permutation = [1, 0]
  %empty_generic = tensor.empty() : tensor<4x2xf32>
  %generic = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%transposed, %transposed : tensor<4x2xf32>, tensor<4x2xf32>)
    outs(%empty_generic : tensor<4x2xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %i = linalg.index 0 : index
    %sum = arith.addf %in0, %in1 : f32
    linalg.yield %sum : f32
  } -> tensor<4x2xf32>
  %empty1d = tensor.empty() : tensor<2xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty1d : tensor<2xf32>) -> tensor<2xf32>
  %reduced = linalg.reduce ins(%generic : tensor<4x2xf32>) outs(%init : tensor<2xf32>) dimensions = [0]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %reduced : tensor<2xf32>
}
