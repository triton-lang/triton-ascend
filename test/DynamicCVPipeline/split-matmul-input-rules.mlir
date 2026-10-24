// RUN: triton-opt %s --ssbuf-split-matmul --verify-each | FileCheck %s
// RUN: triton-opt %s --ssbuf-standardize-op --verify-each > /dev/null

// CHECK-LABEL: func.func @empty_input
// CHECK: %[[EMPTY:.*]] = tensor.empty
// CHECK-NOT: linalg.fill
// CHECK: linalg.matmul {{.*}}outs(%[[EMPTY]]
// CHECK-NOT: arith.addf
// CHECK: return
func.func @empty_input(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = tensor.empty() : tensor<4x4xf32>
  %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @broadcast_input
// CHECK: %[[BIAS:.*]] = linalg.broadcast
// CHECK: linalg.matmul {{.*}}outs(%[[BIAS]]
// CHECK-NOT: arith.addf
// CHECK: return
func.func @broadcast_input(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %bias: tensor<4xf32>) -> tensor<4x4xf32> {
  %empty = tensor.empty() : tensor<4x4xf32>
  %c = linalg.broadcast ins(%bias : tensor<4xf32>) outs(%empty : tensor<4x4xf32>) dimensions = [0]
  %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @nonzero_fill_input
// CHECK: %[[BIAS:.*]] = linalg.fill
// CHECK: linalg.fill
// CHECK: %[[MM:.*]] = linalg.matmul
// CHECK: arith.addf %[[MM]], %[[BIAS]]
func.func @nonzero_fill_input(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %one = arith.constant 1.0 : f32
  %empty = tensor.empty() : tensor<4x4xf32>
  %c = linalg.fill ins(%one : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @broadcast_at_capacity
// CHECK: %[[BIAS:.*]] = linalg.broadcast
// CHECK-NOT: linalg.fill
// CHECK: linalg.matmul {{.*}}outs(%[[BIAS]]
// CHECK-NOT: arith.addf
// CHECK: return
func.func @broadcast_at_capacity(%a: tensor<4x4xf32>, %b: tensor<4x1024xf32>, %bias: tensor<1024xf32>) -> tensor<4x1024xf32> {
  %empty = tensor.empty() : tensor<4x1024xf32>
  %c = linalg.broadcast ins(%bias : tensor<1024xf32>) outs(%empty : tensor<4x1024xf32>) dimensions = [0]
  %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x1024xf32>) outs(%c : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  return %m : tensor<4x1024xf32>
}

// CHECK-LABEL: func.func @broadcast_over_capacity
// CHECK: %[[BIAS:.*]] = linalg.broadcast
// CHECK: %[[ZERO:.*]] = linalg.fill
// CHECK: %[[MM:.*]] = linalg.matmul {{.*}}outs(%[[ZERO]]
// CHECK: arith.addf %[[MM]], %[[BIAS]]
func.func @broadcast_over_capacity(%a: tensor<4x4xf32>, %b: tensor<4x1025xf32>, %bias: tensor<1025xf32>) -> tensor<4x1025xf32> {
  %empty = tensor.empty() : tensor<4x1025xf32>
  %c = linalg.broadcast ins(%bias : tensor<1025xf32>) outs(%empty : tensor<4x1025xf32>) dimensions = [0]
  %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x1025xf32>) outs(%c : tensor<4x1025xf32>) -> tensor<4x1025xf32>
  return %m : tensor<4x1025xf32>
}

// CHECK-LABEL: func.func @broadcast_wrong_dimension
// CHECK: %[[BIAS:.*]] = linalg.broadcast
// CHECK: %[[ZERO:.*]] = linalg.fill
// CHECK: %[[MM:.*]] = linalg.matmul {{.*}}outs(%[[ZERO]]
// CHECK: arith.addf %[[MM]], %[[BIAS]]
func.func @broadcast_wrong_dimension(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %bias: tensor<4xf32>) -> tensor<4x4xf32> {
  %empty = tensor.empty() : tensor<4x4xf32>
  %c = linalg.broadcast ins(%bias : tensor<4xf32>) outs(%empty : tensor<4x4xf32>) dimensions = [1]
  %m = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}
