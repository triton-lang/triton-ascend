// RUN: triton-opt %s | FileCheck %s

// CHECK-LABEL: func.func @tensor_misc(
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[C2:.*]] = arith.constant 2 : i64
// CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty() : tensor<1x2xf32>
// CHECK-NEXT: %[[FROM:.*]] = tensor.from_elements %arg0, %arg1 : tensor<2xf32>
// CHECK-NEXT: %[[SHAPE:.*]] = tensor.from_elements %[[C1]], %[[C2]] : tensor<2xi64>
// CHECK-NEXT: %[[RESHAPE:.*]] = tensor.reshape %[[FROM]](%[[SHAPE]]) : (tensor<2xf32>, tensor<2xi64>) -> tensor<1x2xf32>
// CHECK-NEXT: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[RESHAPE]] {{\[\[0, 1\]\]}} : tensor<1x2xf32> into tensor<2xf32>
// CHECK-NEXT: %[[EXPAND:.*]] = tensor.expand_shape %[[COLLAPSE]] {{\[\[0, 1\]\]}} output_shape [1, 2] : tensor<2xf32> into tensor<1x2xf32>
// CHECK-NEXT: %[[EXTRACT:.*]] = tensor.extract_slice %[[EXPAND]][0, 0] [1, 1] [1, 1] : tensor<1x2xf32> to tensor<1x1xf32>
// CHECK-NEXT: %[[INSERT_A:.*]] = tensor.insert_slice %[[EXTRACT]] into %[[EMPTY]][0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x2xf32>
// CHECK-NEXT: %[[SPLAT:.*]] = tensor.splat %arg0 : tensor<1x1xf32>
// CHECK-NEXT: %[[INSERT_B:.*]] = tensor.insert_slice %[[SPLAT]] into %[[INSERT_A]][0, 1] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x2xf32>
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<1x2xf32>
// CHECK-NEXT: bufferization.materialize_in_destination %[[INSERT_B]] in writable %[[ALLOC]] : (tensor<1x2xf32>, memref<1x2xf32>) -> ()
// CHECK-NEXT: memref.dealloc %[[ALLOC]] : memref<1x2xf32>
// CHECK-NEXT: return
func.func @tensor_misc(%arg0: f32, %arg1: f32) {
  %c1 = arith.constant 1 : i64
  %c2 = arith.constant 2 : i64
  %empty = tensor.empty() : tensor<1x2xf32>
  %from = tensor.from_elements %arg0, %arg1 : tensor<2xf32>
  %shape = tensor.from_elements %c1, %c2 : tensor<2xi64>
  %reshape = tensor.reshape %from(%shape) : (tensor<2xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  %collapsed = tensor.collapse_shape %reshape [[0, 1]] : tensor<1x2xf32> into tensor<2xf32>
  %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [1, 2] : tensor<2xf32> into tensor<1x2xf32>
  %extracted = tensor.extract_slice %expanded[0, 0] [1, 1] [1, 1] : tensor<1x2xf32> to tensor<1x1xf32>
  %insert_a = tensor.insert_slice %extracted into %empty[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x2xf32>
  %splat = tensor.splat %arg0 : tensor<1x1xf32>
  %insert_b = tensor.insert_slice %splat into %insert_a[0, 1] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x2xf32>
  %alloc = memref.alloc() : memref<1x2xf32>
  bufferization.materialize_in_destination %insert_b in writable %alloc : (tensor<1x2xf32>, memref<1x2xf32>) -> ()
  memref.dealloc %alloc : memref<1x2xf32>
  return
}
