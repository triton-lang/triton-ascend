// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// The index operand of a max-with-index reduction must be annotated with
// `UseIndexInput` so that LinalgToHFusion keeps passing it through to
// hfusion.reduce_with_index. Without the annotation the index tensor is
// dropped and the downstream reduce_*_with_index template synthesises
// 0..chunk-1 local indices, which is wrong once the reduction axis is split
// (R0_BLOCK < rnumel): values stay correct but the returned indices are
// chunk-local instead of global.

module {
  func.func @max_with_index_annotates_index_input(%logits: tensor<2x4xf32>, %indices: tensor<2x4xi32>) -> (tensor<2xf32>, tensor<2xi32>) {
    %max:2 = "tt.reduce"(%logits, %indices) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %in_index: i32, %init: f32, %init_index: i32):
      %is_greater = arith.cmpf ogt, %in, %init : f32
      %new_value = arith.select %is_greater, %in, %init : f32
      %new_index = arith.select %is_greater, %in_index, %init_index : i32
      tt.reduce.return %new_value, %new_index : f32, i32
    }) : (tensor<2x4xf32>, tensor<2x4xi32>) -> (tensor<2xf32>, tensor<2xi32>)
    return %max#0, %max#1 : tensor<2xf32>, tensor<2xi32>
  }
}

// CHECK-LABEL: func.func @max_with_index_annotates_index_input
// CHECK: annotation.mark %{{.*}} {UseIndexInput} : tensor<2x4xi32>
// CHECK: linalg.reduce
// CHECK-SAME: reduce_mode = "max_with_index"

// -----

// min-with-index must be annotated the same way.
module {
  func.func @min_with_index_annotates_index_input(%logits: tensor<2x4xf32>, %indices: tensor<2x4xi32>) -> (tensor<2xf32>, tensor<2xi32>) {
    %min:2 = "tt.reduce"(%logits, %indices) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %in_index: i32, %init: f32, %init_index: i32):
      %is_less = arith.cmpf olt, %in, %init : f32
      %new_value = arith.select %is_less, %in, %init : f32
      %new_index = arith.select %is_less, %in_index, %init_index : i32
      tt.reduce.return %new_value, %new_index : f32, i32
    }) : (tensor<2x4xf32>, tensor<2x4xi32>) -> (tensor<2xf32>, tensor<2xi32>)
    return %min#0, %min#1 : tensor<2xf32>, tensor<2xi32>
  }
}

// CHECK-LABEL: func.func @min_with_index_annotates_index_input
// CHECK: annotation.mark %{{.*}} {UseIndexInput} : tensor<2x4xi32>
// CHECK: linalg.reduce
// CHECK-SAME: reduce_mode = "min_with_index"

// -----

// A plain (non-index) reduction must NOT gain the annotation.
module {
  func.func @plain_reduce_has_no_index_annotation(%in: tensor<2x4xf32>) -> tensor<2xf32> {
    %sum = "tt.reduce"(%in) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %add = arith.addf %a, %b : f32
      tt.reduce.return %add : f32
    }) : (tensor<2x4xf32>) -> tensor<2xf32>
    return %sum : tensor<2xf32>
  }
}

// CHECK-LABEL: func.func @plain_reduce_has_no_index_annotation
// CHECK-NOT: annotation.mark
// CHECK: linalg.reduce
