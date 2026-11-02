// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// A reduce-with-index keeps the user index operand only when the index is not
// a canonical iota, i.e. when its value is not the element position along the
// reduction axis. Otherwise the backend is free to synthesize the index from
// positions and the operand is dropped.

// Canonical: index = broadcast(expand_dims(make_range(0, 8), axis = 0)) over a
// (4, 8) value reduced along axis 1. The iota dimension lands on the reduction
// axis, so no annotation is needed.
// CHECK-LABEL: func.func @canonical_iota_along_reduce_axis
// CHECK-NOT: annotation.mark
// CHECK: linalg.reduce
// CHECK-SAME: dimensions = [1]
// CHECK-SAME: reduce_mode = "max_with_index"
module {
  func.func @canonical_iota_along_reduce_axis(%v: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xi32>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %e = tt.expand_dims %r {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %idx = tt.broadcast %e : tensor<1x8xi32> -> tensor<4x8xi32>
    %m:2 = "tt.reduce"(%v, %idx) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %in_index: i32, %init: f32, %init_index: i32):
      %is_greater = arith.cmpf ogt, %in, %init : f32
      %new_value = arith.select %is_greater, %in, %init : f32
      %new_index = arith.select %is_greater, %in_index, %init_index : i32
      tt.reduce.return %new_value, %new_index : f32, i32
    }) : (tensor<4x8xf32>, tensor<4x8xi32>) -> (tensor<4xf32>, tensor<4xi32>)
    return %m#0, %m#1 : tensor<4xf32>, tensor<4xi32>
  }
}

// Canonical after widening the iota to i64: extsi is value-preserving and must
// not turn the index into a non-canonical one.
// CHECK-LABEL: func.func @canonical_iota_widened_to_i64
// CHECK-NOT: annotation.mark
// CHECK: linalg.reduce
// CHECK-SAME: reduce_mode = "max_with_index"
module {
  func.func @canonical_iota_widened_to_i64(%v: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xi64>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %w = arith.extsi %r : tensor<8xi32> to tensor<8xi64>
    %e = tt.expand_dims %w {axis = 0 : i32} : tensor<8xi64> -> tensor<1x8xi64>
    %idx = tt.broadcast %e : tensor<1x8xi64> -> tensor<4x8xi64>
    %m:2 = "tt.reduce"(%v, %idx) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %in_index: i64, %init: f32, %init_index: i64):
      %is_greater = arith.cmpf ogt, %in, %init : f32
      %new_value = arith.select %is_greater, %in, %init : f32
      %new_index = arith.select %is_greater, %in_index, %init_index : i64
      tt.reduce.return %new_value, %new_index : f32, i64
    }) : (tensor<4x8xf32>, tensor<4x8xi64>) -> (tensor<4xf32>, tensor<4xi64>)
    return %m#0, %m#1 : tensor<4xf32>, tensor<4xi64>
  }
}

// Canonical through a unit-dim reshape: extsi followed by reshaping [8] to
// [1, 8] is equivalent to expand_dims(axis = 0) on the iota, so the index still
// matches the reduction axis. The frontend emits tt.reshape for
// `indices.reshape(1, N)` instead of tt.expand_dims, so this form must be
// accepted rather than treated as non-canonical.
// CHECK-LABEL: func.func @canonical_iota_reshaped_to_row
// CHECK-NOT: annotation.mark
// CHECK: linalg.reduce
// CHECK-SAME: dimensions = [1]
// CHECK-SAME: reduce_mode = "max_with_index"
module {
  func.func @canonical_iota_reshaped_to_row(%v: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xi64>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %w = arith.extsi %r : tensor<8xi32> to tensor<8xi64>
    %e = tt.reshape %w : tensor<8xi64> -> tensor<1x8xi64>
    %idx = tt.broadcast %e : tensor<1x8xi64> -> tensor<4x8xi64>
    %m:2 = "tt.reduce"(%v, %idx) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %in_index: i64, %init: f32, %init_index: i64):
      %is_greater = arith.cmpf ogt, %in, %init : f32
      %new_value = arith.select %is_greater, %in, %init : f32
      %new_index = arith.select %is_greater, %in_index, %init_index : i64
      tt.reduce.return %new_value, %new_index : f32, i64
    }) : (tensor<4x8xf32>, tensor<4x8xi64>) -> (tensor<4xf32>, tensor<4xi64>)
    return %m#0, %m#1 : tensor<4xf32>, tensor<4xi64>
  }
}

// Non-canonical: make_range + 128. The values are 128..135 along the reduction
// axis, so the operand must be annotated and kept.
// CHECK-LABEL: func.func @noncanonical_offset_index
// CHECK: annotation.mark %{{.*}} {UseIndexInput} : tensor<4x8xi32>
// CHECK: linalg.reduce ins(%{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<4x8xi32>)
// CHECK-SAME: reduce_mode = "max_with_index"
module {
  func.func @noncanonical_offset_index(%v: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xi32>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %c = arith.constant dense<128> : tensor<8xi32>
    %off = arith.addi %r, %c : tensor<8xi32>
    %e = tt.expand_dims %off {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %idx = tt.broadcast %e : tensor<1x8xi32> -> tensor<4x8xi32>
    %m:2 = "tt.reduce"(%v, %idx) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %in_index: i32, %init: f32, %init_index: i32):
      %is_greater = arith.cmpf ogt, %in, %init : f32
      %new_value = arith.select %is_greater, %in, %init : f32
      %new_index = arith.select %is_greater, %in_index, %init_index : i32
      tt.reduce.return %new_value, %new_index : f32, i32
    }) : (tensor<4x8xf32>, tensor<4x8xi32>) -> (tensor<4xf32>, tensor<4xi32>)
    return %m#0, %m#1 : tensor<4xf32>, tensor<4xi32>
  }
}

// Axis trap: on a square (8, 8) value reduced along axis 1, an iota inserted
// at axis 1 of a rank-1 tensor becomes axis 0 after expand_dims. The values are
// constant along the reduction axis, hence non-canonical despite the extent
// matching.
// CHECK-LABEL: func.func @iota_on_other_axis
// CHECK: annotation.mark %{{.*}} {UseIndexInput} : tensor<8x8xi32>
// CHECK: linalg.reduce
// CHECK-SAME: dimensions = [1]
module {
  func.func @iota_on_other_axis(%v: tensor<8x8xf32>) -> (tensor<8xf32>, tensor<8xi32>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %e = tt.expand_dims %r {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %idx = tt.broadcast %e : tensor<8x1xi32> -> tensor<8x8xi32>
    %m:2 = "tt.reduce"(%v, %idx) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %in_index: i32, %init: f32, %init_index: i32):
      %is_greater = arith.cmpf ogt, %in, %init : f32
      %new_value = arith.select %is_greater, %in, %init : f32
      %new_index = arith.select %is_greater, %in_index, %init_index : i32
      tt.reduce.return %new_value, %new_index : f32, i32
    }) : (tensor<8x8xf32>, tensor<8x8xi32>) -> (tensor<8xf32>, tensor<8xi32>)
    return %m#0, %m#1 : tensor<8xf32>, tensor<8xi32>
  }
}

// Index that does not come from make_range at all (a kernel argument).
// CHECK-LABEL: func.func @index_from_argument
// CHECK: annotation.mark %{{.*}} {UseIndexInput} : tensor<4x8xi32>
// CHECK: linalg.reduce
// CHECK-SAME: reduce_mode = "max_with_index"
module {
  func.func @index_from_argument(%v: tensor<4x8xf32>, %idx: tensor<4x8xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
    %m:2 = "tt.reduce"(%v, %idx) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %in_index: i32, %init: f32, %init_index: i32):
      %is_greater = arith.cmpf ogt, %in, %init : f32
      %new_value = arith.select %is_greater, %in, %init : f32
      %new_index = arith.select %is_greater, %in_index, %init_index : i32
      tt.reduce.return %new_value, %new_index : f32, i32
    }) : (tensor<4x8xf32>, tensor<4x8xi32>) -> (tensor<4xf32>, tensor<4xi32>)
    return %m#0, %m#1 : tensor<4xf32>, tensor<4xi32>
  }
}
