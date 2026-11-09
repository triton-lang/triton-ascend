// RUN: triton-opt --triton-to-linalg --split-input-file %s -verify-diagnostics

// A non-canonical i64 index cannot be retained by the device templates, which
// require an i32 index payload. Keep this coverage separate from the canonical
// widened-iota case in reduce_with_index_annotation.mlir.
module {
  // The reduce below fails as expected, but the pass still repairs locations
  // on the remaining IR. MLIR's linalg.fill builder gives the fill's
  // `linalg.yield` an unknown location, so the warning below comes from the
  // `dense<128>` constant in this module and is unrelated to the diagnostic
  // under test; match it so that -verify-diagnostics does not reject it.
  // expected-warning @unknown {{linalg.yield}}
  func.func @noncanonical_i64_offset_index(%v: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xi64>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %w = arith.extsi %r : tensor<8xi32> to tensor<8xi64>
    %c = arith.constant dense<128> : tensor<8xi64>
    %off = arith.addi %w, %c : tensor<8xi64>
    %e = tt.expand_dims %off {axis = 0 : i32} : tensor<8xi64> -> tensor<1x8xi64>
    %idx = tt.broadcast %e : tensor<1x8xi64> -> tensor<4x8xi64>
    // expected-error @+1 {{reduce-with-index requires an i32 index operand when the index is not a canonical iota}}
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

// -----

module {
  // Even a lossless truncation cannot be retained: the device templates only
  // accept i32/i64 index payloads.
  func.func @noncanonical_i8_truncated_index(%v: tensor<1x8xf32>) -> (tensor<1xf32>, tensor<1xi8>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %n = arith.trunci %r : tensor<8xi32> to tensor<8xi8>
    %idx = tt.expand_dims %n {axis = 0 : i32} : tensor<8xi8> -> tensor<1x8xi8>
    // expected-error @+1 {{reduce-with-index requires an i32 index operand when the index is not a canonical iota}}
    %m:2 = "tt.reduce"(%v, %idx) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %in_index: i8, %init: f32, %init_index: i8):
      %is_greater = arith.cmpf ogt, %in, %init : f32
      %new_value = arith.select %is_greater, %in, %init : f32
      %new_index = arith.select %is_greater, %in_index, %init_index : i8
      tt.reduce.return %new_value, %new_index : f32, i8
    }) : (tensor<1x8xf32>, tensor<1x8xi8>) -> (tensor<1xf32>, tensor<1xi8>)
    return %m#0, %m#1 : tensor<1xf32>, tensor<1xi8>
  }
}

// -----

module {
  func.func @noncanonical_i64_reshaped_index(%v: tensor<2x4xf32>) -> (tensor<2xf32>, tensor<2xi64>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %w = arith.extsi %r : tensor<8xi32> to tensor<8xi64>
    %idx = tt.reshape %w : tensor<8xi64> -> tensor<2x4xi64>
    // expected-error @+1 {{reduce-with-index requires an i32 index operand when the index is not a canonical iota}}
    %m:2 = "tt.reduce"(%v, %idx) <{axis = 1 : i32}> ({
    ^bb0(%in: f32, %in_index: i64, %init: f32, %init_index: i64):
      %is_greater = arith.cmpf ogt, %in, %init : f32
      %new_value = arith.select %is_greater, %in, %init : f32
      %new_index = arith.select %is_greater, %in_index, %init_index : i64
      tt.reduce.return %new_value, %new_index : f32, i64
    }) : (tensor<2x4xf32>, tensor<2x4xi64>) -> (tensor<2xf32>, tensor<2xi64>)
    return %m#0, %m#1 : tensor<2xf32>, tensor<2xi64>
  }
}
