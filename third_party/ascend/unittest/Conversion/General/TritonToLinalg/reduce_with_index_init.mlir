// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=True" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @max_i8_value_only
// CHECK-DAG: %[[V:.*]] = arith.constant -128 : i8
// CHECK-DAG: %[[I:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[VINIT:.*]] = linalg.fill ins(%[[V]] : i8)
// CHECK: %[[IINIT:.*]] = linalg.fill ins(%[[I]] : i32)
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<i8>, tensor<i32>)
// CHECK-SAME: dimensions = [0]
// CHECK-SAME: reduce_mode = "max_with_index"
// CHECK-SAME: tie_break_left = "true"
// CHECK-SAME: unsigned_src = "false"
func.func @max_i8_value_only(%x: tensor<64xi8>, %indices: tensor<64xi32>) -> i8 {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i8, %ia: i32, %b: i8, %ib: i32):
  %equal = arith.cmpi eq, %a, %b : i8
  %index_better = arith.cmpi slt, %ia, %ib : i32
  %tie = arith.andi %equal, %index_better : i1
  %better = arith.cmpi sgt, %a, %b : i8
  %choose = arith.ori %better, %tie : i1
  %v = arith.select %choose, %a, %b : i8
  %i = arith.select %choose, %ia, %ib : i32
  tt.reduce.return %v, %i : i8, i32
  }) : (tensor<64xi8>, tensor<64xi32>) -> (i8, i32)
  return %r#0 : i8
}

// -----

// CHECK-LABEL: func.func @min_i16_rows
// CHECK-DAG: %[[V:.*]] = arith.constant 32767 : i16
// CHECK-DAG: %[[I:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[VINIT:.*]] = linalg.fill ins(%[[V]] : i16)
// CHECK: %[[IINIT:.*]] = linalg.fill ins(%[[I]] : i32)
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<2xi16>, tensor<2xi32>)
// CHECK-SAME: dimensions = [1]
// CHECK-SAME: reduce_mode = "min_with_index"
// CHECK-SAME: tie_break_left = "true"
// CHECK-SAME: unsigned_src = "false"
func.func @min_i16_rows(%x: tensor<2x64xi16>, %indices: tensor<2x64xi32>) -> (tensor<2xi16>, tensor<2xi32>) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 1 : i32}> ({
  ^bb0(%a: i16, %ia: i32, %b: i16, %ib: i32):
  %equal = arith.cmpi eq, %a, %b : i16
  %index_better = arith.cmpi slt, %ia, %ib : i32
  %tie = arith.andi %equal, %index_better : i1
  %better = arith.cmpi slt, %a, %b : i16
  %choose = arith.ori %better, %tie : i1
  %v = arith.select %choose, %a, %b : i16
  %i = arith.select %choose, %ia, %ib : i32
  tt.reduce.return %v, %i : i16, i32
  }) : (tensor<2x64xi16>, tensor<2x64xi32>) -> (tensor<2xi16>, tensor<2xi32>)
  return %r#0, %r#1 : tensor<2xi16>, tensor<2xi32>
}

// -----

// CHECK-LABEL: func.func @min_u64
// CHECK-DAG: %[[V:.*]] = arith.constant -1 : i64
// CHECK-DAG: %[[I:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[VINIT:.*]] = linalg.fill ins(%[[V]] : i64)
// CHECK: %[[IINIT:.*]] = linalg.fill ins(%[[I]] : i32)
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<i64>, tensor<i32>)
// CHECK-SAME: dimensions = [0]
// CHECK-SAME: reduce_mode = "min_with_index"
// CHECK-SAME: tie_break_left = "true"
// CHECK-SAME: unsigned_src = "true"
func.func @min_u64(%x: tensor<64xi64>, %indices: tensor<64xi32>) -> (i64, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i64, %ia: i32, %b: i64, %ib: i32):
  %equal = arith.cmpi eq, %a, %b : i64
  %index_better = arith.cmpi slt, %ia, %ib : i32
  %tie = arith.andi %equal, %index_better : i1
  %better = arith.cmpi ult, %a, %b : i64
  %choose = arith.ori %better, %tie : i1
  %v = arith.select %choose, %a, %b : i64
  %i = arith.select %choose, %ia, %ib : i32
  tt.reduce.return %v, %i : i64, i32
  }) : (tensor<64xi64>, tensor<64xi32>) -> (i64, i32)
  return %r#0, %r#1 : i64, i32
}

// -----

// CHECK-LABEL: func.func @max_i64_i64_index
// CHECK-DAG: %[[V:.*]] = arith.constant -9223372036854775808 : i64
// CHECK-DAG: %[[I:.*]] = arith.constant 9223372036854775807 : i64
// CHECK: %[[VINIT:.*]] = linalg.fill ins(%[[V]] : i64)
// CHECK: %[[IINIT:.*]] = linalg.fill ins(%[[I]] : i64)
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<i64>, tensor<i64>)
// CHECK-SAME: dimensions = [0]
// CHECK-SAME: reduce_mode = "max_with_index"
// CHECK-SAME: tie_break_left = "true"
// CHECK-SAME: unsigned_src = "false"
func.func @max_i64_i64_index(%x: tensor<64xi64>, %indices: tensor<64xi64>) -> (i64, i64) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i64, %ia: i64, %b: i64, %ib: i64):
  %equal = arith.cmpi eq, %a, %b : i64
  %index_better = arith.cmpi slt, %ia, %ib : i64
  %tie = arith.andi %equal, %index_better : i1
  %better = arith.cmpi sgt, %a, %b : i64
  %choose = arith.ori %better, %tie : i1
  %v = arith.select %choose, %a, %b : i64
  %i = arith.select %choose, %ia, %ib : i64
  tt.reduce.return %v, %i : i64, i64
  }) : (tensor<64xi64>, tensor<64xi64>) -> (i64, i64)
  return %r#0, %r#1 : i64, i64
}

// -----

// CHECK-LABEL: func.func @min_f32
// CHECK: %[[VINIT:.*]] = tensor.extract_slice
// CHECK: %[[IINIT:.*]] = tensor.extract_slice
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<f32>, tensor<i32>)
// CHECK-SAME: dimensions = [0]
// CHECK-SAME: reduce_mode = "min_with_index"
// CHECK-SAME: tie_break_left = "true"
// CHECK-SAME: unsigned_src = "false"
func.func @min_f32(%x: tensor<64xf32>, %indices: tensor<64xi32>) -> (f32, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %ia: i32, %b: f32, %ib: i32):
  %equal = arith.cmpf oeq, %a, %b : f32
  %index_better = arith.cmpi slt, %ia, %ib : i32
  %tie = arith.andi %equal, %index_better : i1
  %better = arith.cmpf olt, %a, %b : f32
  %choose = arith.ori %better, %tie : i1
  %v = arith.select %choose, %a, %b : f32
  %i = arith.select %choose, %ia, %ib : i32
  tt.reduce.return %v, %i : f32, i32
  }) : (tensor<64xf32>, tensor<64xi32>) -> (f32, i32)
  return %r#0, %r#1 : f32, i32
}

// -----

// CHECK-LABEL: func.func @max_f16
// CHECK: %[[VINIT:.*]] = tensor.extract_slice
// CHECK: %[[IINIT:.*]] = tensor.extract_slice
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<f16>, tensor<i32>)
// CHECK-SAME: dimensions = [0]
// CHECK-SAME: reduce_mode = "max_with_index"
// CHECK-SAME: tie_break_left = "true"
// CHECK-SAME: unsigned_src = "false"
func.func @max_f16(%x: tensor<64xf16>, %indices: tensor<64xi32>) -> (f16, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: f16, %ia: i32, %b: f16, %ib: i32):
  %equal = arith.cmpf oeq, %a, %b : f16
  %index_better = arith.cmpi slt, %ia, %ib : i32
  %tie = arith.andi %equal, %index_better : i1
  %better = arith.cmpf ogt, %a, %b : f16
  %choose = arith.ori %better, %tie : i1
  %v = arith.select %choose, %a, %b : f16
  %i = arith.select %choose, %ia, %ib : i32
  tt.reduce.return %v, %i : f16, i32
  }) : (tensor<64xf16>, tensor<64xi32>) -> (f16, i32)
  return %r#0, %r#1 : f16, i32
}

// -----

// CHECK-LABEL: func.func @max_u16_right
// CHECK-DAG: %[[V:.*]] = arith.constant 0 : i16
// CHECK-DAG: %[[I:.*]] = arith.constant -2147483648 : i32
// CHECK: %[[VINIT:.*]] = linalg.fill ins(%[[V]] : i16)
// CHECK: %[[IINIT:.*]] = linalg.fill ins(%[[I]] : i32)
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<i16>, tensor<i32>)
// CHECK-SAME: dimensions = [0]
// CHECK-SAME: reduce_mode = "max_with_index"
// CHECK-SAME: tie_break_left = "false"
// CHECK-SAME: unsigned_src = "true"
func.func @max_u16_right(%x: tensor<64xi16>, %indices: tensor<64xi32>) -> (i16, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i16, %ia: i32, %b: i16, %ib: i32):
  %equal = arith.cmpi eq, %a, %b : i16
  %index_better = arith.cmpi sgt, %ia, %ib : i32
  %tie = arith.andi %equal, %index_better : i1
  %better = arith.cmpi ugt, %a, %b : i16
  %choose = arith.ori %better, %tie : i1
  %v = arith.select %choose, %a, %b : i16
  %i = arith.select %choose, %ia, %ib : i32
  tt.reduce.return %v, %i : i16, i32
  }) : (tensor<64xi16>, tensor<64xi32>) -> (i16, i32)
  return %r#0, %r#1 : i16, i32
}

// -----

// CHECK-LABEL: func.func @max_i8_fast
// CHECK: %[[VINIT:.*]] = tensor.extract_slice
// CHECK: %[[IINIT:.*]] = tensor.extract_slice
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<i8>, tensor<i32>)
// CHECK-SAME: dimensions = [0]
// CHECK-SAME: reduce_mode = "max_with_index"
// CHECK-SAME: tie_break_left = "false"
// CHECK-SAME: unsigned_src = "false"
func.func @max_i8_fast(%x: tensor<64xi8>, %indices: tensor<64xi32>) -> (i8, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i8, %ia: i32, %b: i8, %ib: i32):
  %better = arith.cmpi sgt, %a, %b : i8
  %v = arith.select %better, %a, %b : i8
  %i = arith.select %better, %ia, %ib : i32
  tt.reduce.return %v, %i : i8, i32
  }) : (tensor<64xi8>, tensor<64xi32>) -> (i8, i32)
  return %r#0, %r#1 : i8, i32
}

// -----

// CHECK-LABEL: func.func @min_i32_fast_columns
// CHECK: %[[VINIT:.*]] = tensor.extract_slice
// CHECK: %[[IINIT:.*]] = tensor.extract_slice
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<4xi32>, tensor<4xi32>)
// CHECK-SAME: dimensions = [0]
// CHECK-SAME: reduce_mode = "min_with_index"
// CHECK-SAME: tie_break_left = "false"
// CHECK-SAME: unsigned_src = "false"
func.func @min_i32_fast_columns(%x: tensor<64x4xi32>, %indices: tensor<64x4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %ia: i32, %b: i32, %ib: i32):
  %better = arith.cmpi slt, %a, %b : i32
  %v = arith.select %better, %a, %b : i32
  %i = arith.select %better, %ia, %ib : i32
  tt.reduce.return %v, %i : i32, i32
  }) : (tensor<64x4xi32>, tensor<64x4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
  return %r#0, %r#1 : tensor<4xi32>, tensor<4xi32>
}

// -----

// CHECK-LABEL: func.func @max_i32_fast_rows
// CHECK: %[[VINIT:.*]] = tensor.extract_slice
// CHECK: %[[IINIT:.*]] = tensor.extract_slice
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<4xi32>, tensor<4xi32>)
// CHECK-SAME: dimensions = [1]
// CHECK-SAME: reduce_mode = "max_with_index"
// CHECK-SAME: tie_break_left = "false"
// CHECK-SAME: unsigned_src = "false"
func.func @max_i32_fast_rows(%x: tensor<4x64xi32>, %indices: tensor<4x64xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %ia: i32, %b: i32, %ib: i32):
  %better = arith.cmpi sgt, %a, %b : i32
  %v = arith.select %better, %a, %b : i32
  %i = arith.select %better, %ia, %ib : i32
  tt.reduce.return %v, %i : i32, i32
  }) : (tensor<4x64xi32>, tensor<4x64xi32>) -> (tensor<4xi32>, tensor<4xi32>)
  return %r#0, %r#1 : tensor<4xi32>, tensor<4xi32>
}

// -----

// CHECK-LABEL: func.func @max_i32_reordered
// CHECK-DAG: %[[V:.*]] = arith.constant -2147483648 : i32
// CHECK-DAG: %[[I:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[VINIT:.*]] = linalg.fill ins(%[[V]] : i32)
// CHECK: %[[IINIT:.*]] = linalg.fill ins(%[[I]] : i32)
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<i32>, tensor<i32>)
// CHECK-SAME: dimensions = [0]
// CHECK-SAME: reduce_mode = "max_with_index"
// CHECK-SAME: tie_break_left = "true"
// CHECK-SAME: unsigned_src = "false"
func.func @max_i32_reordered(%x: tensor<64xi32>, %indices: tensor<64xi32>) -> (i32, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %ia: i32, %b: i32, %ib: i32):
  %better = arith.cmpi slt, %a, %b : i32
  %index_better = arith.cmpi slt, %ib, %ia : i32
  %equal = arith.cmpi eq, %b, %a : i32
  %tie = arith.andi %index_better, %equal : i1
  %choose = arith.ori %tie, %better : i1
  %v = arith.select %choose, %b, %a : i32
  %i = arith.select %choose, %ib, %ia : i32
  tt.reduce.return %v, %i : i32, i32
  }) : (tensor<64xi32>, tensor<64xi32>) -> (i32, i32)
  return %r#0, %r#1 : i32, i32
}

// -----

// Preserve the NaN-aware combiner used by Inductor: NaNs win over numbers,
// and two NaNs are tied by index. A floating sentinel is not an identity.
// CHECK-LABEL: func.func @min_f32_nan_aware
// CHECK: %[[VINIT:.*]] = tensor.extract_slice
// CHECK: %[[IINIT:.*]] = tensor.extract_slice
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<f32>, tensor<i32>)
// CHECK-SAME: reduce_mode = "min_with_index"
// CHECK-SAME: tie_break_left = "true"
func.func @min_f32_nan_aware(%x: tensor<64xf32>, %indices: tensor<64xi32>) -> (f32, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %ia: i32, %b: f32, %ib: i32):
    %true = arith.constant true
    %less = arith.cmpf olt, %a, %b : f32
    %equal = arith.cmpf oeq, %a, %b : f32
    %a_nan = arith.cmpf une, %a, %a : f32
    %b_nan = arith.cmpf une, %b, %b : f32
    %b_number = arith.xori %b_nan, %true : i1
    %a_nan_only = arith.andi %a_nan, %b_number : i1
    %better = arith.ori %less, %a_nan_only : i1
    %both_nan = arith.andi %a_nan, %b_nan : i1
    %equal_or_nan = arith.ori %equal, %both_nan : i1
    %left = arith.cmpi slt, %ia, %ib : i32
    %tie = arith.andi %equal_or_nan, %left : i1
    %choose = arith.ori %better, %tie : i1
    %v = arith.select %choose, %a, %b : f32
    %i = arith.select %choose, %ia, %ib : i32
    tt.reduce.return %v, %i : f32, i32
  }) : (tensor<64xf32>, tensor<64xi32>) -> (f32, i32)
  return %r#0, %r#1 : f32, i32
}
