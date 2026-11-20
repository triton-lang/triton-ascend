// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=True" --split-input-file %s | FileCheck %s

// The original failure stores only the i8 value. Both accumulators must still
// be initialized before NPU IR drops the unused index result.
// CHECK-LABEL: func.func @max_i8_value_only
// CHECK-DAG: %[[V:.*]] = arith.constant -128 : i8
// CHECK-DAG: %[[I:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[VINIT:.*]] = linalg.fill ins(%[[V]] : i8)
// CHECK: %[[IINIT:.*]] = linalg.fill ins(%[[I]] : i32)
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<i8>, tensor<i32>)
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

// CHECK-LABEL: func.func @min_i16
// CHECK-DAG: %[[V:.*]] = arith.constant 32767 : i16
// CHECK-DAG: %[[I:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[VINIT:.*]] = linalg.fill ins(%[[V]] : i16)
// CHECK: %[[IINIT:.*]] = linalg.fill ins(%[[I]] : i32)
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<i16>, tensor<i32>)
// CHECK-SAME: reduce_mode = "min_with_index"
// CHECK-SAME: tie_break_left = "true"
// CHECK-SAME: unsigned_src = "false"
func.func @min_i16(%x: tensor<64xi16>, %indices: tensor<64xi32>) -> (i16, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i16, %ia: i32, %b: i16, %ib: i32):
    %equal = arith.cmpi eq, %a, %b : i16
    %index_better = arith.cmpi slt, %ia, %ib : i32
    %tie = arith.andi %equal, %index_better : i1
    %better = arith.cmpi slt, %a, %b : i16
    %choose = arith.ori %better, %tie : i1
    %v = arith.select %choose, %a, %b : i16
    %i = arith.select %choose, %ia, %ib : i32
    tt.reduce.return %v, %i : i16, i32
  }) : (tensor<64xi16>, tensor<64xi32>) -> (i16, i32)
  return %r#0, %r#1 : i16, i32
}

// -----

// CHECK-LABEL: func.func @max_i64_i64_index
// CHECK-DAG: %[[V:.*]] = arith.constant -9223372036854775808 : i64
// CHECK-DAG: %[[I:.*]] = arith.constant 9223372036854775807 : i64
// CHECK: %[[VINIT:.*]] = linalg.fill ins(%[[V]] : i64)
// CHECK: %[[IINIT:.*]] = linalg.fill ins(%[[I]] : i64)
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<i64>, tensor<i64>)
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

// CHECK-LABEL: func.func @min_u64
// CHECK-DAG: %[[V:.*]] = arith.constant -1 : i64
// CHECK-DAG: %[[I:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[VINIT:.*]] = linalg.fill ins(%[[V]] : i64)
// CHECK: %[[IINIT:.*]] = linalg.fill ins(%[[I]] : i32)
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<i64>, tensor<i32>)
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

// CHECK-LABEL: func.func @max_u16
// CHECK-DAG: %[[V:.*]] = arith.constant 0 : i16
// CHECK-DAG: %[[I:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[VINIT:.*]] = linalg.fill ins(%[[V]] : i16)
// CHECK: %[[IINIT:.*]] = linalg.fill ins(%[[I]] : i32)
// CHECK: linalg.reduce ins(
// CHECK-SAME: outs(%[[VINIT]], %[[IINIT]] : tensor<i16>, tensor<i32>)
// CHECK-SAME: reduce_mode = "max_with_index"
// CHECK-SAME: tie_break_left = "true"
// CHECK-SAME: unsigned_src = "true"
func.func @max_u16(%x: tensor<64xi16>, %indices: tensor<64xi32>) -> (i16, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i16, %ia: i32, %b: i16, %ib: i32):
    %equal = arith.cmpi eq, %a, %b : i16
    %index_better = arith.cmpi slt, %ia, %ib : i32
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

// Fast comparisons do not match the specialized pattern. The generic
// converter must remain available after increasing the specialized benefit.
// CHECK-LABEL: func.func @fast_max_i32
// CHECK: linalg.reduce
// CHECK-SAME: reduce_mode = "max_with_index"
// CHECK: return
func.func @fast_max_i32(%x: tensor<64xi32>, %indices: tensor<64xi32>) -> (i32, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %ia: i32, %b: i32, %ib: i32):
    %choose = arith.cmpi sgt, %a, %b : i32
    %v = arith.select %choose, %a, %b : i32
    %i = arith.select %choose, %ia, %ib : i32
    tt.reduce.return %v, %i : i32, i32
  }) : (tensor<64xi32>, tensor<64xi32>) -> (i32, i32)
  return %r#0, %r#1 : i32, i32
}
