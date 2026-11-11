// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s --implicit-check-not=linalg.reduce --implicit-check-not=reduce_mode
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=True" --split-input-file %s | FileCheck %s --implicit-check-not=linalg.reduce --implicit-check-not=reduce_mode

// Unsupported tuple combiners remain in TTIR during partial conversion.
// They must not be lowered or tagged as a value/index minimum or maximum.

// A maximum and a count sum are not a value/index selection, even though the
// comparison predicate sequence looks like max-with-index.
// CHECK-LABEL: func.func @max_and_count
// CHECK: "tt.reduce"
// CHECK: arith.addi
// CHECK: tt.reduce.return
func.func @max_and_count(%x: tensor<64xi32>, %counts: tensor<64xi32>) -> (i32, i32) {
  %r:2 = "tt.reduce"(%x, %counts) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %ca: i32, %b: i32, %cb: i32):
    %greater = arith.cmpi sgt, %a, %b : i32
    %v = arith.select %greater, %a, %b : i32
    %count = arith.addi %ca, %cb : i32
    tt.reduce.return %v, %count : i32, i32
  }) : (tensor<64xi32>, tensor<64xi32>) -> (i32, i32)
  return %r#0, %r#1 : i32, i32
}

// -----

// The value and index must come from the same pair.
// CHECK-LABEL: func.func @mismatched_index
// CHECK: "tt.reduce"
// CHECK: tt.reduce.return
func.func @mismatched_index(%x: tensor<64xi32>, %indices: tensor<64xi32>) -> (i32, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %ia: i32, %b: i32, %ib: i32):
    %greater = arith.cmpi sgt, %a, %b : i32
    %v = arith.select %greater, %a, %b : i32
    %i = arith.select %greater, %ib, %ia : i32
    tt.reduce.return %v, %i : i32, i32
  }) : (tensor<64xi32>, tensor<64xi32>) -> (i32, i32)
  return %r#0, %r#1 : i32, i32
}

// -----

// Selecting a pair by index alone is not a maximum of its value.
// CHECK-LABEL: func.func @compare_indices
// CHECK: "tt.reduce"
// CHECK: tt.reduce.return
func.func @compare_indices(%x: tensor<64xi32>, %indices: tensor<64xi32>) -> (i32, i32) {
  %r:2 = "tt.reduce"(%x, %indices) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %ia: i32, %b: i32, %ib: i32):
    %greater = arith.cmpi sgt, %ia, %ib : i32
    %v = arith.select %greater, %a, %b : i32
    %i = arith.select %greater, %ia, %ib : i32
    tt.reduce.return %v, %i : i32, i32
  }) : (tensor<64xi32>, tensor<64xi32>) -> (i32, i32)
  return %r#0, %r#1 : i32, i32
}
