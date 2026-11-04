// RUN: triton-opt --triton-to-linalg="compile-on-910-95=false" --split-input-file --verify-each %s | FileCheck %s --check-prefixes=COMMON,A2 --enable-var-scope
// RUN: triton-opt --triton-to-linalg="compile-on-910-95=true" --split-input-file --verify-each %s | FileCheck %s --check-prefixes=COMMON,A5 --enable-var-scope

// A runtime point must retain the out-of-bounds guard on both targets. On 950,
// only the selected element is copied; A2/A3 keep every slice start at zero.
module {
  func.func @dynamic_point(%true: tensor<16xi32>, %false: tensor<16xi32>, %position: i32) -> tensor<16xi32> {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %positions = tt.splat %position : i32 -> tensor<16xi32>
    %mask = arith.cmpi eq, %range, %positions : tensor<16xi32>
    %result = arith.select %mask, %true, %false : tensor<16xi1>, tensor<16xi32>
    return %result : tensor<16xi32>
  }
}
// COMMON-LABEL: func.func @dynamic_point(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<16xi32>, %[[F:[^ :]+]]: tensor<16xi32>, %[[P:[^ :]+]]: i32
// COMMON-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// COMMON-DAG: %[[N:.*]] = arith.constant 16 : index
// COMMON: %[[POS:.*]] = arith.index_cast %[[P]] : i32 to index
// COMMON-NOT: tensor.extract_slice
// COMMON-NOT: tensor.insert_slice
// COMMON: %[[NEG:.*]] = arith.cmpi slt, %[[POS]], %[[ZERO]]
// COMMON: %[[PAST:.*]] = arith.cmpi sge, %[[POS]], %[[N]]
// COMMON: %[[OOB:.*]] = arith.ori %[[NEG]], %[[PAST]]
// COMMON: %[[RESULT:.*]] = scf.if %[[OOB]] -> (tensor<16xi32>) {
// COMMON-NEXT: scf.yield %[[F]] : tensor<16xi32>
// COMMON-NEXT: } else {
// A5-NEXT: %[[PART:.*]] = tensor.extract_slice %[[T]][%[[POS]]] [1] [1]
// A5-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][%[[POS]]] [1] [1]
// A2-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %[[F]][0] [%[[POS]]] [1]
// A2-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %[[T]][0] [%[[POS]]] [1]
// A2-NEXT: %[[PART:.*]] = tensor.extract_slice %[[TMP]][0] [%[[END:.*]]] [1]
// A2-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][0] [%[[END]]] [1]
// COMMON-NEXT: scf.yield %[[INSERT]] : tensor<16xi32>
// COMMON-NEXT: }
// COMMON-NEXT: return %[[RESULT]] : tensor<16xi32>

// -----

module {
  func.func @dynamic_window(%true: tensor<16xf32>, %false: tensor<16xf32>, %lo: i32, %hi: i32) -> tensor<16xf32> {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %los = tt.splat %lo : i32 -> tensor<16xi32>
    %his = tt.splat %hi : i32 -> tensor<16xi32>
    %lower = arith.cmpi sge, %range, %los : tensor<16xi32>
    %upper = arith.cmpi slt, %range, %his : tensor<16xi32>
    %mask = arith.andi %lower, %upper : tensor<16xi1>
    %result = arith.select %mask, %true, %false : tensor<16xi1>, tensor<16xf32>
    return %result : tensor<16xf32>
  }
}
// COMMON-LABEL: func.func @dynamic_window(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<16xf32>, %[[F:[^ :]+]]: tensor<16xf32>
// COMMON-NOT: tensor.extract_slice
// COMMON-NOT: tensor.insert_slice
// COMMON: scf.if
// COMMON-NEXT: scf.yield %[[F]]
// COMMON-NEXT: } else {
// A5-NEXT: %[[PART:.*]] = tensor.extract_slice %[[T]][%[[OFF:.*]]] [%[[SIZE:.*]]] [1]
// A5-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][%[[OFF]]] [%[[SIZE]]] [1]
// A2-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %[[F]][0] [%[[OFF:.*]]] [1]
// A2-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %[[T]][0] [%[[OFF]]] [1]
// A2-NEXT: %[[PART:.*]] = tensor.extract_slice %[[TMP]][0] [%[[END:.*]]] [1]
// A2-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][0] [%[[END]]] [1]
// COMMON-NEXT: scf.yield %[[INSERT]]

// -----

module {
  func.func @static_window(%true: tensor<16xf32>, %false: tensor<16xf32>) -> tensor<16xf32> {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %lo = arith.constant dense<3> : tensor<16xi32>
    %hi = arith.constant dense<11> : tensor<16xi32>
    %lower = arith.cmpi sge, %range, %lo : tensor<16xi32>
    %upper = arith.cmpi slt, %range, %hi : tensor<16xi32>
    %mask = arith.andi %lower, %upper : tensor<16xi1>
    %result = arith.select %mask, %true, %false : tensor<16xi1>, tensor<16xf32>
    return %result : tensor<16xf32>
  }
}
// COMMON-LABEL: func.func @static_window(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<16xf32>, %[[F:[^ :]+]]: tensor<16xf32>
// COMMON-NOT: scf.if
// A5-NEXT: %[[PART:.*]] = tensor.extract_slice %[[T]][3] [8] [1]
// A5-NEXT: %[[RESULT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][3] [8] [1]
// A2-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %[[F]][0] [3] [1]
// A2-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %[[T]][0] [3] [1]
// A2-NEXT: %[[PART:.*]] = tensor.extract_slice %[[TMP]][0] [11] [1]
// A2-NEXT: %[[RESULT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][0] [11] [1]
// COMMON-NEXT: return %[[RESULT]]

// -----

// A prefix already has an aligned zero start on both targets.
module {
  func.func @dynamic_head(%true: tensor<16xf32>, %false: tensor<16xf32>, %end: i32) -> tensor<16xf32> {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %ends = tt.splat %end : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %range, %ends : tensor<16xi32>
    %result = arith.select %mask, %true, %false : tensor<16xi1>, tensor<16xf32>
    return %result : tensor<16xf32>
  }
}
// COMMON-LABEL: func.func @dynamic_head(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<16xf32>, %[[F:[^ :]+]]: tensor<16xf32>
// COMMON-NOT: scf.if
// COMMON: %[[PART:.*]] = tensor.extract_slice %[[T]][0] [%[[SIZE:.*]]] [1]
// COMMON-NEXT: %[[RESULT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][0] [%[[SIZE]]] [1]
// COMMON-NEXT: return %[[RESULT]]

// -----

// A dynamic last-axis slice across multiple rows is not a single-element case.
module {
  func.func @dynamic_columns(%true: tensor<4x16xf32>, %false: tensor<4x16xf32>, %lo: i32, %hi: i32) -> tensor<4x16xf32> {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %los = tt.splat %lo : i32 -> tensor<16xi32>
    %his = tt.splat %hi : i32 -> tensor<16xi32>
    %lower = arith.cmpi sge, %range, %los : tensor<16xi32>
    %upper = arith.cmpi slt, %range, %his : tensor<16xi32>
    %mask = arith.andi %lower, %upper : tensor<16xi1>
    %expanded = tt.expand_dims %mask {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %broadcast = tt.broadcast %expanded : tensor<1x16xi1> -> tensor<4x16xi1>
    %result = arith.select %broadcast, %true, %false : tensor<4x16xi1>, tensor<4x16xf32>
    return %result : tensor<4x16xf32>
  }
}
// COMMON-LABEL: func.func @dynamic_columns(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<4x16xf32>, %[[F:[^ :]+]]: tensor<4x16xf32>
// COMMON-NOT: tensor.extract_slice
// COMMON-NOT: tensor.insert_slice
// COMMON: scf.if
// COMMON-NEXT: scf.yield %[[F]]
// COMMON-NEXT: } else {
// A5-NEXT: %[[PART:.*]] = tensor.extract_slice %[[T]][0, %[[OFF:.*]]] [4, %[[SIZE:.*]]] [1, 1]
// A5-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][0, %[[OFF]]] [4, %[[SIZE]]] [1, 1]
// A2-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %[[F]][0, 0] [4, %[[OFF:.*]]] [1, 1]
// A2-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %[[T]][0, 0] [4, %[[OFF]]] [1, 1]
// A2-NEXT: %[[PART:.*]] = tensor.extract_slice %[[TMP]][0, 0] [4, %[[END:.*]]] [1, 1]
// A2-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][0, 0] [4, %[[END]]] [1, 1]
// COMMON-NEXT: scf.yield %[[INSERT]]

// -----

// A mask that is not a contiguous interval keeps the ordinary select lowering.
module {
  func.func @arbitrary_mask(%true: tensor<16xi32>, %false: tensor<16xi32>, %mask: tensor<16xi1>) -> tensor<16xi32> {
    %result = arith.select %mask, %true, %false : tensor<16xi1>, tensor<16xi32>
    return %result : tensor<16xi32>
  }
}
// COMMON-LABEL: func.func @arbitrary_mask(
// COMMON-NOT: tensor.extract_slice
// COMMON-NOT: tensor.insert_slice
// COMMON: arith.select
// COMMON: return
