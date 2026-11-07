// RUN: triton-opt --triton-to-linalg="compile-on-910-95=false" --split-input-file --verify-each %s | FileCheck %s --check-prefixes=COMMON,A2 --enable-var-scope
// RUN: triton-opt --triton-to-linalg="compile-on-910-95=true" --split-input-file --verify-each %s | FileCheck %s --check-prefixes=COMMON,A5 --enable-var-scope

// A pure-vector runtime point retains the guard on both targets. On 950,
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

// -----

// Matrix operations do not change the select-slice policy.
module {
  func.func @mixed_dynamic_point(%lhs: tensor<16x32xf16>, %rhs: tensor<32x32xf16>, %false: tensor<16x32xf32>, %start: i32, %end: i32) -> tensor<16x32xf32> {
    %zero = arith.constant dense<0.0> : tensor<16x32xf32>
    %true = tt.dot %lhs, %rhs, %zero : tensor<16x32xf16> * tensor<32x32xf16> -> tensor<16x32xf32>
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %starts = tt.splat %start : i32 -> tensor<16xi32>
    %lower = arith.cmpi eq, %range, %starts : tensor<16xi32>
    %expanded = tt.expand_dims %lower {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %broadcast = tt.broadcast %expanded : tensor<16x1xi1> -> tensor<16x32xi1>
    %result = arith.select %broadcast, %true, %false : tensor<16x32xi1>, tensor<16x32xf32>
    return %result : tensor<16x32xf32>
  }
}
// COMMON-LABEL: func.func @mixed_dynamic_point(
// COMMON: scf.if
// COMMON-NEXT: scf.yield
// COMMON-NEXT: } else {
// A5-NEXT: %[[PART:.*]] = tensor.extract_slice %{{.*}}[%[[OFF:.*]], 0] [1, 32] [1, 1]
// A5-NEXT: %[[RESULT:.*]] = tensor.insert_slice %[[PART]] into %{{.*}}[%[[OFF]], 0] [1, 32] [1, 1]
// A2-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %{{.*}}[0, 0] [%[[OFF:.*]], 32] [1, 1]
// A2-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %{{.*}}[0, 0] [%[[OFF]], 32] [1, 1]
// A2-NEXT: %[[PART:.*]] = tensor.extract_slice %[[TMP]][0, 0] [%[[END:.*]], 32] [1, 1]
// A2-NEXT: %[[RESULT:.*]] = tensor.insert_slice %[[PART]] into %{{.*}}[0, 0] [%[[END]], 32] [1, 1]
// COMMON-NEXT: scf.yield

// -----

// Matrix operations do not change the select-slice policy.
module {
  func.func @mixed_dynamic_window(%lhs: tensor<16x32xf16>, %rhs: tensor<32x32xf16>, %false: tensor<16x32xf32>, %start: i32, %end: i32) -> tensor<16x32xf32> {
    %zero = arith.constant dense<0.0> : tensor<16x32xf32>
    %true = tt.dot %lhs, %rhs, %zero : tensor<16x32xf16> * tensor<32x32xf16> -> tensor<16x32xf32>
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %starts = tt.splat %start : i32 -> tensor<16xi32>
    %lower = arith.cmpi sge, %range, %starts : tensor<16xi32>
    %ends = tt.splat %end : i32 -> tensor<16xi32>
    %upper = arith.cmpi slt, %range, %ends : tensor<16xi32>
    %mask = arith.andi %lower, %upper : tensor<16xi1>
    %expanded = tt.expand_dims %mask {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %broadcast = tt.broadcast %expanded : tensor<16x1xi1> -> tensor<16x32xi1>
    %result = arith.select %broadcast, %true, %false : tensor<16x32xi1>, tensor<16x32xf32>
    return %result : tensor<16x32xf32>
  }
}
// COMMON-LABEL: func.func @mixed_dynamic_window(
// COMMON: scf.if
// COMMON-NEXT: scf.yield
// COMMON-NEXT: } else {
// A5-NEXT: %[[PART:.*]] = tensor.extract_slice %{{.*}}[%[[OFF:.*]], 0] [%[[SIZE:.*]], 32] [1, 1]
// A5-NEXT: %[[RESULT:.*]] = tensor.insert_slice %[[PART]] into %{{.*}}[%[[OFF]], 0] [%[[SIZE]], 32] [1, 1]
// A2-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %{{.*}}[0, 0] [%[[OFF:.*]], 32] [1, 1]
// A2-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %{{.*}}[0, 0] [%[[OFF]], 32] [1, 1]
// A2-NEXT: %[[PART:.*]] = tensor.extract_slice %[[TMP]][0, 0] [%[[END:.*]], 32] [1, 1]
// A2-NEXT: %[[RESULT:.*]] = tensor.insert_slice %[[PART]] into %{{.*}}[0, 0] [%[[END]], 32] [1, 1]
// COMMON-NEXT: scf.yield

// -----

// Matrix operations do not change the select-slice policy.
module {
  func.func @mixed_dynamic_tail(%lhs: tensor<16x32xf16>, %rhs: tensor<32x32xf16>, %false: tensor<16x32xf32>, %start: i32, %end: i32) -> tensor<16x32xf32> {
    %zero = arith.constant dense<0.0> : tensor<16x32xf32>
    %true = tt.dot %lhs, %rhs, %zero : tensor<16x32xf16> * tensor<32x32xf16> -> tensor<16x32xf32>
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %starts = tt.splat %start : i32 -> tensor<16xi32>
    %lower = arith.cmpi sge, %range, %starts : tensor<16xi32>
    %expanded = tt.expand_dims %lower {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %broadcast = tt.broadcast %expanded : tensor<16x1xi1> -> tensor<16x32xi1>
    %result = arith.select %broadcast, %true, %false : tensor<16x32xi1>, tensor<16x32xf32>
    return %result : tensor<16x32xf32>
  }
}
// COMMON-LABEL: func.func @mixed_dynamic_tail(
// COMMON: scf.if
// COMMON-NEXT: scf.yield
// COMMON-NEXT: } else {
// A5-NEXT: %[[PART:.*]] = tensor.extract_slice %{{.*}}[%[[OFF:.*]], 0] [%[[SIZE:.*]], 32] [1, 1]
// A5-NEXT: %[[RESULT:.*]] = tensor.insert_slice %[[PART]] into %{{.*}}[%[[OFF]], 0] [%[[SIZE]], 32] [1, 1]
// A2-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %{{.*}}[0, 0] [%[[OFF:.*]], 32] [1, 1]
// A2-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %{{.*}}[0, 0] [%[[OFF]], 32] [1, 1]
// COMMON-NEXT: scf.yield

// -----

// KDA masks computed rows with a shared zero tensor. Overwriting the false
// prefix permits reuse of the computed value without copying the whole zero tensor.
module {
  func.func @dynamic_tail_constant_false(%a: tensor<32x128xf32>, %b: tensor<32x128xf32>, %start: i32) -> tensor<32x128xf32> {
    %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %starts = tt.splat %start : i32 -> tensor<32xi32>
    %mask = arith.cmpi sge, %range, %starts : tensor<32xi32>
    %expanded = tt.expand_dims %mask {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
    %broadcast = tt.broadcast %expanded : tensor<32x1xi1> -> tensor<32x128xi1>
    %true = arith.mulf %a, %b : tensor<32x128xf32>
    %false = arith.constant dense<0.0> : tensor<32x128xf32>
    %result = arith.select %broadcast, %true, %false : tensor<32x128xi1>, tensor<32x128xf32>
    return %result : tensor<32x128xf32>
  }
}
// COMMON-LABEL: func.func @dynamic_tail_constant_false(
// COMMON: %[[F:.*]] = linalg.fill {{.*}} -> tensor<32x128xf32>
// COMMON: %[[T:.*]] = arith.mulf
// COMMON-NOT: tensor.extract_slice
// COMMON-NOT: tensor.insert_slice
// COMMON: %[[RESULT:.*]] = scf.if
// COMMON-NEXT: scf.yield %[[F]]
// COMMON-NEXT: } else {
// COMMON-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %[[F]][0, 0] [%[[OFF:.*]], 128] [1, 1]
// COMMON-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PREFIX]] into %[[T]][0, 0] [%[[OFF]], 128] [1, 1]
// COMMON-NEXT: scf.yield %[[INSERT]]
// COMMON-NEXT: }
// COMMON-NEXT: return %[[RESULT]]

// -----

// The writable-destination choice also applies to a static tail and a nonzero
// constant. It does not depend on the numerical value of the false tensor.
module {
  func.func @static_tail_constant_false(%true: tensor<16xi32>) -> tensor<16xi32> {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %start = arith.constant dense<3> : tensor<16xi32>
    %mask = arith.cmpi sge, %range, %start : tensor<16xi32>
    %false = arith.constant dense<-7> : tensor<16xi32>
    %result = arith.select %mask, %true, %false : tensor<16xi1>, tensor<16xi32>
    return %result : tensor<16xi32>
  }
}
// COMMON-LABEL: func.func @static_tail_constant_false(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<16xi32>
// COMMON-NOT: scf.if
// COMMON: %[[PREFIX:.*]] = linalg.fill {{.*}} -> tensor<3xi32>
// COMMON-NEXT: %[[RESULT:.*]] = tensor.insert_slice %[[PREFIX]] into %[[T]][0] [3] [1]
// COMMON-NEXT: return %[[RESULT]]

// -----

// A constant false value alone must not disable the 950 point optimization.
module {
  func.func @point_constant_false(%true: tensor<16xi32>, %position: i32) -> tensor<16xi32> {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %positions = tt.splat %position : i32 -> tensor<16xi32>
    %mask = arith.cmpi eq, %range, %positions : tensor<16xi32>
    %false = arith.constant dense<0> : tensor<16xi32>
    %result = arith.select %mask, %true, %false : tensor<16xi1>, tensor<16xi32>
    return %result : tensor<16xi32>
  }
}
// COMMON-LABEL: func.func @point_constant_false(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<16xi32>
// COMMON: %[[F:.*]] = linalg.fill {{.*}} -> tensor<16xi32>
// COMMON: scf.if
// COMMON-NEXT: scf.yield %[[F]]
// COMMON-NEXT: } else {
// A5-NEXT: %[[PART:.*]] = tensor.extract_slice %[[T]][%[[POS:.*]]] [1] [1]
// A5-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][%[[POS]]] [1] [1]
// A2-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %[[F]][0] [%[[POS:.*]]] [1]
// A2-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %[[T]][0] [%[[POS]]] [1]
// A2-NEXT: %[[PART:.*]] = tensor.extract_slice %[[TMP]][0] [%[[END:.*]]] [1]
// A2-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][0] [%[[END]]] [1]
// COMMON-NEXT: scf.yield %[[INSERT]]

// -----

// Retain direct slices when true is also live after the select: overwriting it
// would require preserving another copy, even though false is constant.
module {
  func.func @tail_shared_true(%true: tensor<16xf32>, %start: i32) -> (tensor<16xf32>, tensor<16xf32>) {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %starts = tt.splat %start : i32 -> tensor<16xi32>
    %mask = arith.cmpi sge, %range, %starts : tensor<16xi32>
    %false = arith.constant dense<0.0> : tensor<16xf32>
    %result = arith.select %mask, %true, %false : tensor<16xi1>, tensor<16xf32>
    return %result, %true : tensor<16xf32>, tensor<16xf32>
  }
}
// COMMON-LABEL: func.func @tail_shared_true(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<16xf32>
// COMMON: %[[F:.*]] = linalg.fill {{.*}} -> tensor<16xf32>
// COMMON: %[[RESULT:.*]] = scf.if
// COMMON-NEXT: scf.yield %[[F]]
// COMMON-NEXT: } else {
// A5-NEXT: %[[PART:.*]] = tensor.extract_slice %[[T]][%[[OFF:.*]]] [%[[SIZE:.*]]] [1]
// A5-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][%[[OFF]]] [%[[SIZE]]] [1]
// A2-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %[[F]][0] [%[[OFF:.*]]] [1]
// A2-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PREFIX]] into %[[T]][0] [%[[OFF]]] [1]
// COMMON-NEXT: scf.yield %[[INSERT]]
// COMMON-NEXT: }
// COMMON-NEXT: return %[[RESULT]], %[[T]]
