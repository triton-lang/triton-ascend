// RUN: triton-opt --triton-to-linalg="compile-on-910-95=false" --split-input-file --verify-each %s | FileCheck %s --check-prefixes=COMMON,A2 --enable-var-scope
// RUN: triton-opt --triton-to-linalg="compile-on-910-95=true" --split-input-file --verify-each %s | FileCheck %s --check-prefixes=COMMON,A5 --enable-var-scope

// An arbitrary runtime point retains direct slices on 950. Only loop
// induction positions use the single-element prefix fallback there.
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

// -----

// Reproduce mix_fused: each iteration updates one lane of loop-carried scores.
module {
  func.func @loop_carried_point(%initial: tensor<64xf32>, %score: f32, %count: i32) -> tensor<64xf32> {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %count_idx = arith.index_cast %count : i32 to index
    %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %scores = scf.for %row = %zero to %count_idx step %one iter_args(%previous = %initial) -> (tensor<64xf32>) {
      %row_i32 = arith.index_cast %row : index to i32
      %rows = tt.splat %row_i32 : i32 -> tensor<64xi32>
      %value = tt.splat %score : f32 -> tensor<64xf32>
      %mask = arith.cmpi eq, %range, %rows : tensor<64xi32>
      %updated = arith.select %mask, %value, %previous : tensor<64xi1>, tensor<64xf32>
      scf.yield %updated : tensor<64xf32>
    }
    return %scores : tensor<64xf32>
  }
}
// COMMON-LABEL: func.func @loop_carried_point(
// COMMON: scf.for {{.*}} iter_args(%[[PREV:.*]] = {{.*}})
// COMMON: %[[VALUE:.*]] = linalg.fill
// COMMON: scf.if
// COMMON-NEXT: scf.yield %[[PREV]]
// COMMON-NEXT: } else {
// COMMON-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %[[PREV]][0] [%[[POS:.*]]] [1]
// COMMON-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %[[VALUE]][0] [%[[POS]]] [1]
// COMMON-NEXT: %[[PART:.*]] = tensor.extract_slice %[[TMP]][0] [%[[END:.*]]] [1]
// COMMON-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %[[PREV]][0] [%[[END]]] [1]
// COMMON-NEXT: scf.yield %[[INSERT]]

// -----

// A statically positioned single element still uses a direct slice on 950.
module {
  func.func @static_point(%true: tensor<16xi32>, %false: tensor<16xi32>) -> tensor<16xi32> {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %position = arith.constant dense<7> : tensor<16xi32>
    %mask = arith.cmpi eq, %range, %position : tensor<16xi32>
    %result = arith.select %mask, %true, %false : tensor<16xi1>, tensor<16xi32>
    return %result : tensor<16xi32>
  }
}
// COMMON-LABEL: func.func @static_point(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<16xi32>, %[[F:[^ :]+]]: tensor<16xi32>
// COMMON-NOT: scf.if
// A5-NEXT: %[[PART:.*]] = tensor.extract_slice %[[T]][7] [1] [1]
// A5-NEXT: %[[RESULT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][7] [1] [1]
// A2-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %[[F]][0] [7] [1]
// A2-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %[[T]][0] [7] [1]
// A2-NEXT: %[[PART:.*]] = tensor.extract_slice %[[TMP]][0] [8] [1]
// A2-NEXT: %[[RESULT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][0] [8] [1]
// COMMON-NEXT: return %[[RESULT]]

// -----

// The other mix_fused loop selects a probability with the induction variable
// and reduces it. Its constant false tensor also keeps the prefix path.
module {
  func.func @loop_point_probability(%probabilities: tensor<64xf32>, %count: index) -> f32 {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %zero_f32 = arith.constant 0.0 : f32
    %zeros = arith.constant dense<0.0> : tensor<64xf32>
    %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %sum = scf.for %row = %zero to %count step %one iter_args(%acc = %zero_f32) -> (f32) {
      %row_i32 = arith.index_cast %row : index to i32
      %rows = tt.splat %row_i32 : i32 -> tensor<64xi32>
      %mask = arith.cmpi eq, %range, %rows : tensor<64xi32>
      %selected = arith.select %mask, %probabilities, %zeros : tensor<64xi1>, tensor<64xf32>
      %probability = "tt.reduce"(%selected) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %value = arith.addf %lhs, %rhs : f32
        tt.reduce.return %value : f32
      }) {axis = 0 : i32} : (tensor<64xf32>) -> f32
      %next = arith.addf %acc, %probability : f32
      scf.yield %next : f32
    }
    return %sum : f32
  }
}
// COMMON-LABEL: func.func @loop_point_probability(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<64xf32>
// COMMON: %[[F:.*]] = linalg.fill {{.*}} -> tensor<64xf32>
// COMMON: scf.for
// COMMON: %[[SELECTED:.*]] = scf.if
// COMMON-NEXT: scf.yield %[[F]]
// COMMON-NEXT: } else {
// COMMON-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %[[F]][0] [%[[POS:.*]]] [1]
// COMMON-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %[[T]][0] [%[[POS]]] [1]
// COMMON-NEXT: %[[PART:.*]] = tensor.extract_slice %[[TMP]][0] [%[[END:.*]]] [1]
// COMMON-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %[[F]][0] [%[[END]]] [1]
// COMMON-NEXT: scf.yield %[[INSERT]]
// COMMON-NEXT: }
// COMMON: linalg.reduce ins(%[[SELECTED]]

// -----

// A loop-carried integer position is not the induction variable. Keep its
// single-element selection direct on 950, even inside an scf.for.
module {
  func.func @loop_carried_position(%true: tensor<64xi32>, %initial: tensor<64xi32>, %position: i32, %stride: i32, %count: index) -> tensor<64xi32> {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %result:2 = scf.for %iv = %zero to %count step %one iter_args(%previous = %initial, %pos = %position) -> (tensor<64xi32>, i32) {
      %positions = tt.splat %pos : i32 -> tensor<64xi32>
      %mask = arith.cmpi eq, %range, %positions : tensor<64xi32>
      %updated = arith.select %mask, %true, %previous : tensor<64xi1>, tensor<64xi32>
      %next = arith.addi %pos, %stride : i32
      scf.yield %updated, %next : tensor<64xi32>, i32
    }
    return %result#0 : tensor<64xi32>
  }
}
// COMMON-LABEL: func.func @loop_carried_position(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<64xi32>
// COMMON: scf.for {{.*}} iter_args(%[[F:[^ ]+]] =
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

// An induction position selecting a whole row still uses direct slices on
// 950: the prefix fallback applies only when the entire region is one element.
module {
  func.func @loop_point_row(%true: tensor<16x32xf32>, %initial: tensor<16x32xf32>, %count: index) -> tensor<16x32xf32> {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %result = scf.for %row = %zero to %count step %one iter_args(%previous = %initial) -> (tensor<16x32xf32>) {
      %row_i32 = arith.index_cast %row : index to i32
      %rows = tt.splat %row_i32 : i32 -> tensor<16xi32>
      %point = arith.cmpi eq, %range, %rows : tensor<16xi32>
      %expanded = tt.expand_dims %point {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
      %mask = tt.broadcast %expanded : tensor<16x1xi1> -> tensor<16x32xi1>
      %updated = arith.select %mask, %true, %previous : tensor<16x32xi1>, tensor<16x32xf32>
      scf.yield %updated : tensor<16x32xf32>
    }
    return %result : tensor<16x32xf32>
  }
}
// COMMON-LABEL: func.func @loop_point_row(
// COMMON: scf.for
// COMMON: scf.if
// COMMON-NEXT: scf.yield
// COMMON-NEXT: } else {
// A5-NEXT: %[[PART:.*]] = tensor.extract_slice %{{.*}}[%[[POS:.*]], 0] [1, 32] [1, 1]
// A5-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %{{.*}}[%[[POS]], 0] [1, 32] [1, 1]
// A2-NEXT: %[[PREFIX:.*]] = tensor.extract_slice %{{.*}}[0, 0] [%[[POS:.*]], 32] [1, 1]
// A2-NEXT: %[[TMP:.*]] = tensor.insert_slice %[[PREFIX]] into %{{.*}}[0, 0] [%[[POS]], 32] [1, 1]
// A2-NEXT: %[[PART:.*]] = tensor.extract_slice %[[TMP]][0, 0] [%[[END:.*]], 32] [1, 1]
// A2-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[PART]] into %{{.*}}[0, 0] [%[[END]], 32] [1, 1]
// COMMON-NEXT: scf.yield %[[INSERT]]

// -----

// TopK selects a point obtained from a reduction, not a loop induction
// variable. Preserve the direct path for its large candidate vectors on 950.
module {
  func.func @reduction_point(%true: tensor<1024xi32>, %positions: tensor<1024xi32>) -> tensor<1024xi32> {
    %position = "tt.reduce"(%positions) ({
    ^bb0(%lhs: i32, %rhs: i32):
      %minimum = arith.minsi %lhs, %rhs : i32
      tt.reduce.return %minimum : i32
    }) {axis = 0 : i32} : (tensor<1024xi32>) -> i32
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %points = tt.splat %position : i32 -> tensor<1024xi32>
    %mask = arith.cmpi eq, %range, %points : tensor<1024xi32>
    %false = arith.constant dense<-1> : tensor<1024xi32>
    %selected = arith.select %mask, %true, %false : tensor<1024xi1>, tensor<1024xi32>
    return %selected : tensor<1024xi32>
  }
}
// COMMON-LABEL: func.func @reduction_point(
// COMMON-SAME: %[[T:[^ :]+]]: tensor<1024xi32>
// COMMON-DAG: %[[F:.*]] = linalg.fill {{.*}} -> tensor<1024xi32>
// COMMON-DAG: linalg.reduce
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
