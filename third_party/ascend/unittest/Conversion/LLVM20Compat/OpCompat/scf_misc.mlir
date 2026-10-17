// RUN: triton-opt %s | FileCheck %s

// CHECK-LABEL: func.func @scf_misc(%arg0: i32) -> i32 {
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[C4:.*]] = arith.constant 4 : i32
// CHECK-NEXT: %[[COND:.*]] = arith.cmpi sgt, %arg0, %[[C0]] : i32
// CHECK-NEXT: %[[IFR:.*]] = scf.if %[[COND]] -> (i32) {
// CHECK: scf.yield %arg0 : i32
// CHECK: } else {
// CHECK: scf.yield %[[C1]] : i32
// CHECK: }
// CHECK: %[[WHILE:.*]] = scf.while (%[[V0:.*]] = %[[IFR]]) : (i32) -> i32 {
// CHECK: %[[KEEPGOING:.*]] = arith.cmpi slt, %[[V0]], %[[C4]] : i32
// CHECK: scf.condition(%[[KEEPGOING]]) %[[V0]] : i32
// CHECK: } do {
// CHECK: ^bb0(%[[V1:.*]]: i32):
// CHECK: %[[STEP:.*]] = arith.addi %[[V1]], %[[C1]] : i32
// CHECK: scf.yield %[[STEP]] : i32
// CHECK: }
// CHECK-NEXT: return %[[WHILE]] : i32
func.func @scf_misc(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %cond = arith.cmpi sgt, %arg0, %c0 : i32
  %ifv = scf.if %cond -> (i32) {
    scf.yield %arg0 : i32
  } else {
    scf.yield %c1 : i32
  }
  %res = scf.while (%v = %ifv) : (i32) -> i32 {
    %keep_going = arith.cmpi slt, %v, %c4 : i32
    scf.condition(%keep_going) %v : i32
  } do {
  ^bb0(%v: i32):
    %step = arith.addi %v, %c1 : i32
    scf.yield %step : i32
  }
  return %res : i32
}
