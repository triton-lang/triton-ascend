// RUN: triton-opt --triton-to-linalg %s | FileCheck %s --enable-var-scope

module {

func.func @dynamic(%payload: tensor<8xi32>, %winner: i32) -> i32 {

  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %winner : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}


func.func @winner_zero(%payload: tensor<8xi32>, %winner: i32) -> i32 {
  %fixed = arith.constant 0 : i32

  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %fixed : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}


func.func @winner_last(%payload: tensor<8xi32>, %winner: i32) -> i32 {
  %fixed = arith.constant 7 : i32

  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %fixed : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}


func.func @winner_negative(%payload: tensor<8xi32>, %winner: i32) -> i32 {
  %fixed = arith.constant -1 : i32

  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %fixed : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}


func.func @winner_equal_n(%payload: tensor<8xi32>, %winner: i32) -> i32 {
  %fixed = arith.constant 8 : i32

  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %fixed : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}


func.func @winner_above_n(%payload: tensor<8xi32>, %winner: i32) -> i32 {
  %fixed = arith.constant 9 : i32

  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %fixed : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}


func.func @reject_nonzero_range(%payload: tensor<8xi32>, %winner: i32) -> i32 {

  %range = tt.make_range {start = 4 : i32, end = 12 : i32} : tensor<8xi32>
  %splat = tt.splat %winner : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}


func.func @reject_not_equal(%payload: tensor<8xi32>, %winner: i32) -> i32 {

  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %winner : i32 -> tensor<8xi32>
  %mask = arith.cmpi ne, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}


func.func @reject_single_element(%payload: tensor<1xi32>, %winner: i32) -> i32 {

  %range = tt.make_range {start = 0 : i32, end = 1 : i32} : tensor<1xi32>
  %splat = tt.splat %winner : i32 -> tensor<1xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<1xi32>
  %fill = arith.constant dense<-1> : tensor<1xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<1xi1>, tensor<1xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<1xi32>) -> i32
  return %result : i32
}


func.func @reject_unsigned_max(%payload: tensor<8xi32>, %winner: i32) -> i32 {

  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %winner : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxui %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}


func.func @reject_sum(%payload: tensor<8xi32>, %winner: i32) -> i32 {

  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %winner : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.addi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}



func.func @reversed_eq(%payload: tensor<8xi32>, %winner: i32) -> i32 {
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %winner : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %splat, %range : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}

func.func @positive_fill(%payload: tensor<8xi32>, %winner: i32) -> i32 {
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %winner : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<17> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}

func.func @negative_payload(%payload: tensor<8xi32>, %winner: i32) -> i32 {
  %data = arith.constant dense<-3> : tensor<8xi32>
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %winner : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<-1> : tensor<8xi32>
  %selected = arith.select %mask, %data, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}

func.func @reject_non_splat_fill(%payload: tensor<8xi32>, %winner: i32) -> i32 {
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %splat = tt.splat %winner : i32 -> tensor<8xi32>
  %mask = arith.cmpi eq, %range, %splat : tensor<8xi32>
  %fill = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
  %selected = arith.select %mask, %payload, %fill : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %r = arith.maxsi %a, %b : i32
    tt.reduce.return %r : i32
  }) : (tensor<8xi32>) -> i32
  return %result : i32
}
}

// CHECK-LABEL: func.func @dynamic(
// CHECK-SAME: %[[PAYLOAD:.*]]: tensor<8xi32>, %[[WINNER:.*]]: i32
// CHECK-DAG: %[[FILL:.*]] = arith.constant -1 : i32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[SIZE:.*]] = arith.constant 8 : i32
// CHECK: %[[LOWER:.*]] = arith.cmpi sge, %[[WINNER]], %[[ZERO]]
// CHECK: %[[UPPER:.*]] = arith.cmpi slt, %[[WINNER]], %[[SIZE]]
// CHECK: %[[IN_RANGE:.*]] = arith.andi %[[LOWER]], %[[UPPER]]
// CHECK: scf.if %[[IN_RANGE]] -> (i32)
// CHECK: %[[INDEX:.*]] = arith.index_cast %[[WINNER]] : i32 to index
// CHECK: %[[SELECTED:.*]] = tensor.extract %[[PAYLOAD]][%[[INDEX]]]
// CHECK: %[[CLAMPED:.*]] = arith.maxsi %[[SELECTED]], %[[FILL]]
// CHECK: scf.yield %[[CLAMPED]]
// CHECK: } else {
// CHECK: scf.yield %[[FILL]]
// CHECK-NOT: linalg.reduce
// CHECK: return

// CHECK-LABEL: func.func @winner_zero(
// CHECK: tensor.extract
// CHECK: arith.maxsi
// CHECK-NOT: linalg.reduce
// CHECK: return

// CHECK-LABEL: func.func @winner_last(
// CHECK: tensor.extract
// CHECK: arith.maxsi
// CHECK-NOT: linalg.reduce
// CHECK: return

// CHECK-LABEL: func.func @winner_negative(
// CHECK-NOT: tensor.extract
// CHECK-NOT: linalg.reduce
// CHECK: %[[FILL:.*]] = arith.constant -1 : i32
// CHECK-NEXT: return %[[FILL]] : i32

// CHECK-LABEL: func.func @winner_equal_n(
// CHECK-NOT: tensor.extract
// CHECK-NOT: linalg.reduce
// CHECK: %[[FILL:.*]] = arith.constant -1 : i32
// CHECK-NEXT: return %[[FILL]] : i32

// CHECK-LABEL: func.func @winner_above_n(
// CHECK-NOT: tensor.extract
// CHECK-NOT: linalg.reduce
// CHECK: %[[FILL:.*]] = arith.constant -1 : i32
// CHECK-NEXT: return %[[FILL]] : i32

// CHECK-LABEL: func.func @reject_nonzero_range(
// CHECK: linalg.reduce

// CHECK-LABEL: func.func @reject_not_equal(
// CHECK: linalg.reduce

// A valid single lane must not be clamped against the fill value.
// CHECK-LABEL: func.func @reject_single_element(
// CHECK-NOT: arith.maxsi
// CHECK: return

// CHECK-LABEL: func.func @reject_unsigned_max(
// CHECK: linalg.reduce

// CHECK-LABEL: func.func @reject_sum(
// CHECK: linalg.reduce

// CHECK-LABEL: func.func @reversed_eq(
// CHECK: scf.if
// CHECK: tensor.extract
// CHECK: arith.maxsi
// CHECK-NOT: linalg.reduce
// CHECK: return

// CHECK-LABEL: func.func @positive_fill(
// CHECK: %[[FILL:.*]] = arith.constant 17 : i32
// CHECK: scf.if
// CHECK: %[[SELECTED:.*]] = tensor.extract
// CHECK: arith.maxsi %[[SELECTED]], %[[FILL]]
// CHECK-NOT: linalg.reduce
// CHECK: return

// max(where(..., -3, -1)) is always -1, including an out-of-range winner.
// CHECK-LABEL: func.func @negative_payload(
// CHECK: %[[FILL:.*]] = arith.constant -1 : i32
// CHECK-NEXT: return %[[FILL]] : i32

// CHECK-LABEL: func.func @reject_non_splat_fill(
// CHECK: linalg.reduce
