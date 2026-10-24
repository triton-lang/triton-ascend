// RUN: triton-opt %s --split-input-file --plan-vector-block --verify-each | FileCheck %s --enable-var-scope

// Each case reserves three globally unique block IDs: a preplanned cube,
// independent vector work, and vector work deferred until its backedge producer.

// A then-region use of an outer iter_arg must join its updated producer.
// CHECK-LABEL: func.func @for_if_then_backedge(
func.func @for_if_then_backedge(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    %r = scf.if %cond -> tensor<8x8xf32> {
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
      %early = arith.addf %acc, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
      %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %cube = linalg.matmul {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
      // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
      %late = arith.subf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %next = arith.addf %early, %late {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      scf.yield %next : tensor<8x8xf32>
    } else {
      scf.yield %acc : tensor<8x8xf32>
    } {ssbuffer.core_type = "CUBE, VECTOR"}
    scf.yield %r : tensor<8x8xf32>
  } {ssbuffer.core_type = "CUBE, VECTOR"}
  return %r : tensor<8x8xf32>
}

// -----

// The same backedge mapping is needed when the producer is in else.
// CHECK-LABEL: func.func @for_if_else_backedge(
func.func @for_if_else_backedge(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    %r = scf.if %cond -> tensor<8x8xf32> {
      scf.yield %acc : tensor<8x8xf32>
    } else {
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}
      %early = arith.addf %acc, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"}
      %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %cube = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
      // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}
      %late = arith.subf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %next = arith.addf %early, %late {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      scf.yield %next : tensor<8x8xf32>
    } {ssbuffer.core_type = "CUBE, VECTOR"}
    scf.yield %r : tensor<8x8xf32>
  } {ssbuffer.core_type = "CUBE, VECTOR"}
  return %r : tensor<8x8xf32>
}

// -----

// Find the nearest loop through two nested if regions.
// CHECK-LABEL: func.func @for_nested_if_backedge(
func.func @for_nested_if_backedge(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    %r = scf.if %cond -> tensor<8x8xf32> {
      %nested = scf.if %cond -> tensor<8x8xf32> {
        // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"}
        %early = arith.addf %acc, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
        // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"}
        %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
        %cube = linalg.matmul {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
        // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"}
        %late = arith.subf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
        %next = arith.addf %early, %late {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
        scf.yield %next : tensor<8x8xf32>
      } else {
        scf.yield %acc : tensor<8x8xf32>
      }
      scf.yield %nested : tensor<8x8xf32>
    } else {
      scf.yield %acc : tensor<8x8xf32>
    } {ssbuffer.core_type = "CUBE, VECTOR"}
    scf.yield %r : tensor<8x8xf32>
  } {ssbuffer.core_type = "CUBE, VECTOR"}
  return %r : tensor<8x8xf32>
}

// -----

// Trace a single-block execute_region result.
// CHECK-LABEL: func.func @for_execute_region_backedge(
func.func @for_execute_region_backedge(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    %r = scf.if %cond -> tensor<8x8xf32> {
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"}
      %early = arith.addf %acc, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"}
      %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %cube = linalg.matmul {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
      // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"}
      %late = arith.subf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %next = arith.addf %early, %late {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %forward = scf.execute_region -> tensor<8x8xf32> {
        scf.yield %next : tensor<8x8xf32>
      }
      scf.yield %forward : tensor<8x8xf32>
    } else {
      scf.yield %acc : tensor<8x8xf32>
    } {ssbuffer.core_type = "CUBE, VECTOR"}
    scf.yield %r : tensor<8x8xf32>
  } {ssbuffer.core_type = "CUBE, VECTOR"}
  return %r : tensor<8x8xf32>
}

// -----

// Follow a nested loop result and its unchanged iter_arg without cycling.
// CHECK-LABEL: func.func @for_nested_for_backedge(
func.func @for_nested_for_backedge(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    %r = scf.if %cond -> tensor<8x8xf32> {
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}
      %early = arith.addf %acc, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"}
      %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %cube = linalg.matmul {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
      // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"}
      %late = arith.subf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %next = arith.addf %early, %late {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %forward = scf.for %j = %c0 to %n step %c1 iter_args(%inner = %next) -> tensor<8x8xf32> {
        scf.yield %inner : tensor<8x8xf32>
      }
      scf.yield %forward : tensor<8x8xf32>
    } else {
      scf.yield %acc : tensor<8x8xf32>
    } {ssbuffer.core_type = "CUBE, VECTOR"}
    scf.yield %r : tensor<8x8xf32>
  } {ssbuffer.core_type = "CUBE, VECTOR"}
  return %r : tensor<8x8xf32>
}

// -----

// A nested while result follows condition forwarding, init and backedge.
// CHECK-LABEL: func.func @for_nested_while_backedge(
func.func @for_nested_while_backedge(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    %r = scf.if %cond -> tensor<8x8xf32> {
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "VECTOR"}
      %early = arith.addf %acc, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"}
      %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %cube = linalg.matmul {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
      // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "VECTOR"}
      %late = arith.subf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %next = arith.addf %early, %late {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %forward = scf.while (%before = %next) : (tensor<8x8xf32>) -> tensor<8x8xf32> {
        scf.condition(%cond) %before : tensor<8x8xf32>
      } do {
      ^bb0(%after: tensor<8x8xf32>):
        scf.yield %after : tensor<8x8xf32>
      }
      scf.yield %forward : tensor<8x8xf32>
    } else {
      scf.yield %acc : tensor<8x8xf32>
    } {ssbuffer.core_type = "CUBE, VECTOR"}
    scf.yield %r : tensor<8x8xf32>
  } {ssbuffer.core_type = "CUBE, VECTOR"}
  return %r : tensor<8x8xf32>
}

// -----

// An unchanged iter_arg has no producer and must not defer the early use.
// CHECK-LABEL: func.func @for_unchanged_iter_arg(
func.func @for_unchanged_iter_arg(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    %r = scf.if %cond -> tensor<8x8xf32> {
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "VECTOR"}
      %early = arith.addf %acc, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "VECTOR"}
      %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %cube = linalg.matmul {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
      // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"}
      %late = arith.subf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %next = arith.addf %early, %late {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      scf.yield %acc : tensor<8x8xf32>
    } else {
      scf.yield %acc : tensor<8x8xf32>
    } {ssbuffer.core_type = "CUBE, VECTOR"}
    scf.yield %r : tensor<8x8xf32>
  } {ssbuffer.core_type = "CUBE, VECTOR"}
  return %r : tensor<8x8xf32>
}

// -----

// An update forwarded from a function argument has no local producer.
// CHECK-LABEL: func.func @for_external_value(
func.func @for_external_value(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    %r = scf.if %cond -> tensor<8x8xf32> {
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"}
      %early = arith.addf %acc, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"}
      %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %cube = linalg.matmul {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
      // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 23 : i32, ssbuffer.core_type = "VECTOR"}
      %late = arith.subf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %next = arith.addf %early, %late {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      scf.yield %a : tensor<8x8xf32>
    } else {
      scf.yield %a : tensor<8x8xf32>
    } {ssbuffer.core_type = "CUBE, VECTOR"}
    scf.yield %r : tensor<8x8xf32>
  } {ssbuffer.core_type = "CUBE, VECTOR"}
  return %r : tensor<8x8xf32>
}

// -----

// After-region indexing excludes the condition and supports different before/after arities.
// CHECK-LABEL: func.func @while_permuted_condition_args(
func.func @while_permuted_condition_args(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r:3 = scf.while (%old = %a, %idx = %c0) : (tensor<8x8xf32>, index) -> (index, tensor<8x8xf32>, tensor<8x8xf32>) {
    scf.condition(%cond) %idx, %b, %old : index, tensor<8x8xf32>, tensor<8x8xf32>
  } do {
  ^bb0(%i: index, %aux: tensor<8x8xf32>, %acc: tensor<8x8xf32>):
    // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"}
    %early = arith.addf %acc, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
    // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"}
    %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
    %cube = linalg.matmul {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
    // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "VECTOR"}
    %late = arith.subf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
    %next = arith.addf %early, %late {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
    scf.yield %next, %i : tensor<8x8xf32>, index
  }
  return %r#2 : tensor<8x8xf32>
}

// -----

// Two branch-local producers must be lifted to their common scf.if ancestor.
// CHECK-LABEL: func.func @for_if_multiple_producers(
func.func @for_if_multiple_producers(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> tensor<8x8xf32> {
    %r = scf.if %cond -> tensor<8x8xf32> {
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"}
      %early = arith.addf %acc, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"}
      %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %cube = linalg.matmul {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
      // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"}
      %late = arith.subf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %next = arith.addf %early, %late {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %choice = scf.if %cond -> tensor<8x8xf32> {
        %then = arith.addf %next, %a {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
        scf.yield %then : tensor<8x8xf32>
      } else {
        %else = arith.subf %next, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
        scf.yield %else : tensor<8x8xf32>
      }
      scf.yield %choice : tensor<8x8xf32>
    } else {
      scf.yield %acc : tensor<8x8xf32>
    } {ssbuffer.core_type = "CUBE, VECTOR"}
    scf.yield %r : tensor<8x8xf32>
  } {ssbuffer.core_type = "CUBE, VECTOR"}
  return %r : tensor<8x8xf32>
}

// -----

// Map the second loop-carried argument through the second result of scf.if.
// CHECK-LABEL: func.func @for_if_second_result(
func.func @for_if_second_result(%cond: i1, %n: index, %a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%spare = %a, %acc = %a) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
    %r:2 = scf.if %cond -> (tensor<8x8xf32>, tensor<8x8xf32>) {
      // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"}
      %early = arith.addf %acc, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"}
      %indep = arith.mulf %a, %b {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %cube = linalg.matmul {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%a : tensor<8x8xf32>) -> tensor<8x8xf32>
      // CHECK: arith.subf {{.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"}
      %late = arith.subf %cube, %indep {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      %next = arith.addf %early, %late {ssbuffer.core_type = "VECTOR"} : tensor<8x8xf32>
      scf.yield %spare, %next : tensor<8x8xf32>, tensor<8x8xf32>
    } else {
      scf.yield %spare, %acc : tensor<8x8xf32>, tensor<8x8xf32>
    } {ssbuffer.core_type = "VECTOR, VECTOR"}
    scf.yield %r#0, %r#1 : tensor<8x8xf32>, tensor<8x8xf32>
  } {ssbuffer.core_type = "VECTOR, VECTOR"}
  return %r#1 : tensor<8x8xf32>
}
