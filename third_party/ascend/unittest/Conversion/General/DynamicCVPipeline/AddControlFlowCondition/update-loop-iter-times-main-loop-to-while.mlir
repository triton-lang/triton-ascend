// RUN: triton-opt --update-loop-iter-times --split-input-file %s | FileCheck %s

// UpdateLoopIterTimes rewrites the main loops from scf.for into scf.while.
// Only the scf.for carrying the `ssbuffer.main_loop` tag is rewritten. Every
// other scf.for is left untouched.

// CHECK-LABEL: func.func @test_main_loop_for_to_while
// CHECK: scf.while
// CHECK-SAME: : (index, f32) -> (index, f32)
// CHECK: arith.cmpi slt
// CHECK: scf.condition
// CHECK: arith.addi
// CHECK: scf.yield
// CHECK: } attributes {ssbuffer.main_loop = 0 : i32}
// CHECK-NOT: scf.for

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_main_loop_for_to_while(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0.000000e+00 : f32
    %0 = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %init) -> (f32) {
      %next = arith.addf %acc, %init : f32
      scf.yield %next : f32
    } {ssbuffer.main_loop = 0 : i32}
    return
  }
}

// -----

// A main loop whose computation blocks carry a counter: the while keeps the
// upper bound of the for loop and its condition is the OR of the counters of
// the blocks compared against that upper bound. The loop therefore exits only
// when every block reached the upper bound.

// CHECK-LABEL: func.func @test_main_loop_condition_from_counters
// CHECK: scf.while
// CHECK-SAME: : (i32, i32, i32) -> (i32, i32, i32)
// CHECK: %[[CMP1:[^ ]+]] = arith.cmpi slt, %[[CTR1:[^ ,]+]], %[[UB:[^ ,]+]] : i32
// CHECK: %[[CMP2:[^ ]+]] = arith.cmpi slt, %[[CTR2:[^ ,]+]], %[[UB]] : i32
// CHECK: arith.ori %[[CMP1]], %[[CMP2]] : i1
// CHECK: scf.condition
// CHECK-NOT: scf.for

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_main_loop_condition_from_counters() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %0, %1 = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32
        iter_args(%ctr1 = %c0_i32, %ctr2 = %c0_i32) -> (i32, i32) : i32 {
      %cmp1 = arith.cmpi slt, %ctr1, %c2_i32 : i32
      %cond1 = arith.andi %cmp1, %true : i1
      %n1 = scf.if %cond1 -> (i32) {
        %inc = arith.addi %ctr1, %c1_i32 : i32
        scf.yield %inc : i32
      } else {
        scf.yield %ctr1 : i32
      } {ssbuffer.if = 1 : i32}
      %cmp2 = arith.cmpi slt, %ctr2, %c2_i32 : i32
      %cond2 = arith.andi %cmp2, %true : i1
      %n2 = scf.if %cond2 -> (i32) {
        %inc = arith.addi %ctr2, %c1_i32 : i32
        scf.yield %inc : i32
      } else {
        scf.yield %ctr2 : i32
      } {ssbuffer.if = 2 : i32}
      scf.yield %n1, %n2 : i32, i32
    } {ssbuffer.main_loop = 0 : i32}
    return
  }
}

// -----

// A main loop without iter_args only carries the induction variable, so the
// resulting scf.while has a single (unused) result.

// CHECK-LABEL: func.func @test_main_loop_without_iter_args
// CHECK: scf.while
// CHECK-SAME: : (index) -> index
// CHECK: } attributes {ssbuffer.main_loop = 1 : i32}
// CHECK-NOT: scf.for

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_main_loop_without_iter_args(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %arg0 step %c1 {
      %v = arith.addi %i, %i : index
    } {ssbuffer.main_loop = 1 : i32}
    return
  }
}

// -----

// An untagged scf.for is not converted.

// CHECK-LABEL: func.func @test_for_without_main_loop
// CHECK: scf.for
// CHECK-NOT: scf.while

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_for_without_main_loop(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0.000000e+00 : f32
    %0 = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %init) -> (f32) {
      %next = arith.addf %acc, %init : f32
      scf.yield %next : f32
    }
    return
  }
}

// -----

// Only the tagged outer loop is converted; the untagged inner loop stays an
// scf.for.

// CHECK-LABEL: func.func @test_nested_for_without_main_loop
// CHECK: scf.while
// CHECK-SAME: : (index, f32) -> (index, f32)
// CHECK: scf.for
// CHECK: } attributes {ssbuffer.main_loop = 0 : i32}
// CHECK-NOT: scf.while

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_nested_for_without_main_loop(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0.000000e+00 : f32
    %0 = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %init) -> (f32) {
      %1 = scf.for %j = %c0 to %arg0 step %c1 iter_args(%inner = %acc) -> (f32) {
        %next = arith.addf %inner, %init : f32
        scf.yield %next : f32
      }
      %2 = arith.addf %1, %init : f32
      scf.yield %2 : f32
    } {ssbuffer.main_loop = 0 : i32}
    return
  }
}
