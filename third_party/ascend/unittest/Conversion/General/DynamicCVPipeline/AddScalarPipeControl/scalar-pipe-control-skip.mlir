// RUN: triton-opt %s --add-scalar-pipe-control | FileCheck %s

// Negative cases for AddScalarPipeControl: the pass must leave the IR
// untouched whenever it cannot build a correct guard. It never fails the
// pass and never produces a half-guarded loop.
//
// The module targets an A5 device: `try_wait` is only available on the
// register based architectures, so on the memory based ones the pass would
// bail out early and these cases would pass for the wrong reason.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @escape_value() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    %1 = arith.addi %arg2, %c1 : index
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    %2 = arith.muli %1, %c1 : index
    scf.yield %2 : index
  } {ssbuffer.block_id = 15 : i32, ssbuffer.main_loop = 0 : i32}
  return
}

// The value produced inside the guarded range is used after the PIPE_S set,
// i.e. outside of the range. Moving the computation into the guard would
// break that use, so the pass gives up and leaves the loop alone.
// CHECK-LABEL: func.func @escape_value
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: arith.addi
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: arith.muli
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait

func.func @missing_set() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    scf.yield %arg2 : index
  } {ssbuffer.block_id = 16 : i32, ssbuffer.main_loop = 0 : i32}
  return
}

// A PIPE_S wait without a matching PIPE_S set is not a complete pair.
// CHECK-LABEL: func.func @missing_set
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait

func.func @other_flag() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 2
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 2
    scf.yield %arg2 : index
  } {ssbuffer.block_id = 17 : i32, ssbuffer.main_loop = 0 : i32}
  return
}

// Only flag 15 is the reserved PIPE_S scalar sync, other flags are left
// alone.
// CHECK-LABEL: func.func @other_flag
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 2
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 2
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait

func.func @other_pipe() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 15
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 15
    scf.yield %arg2 : index
  } {ssbuffer.block_id = 18 : i32, ssbuffer.main_loop = 0 : i32}
  return
}

// Only PIPE_S on both sides is the scalar sync, other pipes are left alone.
// CHECK-LABEL: func.func @other_pipe
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 15
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 15
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait

func.func @missing_wait() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    scf.yield %arg2 : index
  } {ssbuffer.block_id = 24 : i32, ssbuffer.main_loop = 0 : i32}
  return
}

// A PIPE_S set without a matching PIPE_S wait is not a complete pair.
// CHECK-LABEL: func.func @missing_wait
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait

func.func @set_before_wait() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    scf.yield %arg2 : index
  } {ssbuffer.block_id = 25 : i32, ssbuffer.main_loop = 0 : i32}
  return
}

// The set comes before the wait, so the wait has no set to close it.
// CHECK-LABEL: func.func @set_before_wait
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait

func.func @not_main_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    scf.yield %arg2 : index
  } {ssbuffer.block_id = 26 : i32}
  return
}

// Only a loop tagged as main_loop is guarded, a plain loop is left alone.
// CHECK-LABEL: func.func @not_main_loop
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait

func.func @sync_in_nested_region() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
    %1 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %c0) -> (index) {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      scf.yield %arg4 : index
    }
    scf.yield %1 : index
  } {ssbuffer.block_id = 27 : i32, ssbuffer.main_loop = 0 : i32}
  return
}

// The syncs are not direct children of the main loop body, so there is no
// pair to guard.
// CHECK-LABEL: func.func @sync_in_nested_region
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait

func.func @escape_into_nested_region() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    %1 = arith.addi %arg2, %c1 : index
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    %2 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %c0) -> (index) {
      %3 = arith.muli %1, %c1 : index
      scf.yield %3 : index
    }
    scf.yield %2 : index
  } {ssbuffer.block_id = 29 : i32, ssbuffer.main_loop = 0 : i32}
  return
}
}

// The value is used from a nested region that sits outside of the guarded
// range, so it escapes.
// CHECK-LABEL: func.func @escape_into_nested_region
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: arith.addi
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: arith.muli
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait
