// RUN: triton-opt %s --add-scalar-pipe-control | FileCheck %s

// AddScalarPipeControl guards the PIPE_S flag=15 syncs that live in the body
// of the innermost main_loop (scf.for / scf.while carrying ssbuffer.main_loop)
// with a single `scf.if`.
//
// The guard condition is a non-blocking readiness check built on
// `hivm.hir.try_wait`: the query observes the PIPE_S counter without
// consuming its token, so the blocking wait stays inside the guard.
//
// The guard spans everything between the PIPE_S wait at the head of the loop
// body and the PIPE_S set right before the terminator, so the whole body ends
// up inside one guard.
//
// The loop carries values across iterations, so the guard has to forward them:
// the then branch yields the values the loop used to yield, and the else
// branch yields the iter args, i.e. skipping the body keeps the loop state.
//
// PIPE_S syncs that live outside of a main_loop body are left untouched.
//
// The module targets an A5 device: `try_wait` is only available on the
// register based architectures.

// CHECK-LABEL: func.func @for_main_loop
// CHECK: scf.for
// CHECK: %[[C:.*]] = hivm.hir.try_wait flag = 15, sync_mode = <intra_block> : i64
// CHECK: %[[Z:.*]] = arith.constant 0 : i64
// CHECK: %[[R:.*]] = arith.cmpi ne, %[[C]], %[[Z]] : i64
// CHECK: scf.if %[[R]] -> (index) {
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// A sync on another pipe in between is guarded as well.
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield
// CHECK: }
// The whole body is guarded by a single scf.if.
// CHECK-NOT: scf.if
// CHECK-LABEL: func.func @while_main_loop
// CHECK: scf.while
// The before region stays untouched.
// CHECK: scf.condition
// The guard lives inside the do region, where the PIPE_S syncs are.
// CHECK: do {
// CHECK: ^bb0
// A CUBE side wait observes the counters of the two VECTOR sub-blocks, so
// both flag 15 and flag 15 + 16 have to be queried.
// CHECK: %[[C15:.*]] = hivm.hir.try_wait flag = 15, sync_mode = <intra_block> : i64
// CHECK: %[[Z15:.*]] = arith.constant 0 : i64
// CHECK: %[[R15:.*]] = arith.cmpi ne, %[[C15]], %[[Z15]] : i64
// CHECK: %[[C31:.*]] = hivm.hir.try_wait flag = 31, sync_mode = <intra_block> : i64
// CHECK: %[[Z31:.*]] = arith.constant 0 : i64
// CHECK: %[[R31:.*]] = arith.cmpi ne, %[[C31]], %[[Z31]] : i64
// CHECK: %[[COND:.*]] = arith.andi %[[R15]], %[[R31]] : i1
// CHECK: scf.if %[[COND]] -> (index) {
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
// The op in between is guarded as well.
// CHECK: arith.addi
// CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield
// CHECK: }
// CHECK-NOT: scf.if
// CHECK-LABEL: func.func @pipe_s_outside_main_loop
// A PIPE_S sync outside of a main_loop body stays unguarded.
// CHECK-NOT: hivm.hir.try_wait
// CHECK-NOT: scf.if
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-NOT: scf.if
// CHECK-LABEL: func.func @for_multi_results
// A loop carrying several values of different types forwards all of them
// through the guard.
// CHECK: scf.for
// CHECK: %[[MC:.*]] = hivm.hir.try_wait flag = 15, sync_mode = <intra_block> : i64
// CHECK: %[[MZ:.*]] = arith.constant 0 : i64
// CHECK: %[[MR:.*]] = arith.cmpi ne, %[[MC]], %[[MZ]] : i64
// CHECK: scf.if %[[MR]] -> (index, f32) {
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: arith.addi
// CHECK: arith.addf
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: scf.yield {{.*}} : index, f32
// CHECK: } else {
// The else branch yields the iter args, i.e. the loop state is kept.
// CHECK: scf.yield {{.*}} : index, f32
// CHECK: }
// CHECK-NOT: scf.if
// CHECK-LABEL: func.func @for_no_results
// A loop without iter args has nothing to forward, so the guard has no
// results and no else branch.
// CHECK: scf.for
// CHECK: %[[NC:.*]] = hivm.hir.try_wait flag = 15, sync_mode = <intra_block> : i64
// CHECK: %[[NZ:.*]] = arith.constant 0 : i64
// CHECK: %[[NR:.*]] = arith.cmpi ne, %[[NC]], %[[NZ]] : i64
// CHECK: scf.if %[[NR]] {
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: }
// CHECK-NOT: scf.if
// CHECK-LABEL: func.func @for_two_sync_pairs
// Only the first wait/set pair is guarded; the second pair stays outside.
// CHECK: scf.for
// CHECK: %[[TC:.*]] = hivm.hir.try_wait flag = 15, sync_mode = <intra_block> : i64
// CHECK: %[[TZ:.*]] = arith.constant 0 : i64
// CHECK: %[[TR:.*]] = arith.cmpi ne, %[[TC]], %[[TZ]] : i64
// CHECK: scf.if %[[TR]] -> (index) {
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: arith.addi
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield
// CHECK: }
// The second pair is not guarded, so it has no query either.
// CHECK-NOT: hivm.hir.try_wait
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: arith.addi
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-NOT: scf.if
// CHECK-LABEL: func.func @nested_main_loop
// Only the innermost main_loop is guarded, the outer one is left alone.
// CHECK: scf.for
// The outer PIPE_S wait stays unguarded.
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// The inner main_loop carries the guard.
// CHECK: %[[IC:.*]] = hivm.hir.try_wait flag = 15, sync_mode = <intra_block> : i64
// CHECK: %[[IZ:.*]] = arith.constant 0 : i64
// CHECK: %[[IR:.*]] = arith.cmpi ne, %[[IC]], %[[IZ]] : i64
// CHECK: scf.if %[[IR]] -> (index) {
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: arith.addi
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield
// CHECK: }
// CHECK-NOT: scf.if
// CHECK-LABEL: func.func @while_no_results
// A while loop that carries no value has nothing to forward, so the guard
// has no results and no else branch.
// CHECK: scf.while
// CHECK: scf.condition
// CHECK: do {
// CHECK: %[[WC:.*]] = hivm.hir.try_wait flag = 15, sync_mode = <intra_block> : i64
// CHECK: %[[WZ:.*]] = arith.constant 0 : i64
// CHECK: %[[WR:.*]] = arith.cmpi ne, %[[WC]], %[[WZ]] : i64
// CHECK: scf.if %[[WR]] {
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: }
// CHECK-NOT: scf.if
// CHECK-LABEL: func.func @while_multi_results
// A while loop carries its values in the do region, the guard forwards all
// of them.
// CHECK: scf.while
// CHECK: %[[WMC:.*]] = hivm.hir.try_wait flag = 15, sync_mode = <intra_block> : i64
// CHECK: %[[WMZ:.*]] = arith.constant 0 : i64
// CHECK: %[[WMR:.*]] = arith.cmpi ne, %[[WMC]], %[[WMZ]] : i64
// CHECK: scf.if %[[WMR]] -> (index, f32) {
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: arith.addi
// CHECK: arith.addf
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: scf.yield {{.*}} : index, f32
// CHECK: } else {
// CHECK: scf.yield {{.*}} : index, f32
// CHECK: }
// CHECK-NOT: scf.if
// CHECK-LABEL: func.func @for_nested_region_use
// A nested region inside the guarded range is moved along with it, and the
// value it produces is still consumed inside the range.
// CHECK: scf.for
// CHECK: %[[FC:.*]] = hivm.hir.try_wait flag = 15, sync_mode = <intra_block> : i64
// CHECK: %[[FZ:.*]] = arith.constant 0 : i64
// CHECK: %[[FR:.*]] = arith.cmpi ne, %[[FC]], %[[FZ]] : i64
// CHECK: scf.if %[[FR]] -> (index) {
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: scf.for
// CHECK: arith.addi
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield
// CHECK: }
// CHECK-NOT: scf.if
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @for_main_loop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      scf.yield %arg2 : index
    } {ssbuffer.block_id = 15 : i32, ssbuffer.main_loop = 0 : i32}
    return
  }

  func.func @while_main_loop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.while (%arg1 = %c0) : (index) -> index {
      %1 = arith.cmpi slt, %arg1, %c1 : index
      scf.condition(%1) %arg1 : index
    } do {
    ^bb0(%arg1: index):
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
      %2 = arith.addi %arg1, %c1 : index
      hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
      scf.yield %2 : index
    } attributes {ssbuffer.block_id = 16 : i32, ssbuffer.main_loop = 0 : i32}
    return
  }

  func.func @pipe_s_outside_main_loop() {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    return
  }

  func.func @for_multi_results() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %0:2 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0, %arg3 = %f0) -> (index, f32) {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      %1 = arith.addi %arg2, %c1 : index
      %2 = arith.addf %arg3, %f0 : f32
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      scf.yield %1, %2 : index, f32
    } {ssbuffer.block_id = 17 : i32, ssbuffer.main_loop = 0 : i32}
    return
  }

  func.func @for_no_results() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %c1 step %c1 {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    } {ssbuffer.block_id = 18 : i32, ssbuffer.main_loop = 0 : i32}
    return
  }

  func.func @for_two_sync_pairs() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      %1 = arith.addi %arg2, %c1 : index
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      %2 = arith.addi %arg2, %c1 : index
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      scf.yield %1 : index
    } {ssbuffer.block_id = 19 : i32, ssbuffer.main_loop = 0 : i32}
    return
  }

  func.func @nested_main_loop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      %1 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %c0) -> (index) {
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
        %2 = arith.addi %arg4, %c1 : index
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
        scf.yield %2 : index
      } {ssbuffer.block_id = 21 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      scf.yield %arg2 : index
    } {ssbuffer.block_id = 20 : i32, ssbuffer.main_loop = 0 : i32}
    return
  }

  func.func @while_no_results() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.while : () -> () {
      %1 = arith.cmpi slt, %c0, %c1 : index
      scf.condition(%1)
    } do {
    ^bb0:
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      scf.yield
    } attributes {ssbuffer.block_id = 22 : i32, ssbuffer.main_loop = 0 : i32}
    return
  }

  func.func @while_multi_results() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %0:2 = scf.while (%arg1 = %c0, %arg2 = %f0) : (index, f32) -> (index, f32) {
      %1 = arith.cmpi slt, %arg1, %c1 : index
      scf.condition(%1) %arg1, %arg2 : index, f32
    } do {
    ^bb0(%arg1: index, %arg2: f32):
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      %2 = arith.addi %arg1, %c1 : index
      %3 = arith.addf %arg2, %f0 : f32
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      scf.yield %2, %3 : index, f32
    } attributes {ssbuffer.block_id = 23 : i32, ssbuffer.main_loop = 0 : i32}
    return
  }

  func.func @for_nested_region_use() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      %1 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %c0) -> (index) {
        scf.yield %arg4 : index
      }
      %2 = arith.addi %1, %c1 : index
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      scf.yield %2 : index
    } {ssbuffer.block_id = 28 : i32, ssbuffer.main_loop = 0 : i32}
    return
  }
}
