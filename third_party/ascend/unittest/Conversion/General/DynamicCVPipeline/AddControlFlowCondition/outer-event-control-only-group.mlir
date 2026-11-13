// RUN: triton-opt --process-args %s | FileCheck %s
// RUN: triton-opt --process-args --create-if-ops %s | FileCheck %s --check-prefix=READY

// Group 17 only supplies the control template. Keep the exact common trip
// count, make group 6 its own old-state copy, and omit the empty actor.
// CHECK-LABEL: func.func @remove_control_template(
// CHECK: %{{.*}}:2 = scf.for {{.*}} to %arg2 {{.*}}iter_args(%{{.*}} = %{{.*}}, %[[TASK:arg[0-9]+]] = %{{.*}})
// CHECK-NOT: ssbuffer.block_id = 17
// CHECK: memref.store %[[TASK]], {{.*}}ssbuffer.block_id = 6 : i32
// CHECK: arith.index_cast %[[TASK]] {{.*}}ssbuffer.block_id = 6 : i32
// CHECK: memref.load {{.*}}ssbuffer.block_id = 6 : i32
// CHECK-NOT: ssbuffer.block_id = 17
// CHECK: return
// READY-LABEL: func.func @remove_control_template(
// READY: memref.store
// READY: memref.load
// READY: } {{.*}}ssbuffer.if = 6 : i32}
// READY-NOT: ssbuffer.if = 17
// READY: return
func.func @remove_control_template(%counts: memref<?xi32>, %out: memref<1xi32>, %total: i64) {
  %zero = arith.constant 0 : i32
  %zero_index = arith.constant 0 : index
  %zero64 = arith.constant 0 : i64
  %one64 = arith.constant 1 : i64
  %result = scf.for %event = %zero64 to %total step %one64 iter_args(%task = %zero) -> i32 : i64 {
    // Original slot arithmetic is dead after CloneOps makes the actual
    // consumer independent. Its transfer provenance must not keep an actor.
    %quotient = arith.divsi %event, %one64 {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 0 : i32} : i64
    %two = arith.constant {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 0 : i32} 2 : i64
    %slot = arith.remsi %quotient, %two {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 0 : i32} : i64
    %is_zero_slot = arith.cmpi eq, %slot, %zero64 {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 0 : i32} : i64
    %offset = arith.index_cast %task {ssbuffer.block_id = 17 : i32, ssbuffer.dep_mark = [2 : i32, 3 : i32], ssbuffer.outer_event_control} : i32 to index
    %count = memref.load %counts[%offset] {ssbuffer.block_id = 17 : i32, ssbuffer.outer_event_control} : memref<?xi32>
    %next = arith.addi %task, %count {ssbuffer.block_id = 17 : i32, ssbuffer.outer_event_control} : i32
    memref.store %task, %out[%zero_index] {ssbuffer.block_id = 6 : i32} : memref<1xi32>
    scf.yield %next : i32
  } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}

// A GM write makes the group an actor, even when its other operations are only
// event control. Preserve both stores and their source order.
// CHECK-LABEL: func.func @keep_group_with_gm_write(
// CHECK: %{{.*}}:3 = scf.for
// CHECK: memref.store {{.*}}ssbuffer.block_id = 17 : i32
// CHECK: memref.store {{.*}}ssbuffer.block_id = 6 : i32
// CHECK: return
// READY-LABEL: func.func @keep_group_with_gm_write(
// READY: memref.store
// READY: } {{.*}}ssbuffer.if = 17 : i32}
// READY: memref.store
// READY: } {{.*}}ssbuffer.if = 6 : i32}
func.func @keep_group_with_gm_write(%out: memref<2xi32>, %total: i64) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %zero_index = arith.constant 0 : index
  %one_index = arith.constant 1 : index
  %zero64 = arith.constant 0 : i64
  %one64 = arith.constant 1 : i64
  %result = scf.for %event = %zero64 to %total step %one64 iter_args(%task = %zero) -> i32 : i64 {
    memref.store %task, %out[%zero_index] {ssbuffer.block_id = 17 : i32} : memref<2xi32>
    %next = arith.addi %task, %one {ssbuffer.block_id = 17 : i32, ssbuffer.outer_event_control} : i32
    memref.store %task, %out[%one_index] {ssbuffer.block_id = 6 : i32} : memref<2xi32>
    scf.yield %next : i32
  } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}

// Even scalar-only protocol annotations prevent dropping a group.
// CHECK-LABEL: func.func @keep_protocol_group(
// CHECK: %{{.*}}:3 = scf.for
// CHECK: ssbuffer.block_id = 17 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]
// READY-LABEL: func.func @keep_protocol_group(
// READY: } {{.*}}ssbuffer.if = 17 : i32}
// READY: } {{.*}}ssbuffer.if = 6 : i32}
func.func @keep_protocol_group(%out: memref<1xi32>, %total: i64) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %zero_index = arith.constant 0 : index
  %zero64 = arith.constant 0 : i64
  %one64 = arith.constant 1 : i64
  %result = scf.for %event = %zero64 to %total step %one64 iter_args(%task = %zero) -> i32 : i64 {
    %next = arith.addi %task, %one {ssbuffer.block_id = 17 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32], ssbuffer.outer_event_control} : i32
    memref.store %task, %out[%zero_index] {ssbuffer.block_id = 6 : i32} : memref<1xi32>
    scf.yield %next : i32
  } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}

// The original control result has an external observer, so preserve its owner.
// CHECK-LABEL: func.func @keep_observed_control_result(
// CHECK: %{{.*}}:3 = scf.for
// CHECK: ssbuffer.block_id = 17 : i32
// READY-LABEL: func.func @keep_observed_control_result(
// READY: } {{.*}}ssbuffer.if = 17 : i32}
// READY: } {{.*}}ssbuffer.if = 6 : i32}
func.func @keep_observed_control_result(%out: memref<1xi32>, %total: i64) -> i32 {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %zero_index = arith.constant 0 : index
  %zero64 = arith.constant 0 : i64
  %one64 = arith.constant 1 : i64
  %result = scf.for %event = %zero64 to %total step %one64 iter_args(%task = %zero) -> i32 : i64 {
    %next = arith.addi %task, %one {ssbuffer.block_id = 17 : i32, ssbuffer.outer_event_control} : i32
    memref.store %task, %out[%zero_index] {ssbuffer.block_id = 6 : i32} : memref<1xi32>
    scf.yield %next : i32
  } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return %result : i32
}

// Transfer arithmetic feeding the state transition is not a dead template.
// CHECK-LABEL: func.func @keep_live_transfer_control(
// CHECK: %{{.*}}:3 = scf.for
// CHECK: arith.addi {{.*}}ssbuffer.block_id = 17 : i32{{.*}}ssbuffer.transfer_id = 0 : i32
// CHECK: memref.store
// READY-LABEL: func.func @keep_live_transfer_control(
// READY: } {{.*}}ssbuffer.if = 17 : i32}
// READY: } {{.*}}ssbuffer.if = 6 : i32}
func.func @keep_live_transfer_control(%out: memref<1xi32>, %total: i64) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %zero_index = arith.constant 0 : index
  %zero64 = arith.constant 0 : i64
  %one64 = arith.constant 1 : i64
  %result = scf.for %event = %zero64 to %total step %one64 iter_args(%task = %zero) -> i32 : i64 {
    %next = arith.addi %task, %one {ssbuffer.block_id = 17 : i32, ssbuffer.outer_event_control, ssbuffer.transfer_id = 0 : i32} : i32
    memref.store %task, %out[%zero_index] {ssbuffer.block_id = 6 : i32} : memref<1xi32>
    scf.yield %next : i32
  } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}

// With no payload actor, preserve both control owners. In particular the
// original result must still have a first actor's yielded state to select.
// CHECK-LABEL: func.func @keep_all_control_groups(
// CHECK: %{{.*}}:3 = scf.for {{.*}} to %arg0
// CHECK: ssbuffer.block_id = 17 : i32
// CHECK: ssbuffer.block_id = 6 : i32
// CHECK: return
// READY-LABEL: func.func @keep_all_control_groups(
// READY: } {{.*}}ssbuffer.if = 17 : i32}
// READY: } {{.*}}ssbuffer.if = 6 : i32}
// READY: return
func.func @keep_all_control_groups(%total: i64) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %zero64 = arith.constant 0 : i64
  %one64 = arith.constant 1 : i64
  %result = scf.for %event = %zero64 to %total step %one64 iter_args(%task = %zero) -> i32 : i64 {
    %next17 = arith.addi %task, %one {ssbuffer.block_id = 17 : i32, ssbuffer.outer_event_control} : i32
    %next6 = arith.addi %task, %one {ssbuffer.block_id = 6 : i32, ssbuffer.outer_event_control} : i32
    scf.yield %next17 : i32
  } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}

// The same scalar group remains on the legacy non-event path.
// CHECK-LABEL: func.func @legacy_control_group(
// CHECK: ssbuffer.block_id = 17 : i32
// READY-LABEL: func.func @legacy_control_group(
// READY: } {{.*}}ssbuffer.if = 17 : i32}
// READY: } {{.*}}ssbuffer.if = 6 : i32}
func.func @legacy_control_group(%out: memref<1xi32>, %total: i64) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %zero_index = arith.constant 0 : index
  %zero64 = arith.constant 0 : i64
  %one64 = arith.constant 1 : i64
  %result = scf.for %event = %zero64 to %total step %one64 iter_args(%task = %zero) -> i32 : i64 {
    %next = arith.addi %task, %one {ssbuffer.block_id = 17 : i32} : i32
    memref.store %task, %out[%zero_index] {ssbuffer.block_id = 6 : i32} : memref<1xi32>
    scf.yield %next : i32
  } {ssbuffer.main_loop = 0 : i32}
  return
}

// The counted event loop retains its common trip count. Each stage receives
// a coupled task/j/phase/bound group; its slot expression continues to use the
// for IV, which the scheduler later replaces with that stage's event counter.
// CHECK-LABEL: func.func @counted_event_control(
// CHECK: %{{.*}}:12 = scf.for %[[EVENT:.*]] = %{{.*}} to %[[TOTAL:.*]] step %{{.*}} iter_args(%[[T:.*]] = %{{.*}}, %[[J:.*]] = %{{.*}}, %[[P:.*]] = %{{.*}}, %[[B:.*]] = %{{.*}}, %[[T1:.*]] = %{{.*}}, %[[J1:.*]] = %{{.*}}, %[[P1:.*]] = %{{.*}}, %[[B1:.*]] = %{{.*}}, %[[T2:.*]] = %{{.*}}, %[[J2:.*]] = %{{.*}}, %[[P2:.*]] = %{{.*}}, %[[B2:.*]] = %{{.*}})
// CHECK: arith.remsi %[[EVENT]], {{.*}}ssbuffer.block_id = 1 : i32
// CHECK: arith.addi %[[T1]], %[[J1]] {ssbuffer.block_id = 1 : i32}
// CHECK: %[[ACTIVE1:.*]] = arith.cmpi eq, %[[P1]], {{.*}}ssbuffer.arg = 0 : i32, ssbuffer.block_id = 1 : i32
// CHECK: %[[NEXT1:.*]]:4 = scf.if %[[ACTIVE1]]
// CHECK: arith.index_cast %[[T1]] : i32 to index
// CHECK: memref.load
// CHECK: scf.yield %[[T1]],
// CHECK: arith.addi %[[J1]],
// CHECK: arith.cmpi sge, %{{.*}}, %[[B1]]
// CHECK: } {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 1 : i32}
// CHECK: arith.remsi %[[EVENT]], {{.*}}ssbuffer.block_id = 2 : i32
// CHECK: arith.addi %[[T2]], %[[J2]] {ssbuffer.block_id = 2 : i32}
// CHECK: %[[ACTIVE2:.*]] = arith.cmpi eq, %[[P2]], {{.*}}ssbuffer.arg = 0 : i32, ssbuffer.block_id = 2 : i32
// CHECK: %[[NEXT2:.*]]:4 = scf.if %[[ACTIVE2]]
// CHECK: arith.index_cast %[[T2]] : i32 to index
// CHECK: memref.load
// CHECK: scf.yield %[[T2]],
// CHECK: arith.addi %[[J2]],
// CHECK: arith.cmpi sge, %{{.*}}, %[[B2]]
// CHECK: } {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 2 : i32}
// CHECK: scf.yield %[[NEXT1]]#0, %[[NEXT1]]#1, %[[NEXT1]]#2, %[[NEXT1]]#3, %[[NEXT1]]#0, %[[NEXT1]]#1, %[[NEXT1]]#2, %[[NEXT1]]#3, %[[NEXT2]]#0, %[[NEXT2]]#1, %[[NEXT2]]#2, %[[NEXT2]]#3

// Both complete control groups are inside their own stage's execution guard.
// READY-LABEL: func.func @counted_event_control(
// READY: scf.for
// READY: scf.if
// READY: memref.store
// READY: scf.if
// READY: memref.load
// READY: } {{.*}}ssbuffer.if = 1 : i32}
// READY: scf.if
// READY: memref.store
// READY: scf.if
// READY: memref.load
// READY: } {{.*}}ssbuffer.if = 2 : i32}

func.func @counted_event_control(%counts: memref<?xi32>, %out1: memref<2xi32>,
                                 %out2: memref<2xi32>, %total: i64) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %zero64 = arith.constant 0 : i64
  %one64 = arith.constant 1 : i64
  %two64 = arith.constant 2 : i64
  %result:4 = scf.for %event = %zero64 to %total step %one64
      iter_args(%task = %zero, %j = %zero, %phase = %zero, %bound = %zero)
      -> (i32, i32, i32, i32) : i64 {
    %slot1 = arith.remsi %event, %two64 {ssbuffer.block_id = 1 : i32} : i64
    %idx1 = arith.index_cast %slot1 {ssbuffer.block_id = 1 : i32} : i64 to index
    %payload1 = arith.addi %task, %j {ssbuffer.block_id = 1 : i32} : i32
    memref.store %payload1, %out1[%idx1] {ssbuffer.block_id = 1 : i32} : memref<2xi32>
    %initial = arith.cmpi eq, %phase, %zero {ssbuffer.block_id = 1 : i32} : i32
    %next:4 = scf.if %initial -> (i32, i32, i32, i32) {
      %offset = arith.index_cast %task : i32 to index
      %loaded = memref.load %counts[%offset] : memref<?xi32>
      scf.yield %task, %zero, %one, %loaded : i32, i32, i32, i32
    } else {
      %inc = arith.addi %j, %one : i32
      %last = arith.cmpi sge, %inc, %bound : i32
      %task_inc = arith.addi %task, %one : i32
      %task_next = arith.select %last, %task_inc, %task : i32
      %j_next = arith.select %last, %zero, %inc : i32
      %phase_next = arith.select %last, %zero, %phase : i32
      scf.yield %task_next, %j_next, %phase_next, %bound : i32, i32, i32, i32
    } {ssbuffer.block_id = 1 : i32}
    %slot2 = arith.remsi %event, %two64 {ssbuffer.block_id = 2 : i32} : i64
    %idx2 = arith.index_cast %slot2 {ssbuffer.block_id = 2 : i32} : i64 to index
    %payload2 = arith.addi %task, %j {ssbuffer.block_id = 2 : i32} : i32
    memref.store %payload2, %out2[%idx2] {ssbuffer.block_id = 2 : i32} : memref<2xi32>
    scf.yield %next#0, %next#1, %next#2, %next#3 : i32, i32, i32, i32
  } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop,
     ssbuffer.outer_event_control_count = 4 : i32}
  return
}
