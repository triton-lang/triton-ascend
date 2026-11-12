// RUN: triton-opt --process-args %s | FileCheck %s --check-prefix=REUSE
// RUN: triton-opt --process-args --create-if-ops %s | FileCheck %s --check-prefix=READY
// RUN: sed 's|// ADD-WRITE: ||' %s | triton-opt --process-args | FileCheck %s --check-prefix=WRITE
// RUN: sed 's|// ADD-WRITE: ||' %s | triton-opt --process-args --create-if-ops | FileCheck %s --check-prefix=WRAP-WRITE
// RUN: sed 's|// ADD-SYNC: ||' %s | triton-opt --process-args | FileCheck %s --check-prefix=SYNC
// RUN: sed 's|// ADD-SYNC: ||' %s | triton-opt --process-args --create-if-ops | FileCheck %s --check-prefix=WRAP-SYNC

// Reuse the same old-state bound, except across a write or synchronization.
// REUSE-LABEL: func.func @control_reads
// REUSE: iter_args({{.*}}%[[TASK:arg[0-9]+]] = {{.*}}, %[[J:arg[0-9]+]] = {{.*}}, %[[PHASE:arg[0-9]+]] = {{.*}}, %[[BOUND:arg[0-9]+]] = {{.*}})
// REUSE: %[[FIRST:.*]] = arith.cmpi eq, %[[PHASE]],
// REUSE: %[[LOADED:.*]] = scf.if %[[FIRST]] -> (i32)
// REUSE: arith.index_cast %[[TASK]] : i32 to index
// REUSE: memref.load
// REUSE: scf.yield %[[BOUND]] : i32
// REUSE: tensor.splat %[[LOADED]]
// REUSE-NOT: memref.load
// REUSE: arith.cmpi slt, {{.*}}, %[[LOADED]]
// REUSE-NOT: memref.load
// REUSE: return
// READY-LABEL: func.func @control_reads
// READY: scf.if
// READY: memref.load
// READY-NOT: memref.load
// READY: } {{.*}}ssbuffer.if = 1 : i32}
// WRITE-LABEL: func.func @control_reads
// WRITE: memref.load
// WRITE: memref.store
// WRITE: memref.load
// WRITE-NOT: memref.load
// WRITE: return
// WRAP-WRITE-LABEL: func.func @control_reads
// WRAP-WRITE: scf.if
// WRAP-WRITE: memref.load
// WRAP-WRITE: memref.store
// WRAP-WRITE: memref.load
// WRAP-WRITE: } {{.*}}ssbuffer.if = 1 : i32}
// SYNC-LABEL: func.func @control_reads
// SYNC: memref.load
// SYNC: hivm.hir.sync_block_wait
// SYNC: memref.load
// SYNC-NOT: memref.load
// SYNC: return
// WRAP-SYNC-LABEL: func.func @control_reads
// WRAP-SYNC: scf.if
// WRAP-SYNC: memref.load
// WRAP-SYNC: hivm.hir.sync_block_wait
// WRAP-SYNC: memref.load
// WRAP-SYNC: } {{.*}}ssbuffer.if = 1 : i32}
func.func @control_reads(%counts: memref<?xi32>, %total: i64) -> tensor<1xi32> {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %zero_index = arith.constant 0 : index
  %zero64 = arith.constant 0 : i64
  %one64 = arith.constant 1 : i64
  %seed = arith.constant dense<0> : tensor<1xi32>
  %result:5 = scf.for %event = %zero64 to %total step %one64
      iter_args(%task = %zero, %j = %zero, %phase = %zero, %bound = %zero, %value = %seed)
      -> (i32, i32, i32, i32, tensor<1xi32>) : i64 {
    %first = arith.cmpi eq, %phase, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.outer_event_control} : i32
    %loaded = scf.if %first -> (i32) {
      %offset = arith.index_cast %task : i32 to index
      %count = memref.load %counts[%offset] : memref<?xi32>
      scf.yield %count : i32
    } else {
      scf.yield %bound : i32
    } {ssbuffer.block_id = 1 : i32, ssbuffer.outer_event_control}
    %payload = tensor.splat %loaded {ssbuffer.block_id = 1 : i32} : tensor<1xi32>
    // ADD-WRITE: memref.store %one, %counts[%zero_index] {ssbuffer.block_id = 1 : i32} : memref<?xi32>
    // ADD-SYNC: hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32}[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    %first_copy = arith.cmpi eq, %phase, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.outer_event_control} : i32
    %loaded_copy = scf.if %first_copy -> (i32) {
      %offset = arith.index_cast %task : i32 to index
      %count = memref.load %counts[%offset] : memref<?xi32>
      scf.yield %count : i32
    } else {
      scf.yield %bound : i32
    } {ssbuffer.block_id = 1 : i32, ssbuffer.outer_event_control}
    %j_next = arith.addi %j, %one {ssbuffer.block_id = 1 : i32, ssbuffer.outer_event_control} : i32
    %more_task = arith.cmpi slt, %j_next, %loaded_copy {ssbuffer.block_id = 1 : i32, ssbuffer.outer_event_control} : i32
    %task_inc = arith.addi %task, %one {ssbuffer.block_id = 1 : i32, ssbuffer.outer_event_control} : i32
    %task_next = arith.select %more_task, %task, %task_inc {ssbuffer.block_id = 1 : i32, ssbuffer.outer_event_control} : i32
    %more_phase = arith.cmpi slt, %j_next, %loaded_copy {ssbuffer.block_id = 1 : i32, ssbuffer.outer_event_control} : i32
    %phase_next = arith.select %more_phase, %one, %zero {ssbuffer.block_id = 1 : i32, ssbuffer.outer_event_control} : i32
    scf.yield %task_next, %j_next, %phase_next, %loaded_copy, %payload : i32, i32, i32, i32, tensor<1xi32>
  } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 4 : i32}
  return %result#4 : tensor<1xi32>
}
