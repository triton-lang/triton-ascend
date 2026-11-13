// RUN: triton-opt --split-input-file --reorder-outer-event-carry %s | FileCheck %s

// The full phase consumes the previous partial state; the epilogue consumes
// both previous states. Block numbers deliberately disagree with safe order.
// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK-LABEL: func.func @old_partial_full_and_epilogue
// CHECK: test.tag = "epilogue"
// CHECK: test.tag = "full"
// CHECK: test.tag = "partial_first"
// CHECK: test.tag = "partial_second"
func.func @old_partial_full_and_epilogue(%n: index, %init: tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%partial = %init, %full = %init) -> (tensor<4xf32>, tensor<4xf32>) {
    %p0 = arith.addf %partial, %partial {ssbuffer.block_id = 20 : i32, test.tag = "partial_first"} : tensor<4xf32>
    %p1 = arith.mulf %p0, %p0 {ssbuffer.block_id = 20 : i32, test.tag = "partial_second"} : tensor<4xf32>
    %f = arith.addf %partial, %full {ssbuffer.block_id = 16 : i32, test.tag = "full"} : tensor<4xf32>
    %e = arith.mulf %full, %partial {ssbuffer.block_id = 15 : i32, test.tag = "epilogue"} : tensor<4xf32>
    scf.yield %p1, %f : tensor<4xf32>, tensor<4xf32>
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
  return
}

// -----

// Nested uses resolve to the complete direct-child block, and conditional
// forwarding resolves to the real writer. The predicate and branches stay put.
// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK-LABEL: func.func @conditional_writer
// CHECK: test.tag = "reader"
// CHECK: test.tag = "writer"
func.func @conditional_writer(%n: index, %cond: i1, %init: tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %next = scf.if %cond -> tensor<4xf32> {
      %v = arith.addf %old, %old : tensor<4xf32>
      scf.yield %v : tensor<4xf32>
    } else {
      scf.yield %old : tensor<4xf32>
    } {ssbuffer.block_id = 2 : i32, test.tag = "writer"}
    scf.if %cond {
      %read = arith.mulf %old, %old : tensor<4xf32>
    } {ssbuffer.block_id = 1 : i32, test.tag = "reader"}
    scf.yield %next : tensor<4xf32>
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
  return
}

// -----

// The same carry convention applies to a normalized while's after region.
// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK-LABEL: func.func @while_old_carry
// CHECK: test.tag = "reader"
// CHECK: test.tag = "writer"
func.func @while_old_carry(%cond: i1, %init: tensor<4xf32>) {
  %r = scf.while (%old = %init) : (tensor<4xf32>) -> tensor<4xf32> {
    scf.condition(%cond) %old : tensor<4xf32>
  } do {
  ^bb0(%old: tensor<4xf32>):
    %next = arith.addf %old, %old {ssbuffer.block_id = 2 : i32, test.tag = "writer"} : tensor<4xf32>
    %read = arith.mulf %old, %old {ssbuffer.block_id = 1 : i32, test.tag = "reader"} : tensor<4xf32>
    scf.yield %next : tensor<4xf32>
  } attributes {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
  return
}

// -----

// A reader needing both old and new state creates a real SSA/WAR cycle.
// Fallback leaves the original body untouched.
// CHECK: module attributes {triton_ascend.dynamic_cv_pipeline.rc = 2 : i32}
// CHECK-LABEL: func.func @ssa_cycle
// CHECK: test.tag = "writer"
// CHECK: test.tag = "reader"
func.func @ssa_cycle(%n: index, %init: tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %next = arith.addf %old, %old {ssbuffer.block_id = 2 : i32, test.tag = "writer"} : tensor<4xf32>
    %read = arith.mulf %old, %next {ssbuffer.block_id = 1 : i32, test.tag = "reader"} : tensor<4xf32>
    scf.yield %next : tensor<4xf32>
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
  return
}

// -----

// Do not repair carry order by violating a RAW memory dependency.
// CHECK: module attributes {triton_ascend.dynamic_cv_pipeline.rc = 2 : i32}
// CHECK-LABEL: func.func @memory_cycle
// CHECK: test.tag = "writer"
// CHECK: memref.store
// CHECK: memref.load
// CHECK: test.tag = "reader"
func.func @memory_cycle(%n: index, %init: tensor<4xf32>, %mem: memref<1xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %next = arith.addf %old, %old {ssbuffer.block_id = 2 : i32, test.tag = "writer"} : tensor<4xf32>
    memref.store %v, %mem[%c0] {ssbuffer.block_id = 2 : i32} : memref<1xf32>
    %load = memref.load %mem[%c0] {ssbuffer.block_id = 1 : i32} : memref<1xf32>
    %read = tensor.insert %load into %old[%c0] {ssbuffer.block_id = 1 : i32, test.tag = "reader"} : tensor<4xf32>
    scf.yield %next : tensor<4xf32>
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
  return
}

// -----

// Unknown calls remain complete fences, even when no memref is in the body.
// CHECK: module attributes {triton_ascend.dynamic_cv_pipeline.rc = 2 : i32}
// CHECK-LABEL: func.func @unknown_effect_cycle
// CHECK: test.tag = "writer"
// CHECK: call @unknown
// CHECK: test.tag = "reader"
func.func private @unknown()
func.func @unknown_effect_cycle(%n: index, %init: tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %next = arith.addf %old, %old {ssbuffer.block_id = 2 : i32, test.tag = "writer"} : tensor<4xf32>
    func.call @unknown() : () -> ()
    %read = arith.mulf %old, %old {ssbuffer.block_id = 1 : i32, test.tag = "reader"} : tensor<4xf32>
    scf.yield %next : tensor<4xf32>
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
  return
}

// -----

// Synchronization cannot be crossed to satisfy a carry anti-dependency.
// CHECK: module attributes {triton_ascend.dynamic_cv_pipeline.rc = 2 : i32}
// CHECK-LABEL: func.func @sync_cycle
// CHECK: test.tag = "writer"
// CHECK: gpu.barrier
// CHECK: test.tag = "reader"
func.func @sync_cycle(%n: index, %init: tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %next = arith.addf %old, %old {ssbuffer.block_id = 2 : i32, test.tag = "writer"} : tensor<4xf32>
    gpu.barrier
    %read = arith.mulf %old, %old {ssbuffer.block_id = 1 : i32, test.tag = "reader"} : tensor<4xf32>
    scf.yield %next : tensor<4xf32>
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
  return
}

// -----

// Ordinary loops are outside the new pass's contract.
// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK-LABEL: func.func @legacy_unchanged
// CHECK: test.tag = "writer"
// CHECK: test.tag = "reader"
func.func @legacy_unchanged(%n: index, %init: tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> tensor<4xf32> {
    %next = arith.addf %old, %old {ssbuffer.block_id = 2 : i32, test.tag = "writer"} : tensor<4xf32>
    %read = arith.mulf %old, %old {ssbuffer.block_id = 1 : i32, test.tag = "reader"} : tensor<4xf32>
    scf.yield %next : tensor<4xf32>
  }
  return
}
