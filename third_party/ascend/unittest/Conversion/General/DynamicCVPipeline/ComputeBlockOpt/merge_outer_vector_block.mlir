// RUN: triton-opt --split-input-file --merge-outer-vector-block %s | FileCheck %s

// Contract a tensor chain without changing either guard or its bypass value.
// CHECK-LABEL: func.func @guard_chain
// CHECK: scf.if %arg2
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 30
// CHECK: scf.yield %arg0
// CHECK: } {ssbuffer.block_id = 30
// CHECK: scf.if %arg3
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 30
// CHECK: } {ssbuffer.block_id = 30
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 30
func.func @guard_chain(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %p: i1, %q: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %x = scf.if %p -> (tensor<2x2xf32>) {
      %v = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      scf.yield %v : tensor<2x2xf32>
    } else {
      scf.yield %a : tensor<2x2xf32>
    } {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"}
    %y = scf.if %q -> (tensor<2x2xf32>) {
      %v = arith.mulf %x, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      scf.yield %v : tensor<2x2xf32>
    } else {
      scf.yield %x : tensor<2x2xf32>
    } {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"}
    %z = arith.addf %y, %b {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop}
  return
}

// -----

// An independent cube producer may be crossed: emit it before the contracted
// VECTOR group. The unselected sibling loop stays byte-for-byte in order.
// CHECK-LABEL: func.func @reorder_selected_body
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 15
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 20
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
// CHECK: ssbuffer.explicit_main_loop
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 40
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 50
func.func @reorder_selected_body(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %x = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    %m = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
    %y = arith.mulf %x, %m {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop}
  scf.for %iv = %c0 to %n step %c1 {
    %x = arith.addf %a, %b {ssbuffer.block_id = 40 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    %y = arith.mulf %x, %b {ssbuffer.block_id = 50 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  }
  return
}

// -----

// A direct tensor edge cannot be contracted around an intervening cube path.
// CHECK-LABEL: func.func @cube_cycle
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 10
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 15
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
func.func @cube_cycle(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %x = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    %m = linalg.matmul ins(%x, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
    %y = arith.mulf %x, %m {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop}
  return
}

// -----

// WAR + RAW through a shared buffer creates a path without any SSA connection
// to the intervening store. The tensor edge alone is insufficient to merge.
// CHECK-LABEL: func.func @memory_cycle
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 10
// CHECK: memref.load {{.*}}ssbuffer.block_id = 10
// CHECK: memref.store {{.*}}ssbuffer.block_id = 15
// CHECK: memref.load {{.*}}ssbuffer.block_id = 20
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
func.func @memory_cycle(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %buffer: memref<1xf32>, %value: f32, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %x = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    %before = memref.load %buffer[%c0] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xf32>
    memref.store %value, %buffer[%c0] {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<1xf32>
    %after = memref.load %buffer[%c0] {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xf32>
    %y = arith.mulf %x, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop}
  return
}

// -----

// Unknown effects are full fences even without memref operands or block_id.
func.func private @opaque()
// CHECK-LABEL: func.func @unknown_effect_fence
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 10
// CHECK: call @opaque
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
func.func @unknown_effect_fence(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %x = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    func.call @opaque() : () -> ()
    %y = arith.mulf %x, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop}
  return
}

// -----

// CHECK-LABEL: func.func @sync_fence
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 10
// CHECK: gpu.barrier
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
func.func @sync_fence(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %x = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    gpu.barrier
    %y = arith.mulf %x, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop}
  return
}

// -----

// CHECK-LABEL: func.func @unselected
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 10
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
func.func @unselected(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %x = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    %y = arith.mulf %x, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  }
  return
}

// -----

// Ordinary loop-carried dependencies remain outside the contraction experiment.
// CHECK-LABEL: func.func @loop_carried
// CHECK: iter_args
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 10
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
// CHECK: scf.yield
func.func @loop_carried(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %n: index) -> tensor<2x2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %iv = %c0 to %n step %c1 iter_args(%carry = %a) -> tensor<2x2xf32> {
    %x = arith.addf %carry, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    %y = arith.mulf %x, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    scf.yield %y : tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop}
  return %result : tensor<2x2xf32>
}

// -----

// Real SplitIfByBlockId bypasses have unclassified empty/fill/constants and
// comma-separated result core types. Keep those bypasses inside the same stage.
// CHECK-LABEL: func.func @split_if_placeholders
// CHECK: scf.if %arg2
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 20
// CHECK: tensor.empty() {ssbuffer.block_id = 20
// CHECK: linalg.fill {ssbuffer.block_id = 20
// CHECK: scf.yield
// CHECK: } {ssbuffer.block_id = 20
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
func.func @split_if_placeholders(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %p: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %x:2 = scf.if %p -> (tensor<2x2xf32>, tensor<2x2xf32>) {
      %v = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      scf.yield %v, %v : tensor<2x2xf32>, tensor<2x2xf32>
    } else {
      %empty = tensor.empty() {ssbuffer.block_id = 10 : i32} : tensor<2x2xf32>
      %zero = arith.constant 0.0 : f32
      %fill = linalg.fill {ssbuffer.block_id = 10 : i32} ins(%zero : f32) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
      scf.yield %fill, %a : tensor<2x2xf32>, tensor<2x2xf32>
    } {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR, VECTOR", ssbuffer.splitted_if = 0 : i32}
    %y = arith.mulf %x#0, %x#1 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop}
  return
}

// -----

// Result labels must not hide an unclassified tensor computation.
// CHECK-LABEL: func.func @untagged_matmul_rejected
// CHECK: linalg.matmul
// CHECK: } {ssbuffer.block_id = 10
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
func.func @untagged_matmul_rejected(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %p: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %x = scf.if %p -> (tensor<2x2xf32>) {
      %v = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) -> tensor<2x2xf32>
      scf.yield %v : tensor<2x2xf32>
    } else {
      scf.yield %a : tensor<2x2xf32>
    } {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.splitted_if = 0 : i32}
    %y = arith.mulf %x, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop}
  return
}

// -----

// The placeholder exception is only for tensor fills, never memref writes.
// CHECK-LABEL: func.func @memref_fill_rejected
// CHECK: linalg.fill
// CHECK: } {ssbuffer.block_id = 10
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
func.func @memref_fill_rejected(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %buffer: memref<2x2xf32>, %p: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %n step %c1 {
    %x = scf.if %p -> (tensor<2x2xf32>) {
      %v = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      scf.yield %v : tensor<2x2xf32>
    } else {
      %zero = arith.constant 0.0 : f32
      linalg.fill ins(%zero : f32) outs(%buffer : memref<2x2xf32>)
      scf.yield %a : tensor<2x2xf32>
    } {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.splitted_if = 0 : i32}
    %y = arith.mulf %x, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop}
  return
}

// -----

// Event-local tensor recurrence can merge a pure VECTOR chain. The scalar
// control prefix stays separate and the body/yield semantics are unchanged.
// CHECK-LABEL: func.func @event_carry_chain
// CHECK: iter_args
// CHECK: arith.addi {{.*}}ssbuffer.block_id = 5
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 20
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
// CHECK: scf.yield
func.func @event_carry_chain(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %n: index) -> tensor<2x2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%event = %c0, %carry = %a) -> (index, tensor<2x2xf32>) {
    %nextEvent = arith.addi %event, %c1 {ssbuffer.block_id = 5 : i32, ssbuffer.outer_event_control} : index
    %x = arith.addf %carry, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    %y = arith.mulf %x, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    scf.yield %nextEvent, %y : index, tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return %result#1 : tensor<2x2xf32>
}

// -----

// After contraction the old-carry reader must precede the complete writer
// group. Source order is preserved inside the contracted group.
// CHECK-LABEL: func.func @event_reader_precedes_merged_writer
// CHECK: test.tag = "old_reader"
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 20{{.*}}test.tag = "producer"
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20{{.*}}test.tag = "writer"
func.func @event_reader_precedes_merged_writer(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%event = %c0, %carry = %a) -> (index, tensor<2x2xf32>) {
    %x = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR", test.tag = "producer"} : tensor<2x2xf32>
    %next = arith.mulf %x, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR", test.tag = "writer"} : tensor<2x2xf32>
    %read = arith.addf %carry, %carry {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR", test.tag = "old_reader"} : tensor<2x2xf32>
    scf.yield %event, %next : index, tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}

// -----

// There is a direct VECTOR edge 10->30, but old-carry WAR adds 20->30 to
// SSA's 10->20 path. Contracting 10/30 would cycle through the CUBE reader.
// CHECK-LABEL: func.func @event_war_contraction_cycle
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 10
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 20
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 30
func.func @event_war_contraction_cycle(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%event = %c0, %carry = %a) -> (index, tensor<2x2xf32>) {
    %x = arith.addf %a, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    %read = linalg.matmul {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} ins(%x, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%carry : tensor<2x2xf32>) -> tensor<2x2xf32>
    %next = arith.mulf %x, %b {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    scf.yield %event, %next : index, tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}

// -----

// The event opt-in does not weaken unknown-effect fences.
// CHECK-LABEL: func.func @event_unknown_effect_fence
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 10
// CHECK: call @opaque_event
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 20
func.func private @opaque_event()
func.func @event_unknown_effect_fence(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%event = %c0, %carry = %a) -> (index, tensor<2x2xf32>) {
    %x = arith.addf %carry, %b {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    func.call @opaque_event() : () -> ()
    %next = arith.mulf %x, %b {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    scf.yield %event, %next : index, tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}

// -----

// The preparation stage no longer reads the old recurrence. The writer
// chooses the seed for a new task and otherwise consumes its own old value.
// CHECK-LABEL: func.func @event_seed_at_writer
// CHECK: iter_args(%{{.*}} = %{{.*}}, %[[OLD:.*]] = %arg0)
// CHECK: scf.yield %arg0
// CHECK: scf.yield %arg0
// CHECK: } {ssbuffer.block_id = 10
// CHECK: %[[LOCAL:.*]] = scf.if
// CHECK: scf.yield {{.*}}%arg0
// CHECK: scf.yield {{.*}}%[[OLD]]
// CHECK: } {ssbuffer.block_id = 20
// CHECK: arith.mulf %[[LOCAL]], %arg1
func.func @event_seed_at_writer(%seed: tensor<2x2xf32>, %tile: tensor<2x2xf32>, %n: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %res:2 = scf.for %i = %zero to %n step %one iter_args(%phase = %zero, %old = %seed) -> (index, tensor<2x2xf32>) {
    %first = arith.cmpi eq, %phase, %zero {ssbuffer.block_id = 5 : i32, ssbuffer.outer_event_control} : index
    %entry = scf.if %first -> (tensor<2x2xf32>) {
      scf.yield %seed : tensor<2x2xf32>
    } else {
      scf.yield %old : tensor<2x2xf32>
    } {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"}
    %next = arith.mulf %entry, %tile {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    scf.yield %one, %next : index, tensor<2x2xf32>
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}

// -----

// A resultless, Vector-only epilogue can consume the accumulator directly.
// CHECK-LABEL: func.func @event_store_epilogue
// CHECK: %[[ACC:.*]] = arith.addf {{.*}}ssbuffer.block_id = 20
// CHECK: bufferization.materialize_in_destination %[[ACC]] in writable %arg1
// CHECK: } {ssbuffer.block_id = 20
func.func @event_store_epilogue(%tile: tensor<2x2xf32>, %dest: memref<2x2xf32>, %last: i1, %n: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %n step %one {
    %acc = arith.addf %tile, %tile {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    scf.if %last {
      bufferization.materialize_in_destination %acc in writable %dest {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<2x2xf32>, memref<2x2xf32>) -> ()
    } {ssbuffer.block_id = 20 : i32}
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}
