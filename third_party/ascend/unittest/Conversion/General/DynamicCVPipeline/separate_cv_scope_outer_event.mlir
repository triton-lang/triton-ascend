// RUN: triton-opt %s --separate-cv-scope --preserve-control-attrs-canonicalize | FileCheck %s

// Control has VECTOR provenance before separation, but both engines need the
// same count and the complete task/j/phase/bound transition. The marker must
// preserve scalar subtrees without also preserving the other engine's store.
// CHECK-LABEL: func.func @shared_event_control(
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.load
// CHECK: ssbuffer.outer_event_control
// CHECK: scf.for
// CHECK: scf.if
// CHECK: memref.load
// CHECK: ssbuffer.outer_event_control
// CHECK: arith.addi
// CHECK: memref.store
// CHECK: ssbuffer.outer_event_control_count = 4 : i32
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.load
// CHECK: ssbuffer.outer_event_control
// CHECK: scf.for
// CHECK: scf.if
// CHECK: memref.load
// CHECK: ssbuffer.outer_event_control
// CHECK: arith.muli
// CHECK: memref.store
// CHECK: ssbuffer.outer_event_control_count = 4 : i32
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module attributes {ssbuffer.outer_main_loop} {
func.func @shared_event_control(%counts: memref<?xi32>, %rows: index,
                                %outv: memref<1xi32>, %outc: memref<1xi32>) {
  %zero = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
  %one = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : index
  %zero32 = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : i32
  %one32 = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
  %total = scf.for %row = %zero to %rows step %one iter_args(%sum = %zero) -> index {
    %loaded = memref.load %counts[%row] {ssbuffer.core_type = "VECTOR"} : memref<?xi32>
    %wide = arith.index_cast %loaded {ssbuffer.core_type = "VECTOR"} : i32 to index
    %next = arith.addi %sum, %wide {ssbuffer.core_type = "VECTOR"} : index
    scf.yield {ssbuffer.core_type = "VECTOR"} %next : index
  } {ssbuffer.core_type = "VECTOR", ssbuffer.outer_event_control}
  %result:4 = scf.for %event = %zero to %total step %one
      iter_args(%task = %zero32, %j = %zero32, %phase = %zero32, %bound = %zero32)
      -> (i32, i32, i32, i32) {
    %enter = arith.cmpi eq, %phase, %zero32 {ssbuffer.core_type = "VECTOR", ssbuffer.outer_event_control} : i32
    %next_bound = scf.if %enter -> i32 {
      %offset = arith.index_cast %task {ssbuffer.core_type = "VECTOR"} : i32 to index
      %loaded = memref.load %counts[%offset] {ssbuffer.core_type = "VECTOR"} : memref<?xi32>
      scf.yield {ssbuffer.core_type = "VECTOR"} %loaded : i32
    } else {
      scf.yield {ssbuffer.core_type = "VECTOR"} %bound : i32
    } {ssbuffer.core_type = "VECTOR", ssbuffer.outer_event_control}
    %inc = arith.addi %j, %one32 {ssbuffer.core_type = "VECTOR", ssbuffer.outer_event_control} : i32
    %last = arith.cmpi sge, %inc, %bound {ssbuffer.core_type = "VECTOR", ssbuffer.outer_event_control} : i32
    %task_inc = arith.addi %task, %one32 {ssbuffer.core_type = "VECTOR", ssbuffer.outer_event_control} : i32
    %next_task = arith.select %last, %task_inc, %task {ssbuffer.core_type = "VECTOR", ssbuffer.outer_event_control} : i32
    %next_j = arith.select %last, %zero32, %inc {ssbuffer.core_type = "VECTOR", ssbuffer.outer_event_control} : i32
    %next_phase = arith.select %last, %zero32, %one32 {ssbuffer.core_type = "VECTOR", ssbuffer.outer_event_control} : i32
    %active = arith.cmpi ne, %phase, %zero32 {ssbuffer.core_type = "VECTOR", ssbuffer.outer_event_control} : i32
    scf.if %active {
      %v = arith.addi %task, %j {ssbuffer.core_type = "VECTOR"} : i32
      memref.store %v, %outv[%zero] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
      %c = arith.muli %task, %j {ssbuffer.core_type = "CUBE"} : i32
      memref.store %c, %outc[%zero] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    }
    scf.yield {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR, VECTOR"} %next_task, %next_j, %next_phase, %next_bound : i32, i32, i32, i32
  } {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR, VECTOR", ssbuffer.explicit_main_loop,
     ssbuffer.outer_event_control_count = 4 : i32}
  return
}
}
