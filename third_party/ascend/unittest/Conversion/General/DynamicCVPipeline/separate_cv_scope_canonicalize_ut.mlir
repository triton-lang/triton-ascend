// RUN: triton-opt %s  --separate-cv-scope --preserve-control-attrs-canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// SeparateCVScope + canonicalize focused tests
//
// These cases keep values alive by storing into sink memrefs passed as function
// arguments, instead of local alloc/store chains that canonicalize can delete as
// dead. Control-flow tests use dynamic conditions or dynamic loop bounds, and
// the owning-side results are consumed after the control-flow op.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @basic_scope_split_and_cleanup(
// CHECK: scope.scope : () -> () {
// CHECK: arith.addi
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: arith.muli
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @basic_scope_split_and_cleanup(%vec_in: i32, %cube_in: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c2v = arith.constant {ssbuffer.core_type = "VECTOR"} 2 : i32
    %c3v = arith.constant {ssbuffer.core_type = "VECTOR"} 3 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0 = arith.addi %vec_in, %c2v {ssbuffer.core_type = "VECTOR"} : i32
    %1 = arith.addi %0, %c3v {ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.muli %cube_in, %c2c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @container_if_without_core_type(
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @container_if_without_core_type(%cond: i1, %vec_in: i32, %cube_in: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c2v = arith.constant {ssbuffer.core_type = "VECTOR"} 2 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    scf.if %cond {
      %0 = arith.addi %vec_in, %c2v {ssbuffer.core_type = "VECTOR"} : i32
      %1 = arith.muli %cube_in, %c2c {ssbuffer.core_type = "CUBE"} : i32
      memref.store %0, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
      memref.store %1, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    } else {
      %0 = arith.subi %vec_in, %c2v {ssbuffer.core_type = "VECTOR"} : i32
      %1 = arith.addi %cube_in, %c2c {ssbuffer.core_type = "CUBE"} : i32
      memref.store %0, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
      memref.store %1, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    }
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @mixed_if_yield_neutralize_scalar(
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: scf.yield
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: scf.yield
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @mixed_if_yield_neutralize_scalar(%cond: i1, %vec_in: i32, %cube_in: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0:2 = scf.if %cond -> (i32, i32) {
      %1 = arith.addi %vec_in, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      %2 = arith.addi %cube_in, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %2 : i32, i32
    } else {
      %1 = arith.subi %vec_in, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      %2 = arith.subi %cube_in, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %2 : i32, i32
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %1 = arith.addi %0#0, %c1v {ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.addi %0#1, %c2c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @mixed_for_iter_args_index_neutralize(
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @mixed_for_iter_args_index_neutralize(%lb: i32, %ub: i32, %vec_init: i32, %cube_init: index, %outv: memref<1xi32>, %outc: memref<1xindex>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %stepv = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %stepc = arith.constant {ssbuffer.core_type = "CUBE"} 2 : index
    %0:2 = scf.for %i = %lb to %ub step %stepv iter_args(%a = %vec_init, %b = %cube_init) -> (i32, index) : i32 {
      %1 = arith.addi %a, %stepv {ssbuffer.core_type = "VECTOR"} : i32
      %2 = arith.addi %b, %stepc {ssbuffer.core_type = "CUBE"} : index
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %2 : i32, index
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %1 = arith.addi %0#0, %stepv {ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.addi %0#1, %stepc {ssbuffer.core_type = "CUBE"} : index
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xindex>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @for_forwarding_use_can_be_ignored(
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @for_forwarding_use_can_be_ignored(%lb: i32, %ub: i32, %vec_init: i32, %cube_init: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c1c = arith.constant {ssbuffer.core_type = "CUBE"} 1 : i32
    %0:2 = scf.for %i = %lb to %ub step %c1v iter_args(%a = %vec_init, %b = %cube_init) -> (i32, i32) : i32 {
      %1 = arith.addi %a, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      %2 = arith.addi %b, %c1c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %2 : i32, i32
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %1:2 = scf.for %j = %lb to %ub step %c1v iter_args(%x = %0#0, %y = %0#1) -> (i32, i32) : i32 {
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %x, %y : i32, i32
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %2 = arith.addi %1#0, %c1v {ssbuffer.core_type = "VECTOR"} : i32
    %3 = arith.addi %1#1, %c1c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %2, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %3, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @for_cross_slot_yield_use_does_not_preserve_source(
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK-NOT: arith.addf
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @for_cross_slot_yield_use_does_not_preserve_source(%lb: i32, %ub: i32, %vec_src: f32, %vec_dst: f32, %cube_init: i32, %outv: memref<1xf32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %vf1 = arith.constant {ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c1c = arith.constant {ssbuffer.core_type = "CUBE"} 1 : i32
    %0:3 = scf.for %i = %lb to %ub step %c1v iter_args(%a = %vec_src, %b = %vec_dst, %c = %cube_init) -> (f32, f32, i32) : i32 {
      %1 = arith.addf %a, %vf1 {ssbuffer.core_type = "VECTOR"} : f32
      %2 = arith.addf %1, %b {ssbuffer.core_type = "VECTOR"} : f32
      %3 = arith.addi %c, %c1c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, VECTOR, CUBE"} %1, %2, %3 : f32, f32, i32
    } {ssbuffer.core_type = "VECTOR, VECTOR, CUBE"}
    %1 = arith.addf %0#1, %vf1 {ssbuffer.core_type = "VECTOR"} : f32
    %2 = arith.addi %0#2, %c1c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xf32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @while_dependency_preserved_for_cube(
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: scf.while
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @while_dependency_preserved_for_cube(%cond: i1, %vec_init: i64, %cube_init: i64, %limit: i64, %outv: memref<1xi64>, %outc: memref<1xi64>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i64
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i64
    %0 = scf.if %cond -> (i64) {
      %1 = arith.addi %vec_init, %c1v {ssbuffer.core_type = "VECTOR"} : i64
      scf.yield {ssbuffer.core_type = "VECTOR"} %1 : i64
    } else {
      %1 = arith.subi %limit, %c1v {ssbuffer.core_type = "VECTOR"} : i64
      scf.yield {ssbuffer.core_type = "VECTOR"} %1 : i64
    } {ssbuffer.core_type = "VECTOR"}
    %1:3 = scf.while (%a = %vec_init, %b = %cube_init, %c = %0) : (i64, i64, i64) -> (i64, i64, i64) {
      %2 = arith.addi %a, %c {ssbuffer.core_type = "VECTOR"} : i64
      %3 = arith.cmpi slt, %2, %limit {ssbuffer.core_type = "VECTOR"} : i64
      scf.condition(%3) %2, %b, %c : i64, i64, i64
    } do {
    ^bb0(%a_iter: i64, %b_iter: i64, %c_iter: i64):
      %2 = arith.addi %b_iter, %c2c {ssbuffer.core_type = "CUBE"} : i64
      scf.yield {ssbuffer.core_type = "CUBE, CUBE, VECTOR"} %a_iter, %2, %c_iter : i64, i64, i64
    } attributes {ssbuffer.core_type = "CUBE, CUBE, VECTOR"}
    %2 = arith.addi %1#1, %c2c {ssbuffer.core_type = "CUBE"} : i64
    %3 = arith.addi %1#2, %c1v {ssbuffer.core_type = "VECTOR"} : i64
    memref.store %3, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi64>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi64>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @while_inner_if_terminator_use_blocks_dead_shell_erase(
// CHECK: scope.scope : () -> () {
// CHECK: scf.while
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.while
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @while_inner_if_terminator_use_blocks_dead_shell_erase(%if_cond: i1, %vec_init: i32, %cube_init: i32, %limit: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0:2 = scf.while (%v = %vec_init, %c = %cube_init) : (i32, i32) -> (i32, i32) {
      %keep = arith.cmpi slt, %v, %limit {ssbuffer.core_type = "VECTOR"} : i32
      scf.condition(%keep) %v, %c : i32, i32
    } do {
    ^bb0(%v_iter: i32, %c_iter: i32):
      %next_v = scf.if %if_cond -> (i32) {
        %inc_v = arith.addi %v_iter, %c1v {ssbuffer.core_type = "VECTOR"} : i32
        scf.yield {ssbuffer.core_type = "VECTOR"} %inc_v : i32
      } else {
        scf.yield {ssbuffer.core_type = "VECTOR"} %v_iter : i32
      } {ssbuffer.core_type = "VECTOR"}
      %next_c = arith.addi %c_iter, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %next_v, %next_c : i32, i32
    } attributes {ssbuffer.core_type = "VECTOR, CUBE"}
    %1 = arith.addi %0#0, %c1v {ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.addi %0#1, %c2c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @while_forwarding_use_can_be_ignored(
// CHECK: scope.scope : () -> () {
// CHECK: scf.while
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.while
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @while_forwarding_use_can_be_ignored(%vec_init: i64, %cube_init: i64, %limit: i64, %outv: memref<1xi64>, %outc: memref<1xi64>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i64
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i64
    %0:2 = scf.while (%v = %vec_init, %c = %cube_init) : (i64, i64) -> (i64, i64) {
      %keep = arith.cmpi slt, %v, %limit {ssbuffer.core_type = "VECTOR"} : i64
      scf.condition(%keep) %v, %c : i64, i64
    } do {
    ^bb0(%v_iter: i64, %c_iter: i64):
      %nv = arith.addi %v_iter, %c1v {ssbuffer.core_type = "VECTOR"} : i64
      %nc = arith.addi %c_iter, %c2c {ssbuffer.core_type = "CUBE"} : i64
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %nv, %nc : i64, i64
    } attributes {ssbuffer.core_type = "VECTOR, CUBE"}
    %1:2 = scf.while (%x = %0#0, %y = %0#1) : (i64, i64) -> (i64, i64) {
      %keep = arith.cmpi slt, %x, %limit {ssbuffer.core_type = "VECTOR"} : i64
      scf.condition(%keep) %x, %y : i64, i64
    } do {
    ^bb0(%x_iter: i64, %y_iter: i64):
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %x_iter, %y_iter : i64, i64
    } attributes {ssbuffer.core_type = "VECTOR, CUBE"}
    %2 = arith.addi %1#0, %c1v {ssbuffer.core_type = "VECTOR"} : i64
    %3 = arith.addi %1#1, %c2c {ssbuffer.core_type = "CUBE"} : i64
    memref.store %2, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi64>
    memref.store %3, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi64>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @for_same_slot_loop_carry_preserved(
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: scf.if
// CHECK: scf.while
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: %[[FOR:.*]]:2 = scf.for
// CHECK: %[[IF:.*]]:2 = scf.if
// CHECK: %[[WHILE:.*]]:2 = scf.while
// CHECK: arith.extsi
// CHECK: scf.yield %[[WHILE]]#0, %[[WHILE]]#1 : i32, i64
// CHECK: scf.yield %[[IF]]#0, %[[IF]]#1 : i32, i64
// CHECK: memref.store %[[FOR]]#1
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @for_same_slot_loop_carry_preserved(%lb: i32, %ub: i32, %cond: i1, %vec_init: i32, %cube_init: i64, %outv: memref<1xi32>, %outc: memref<1xi64>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c2v = arith.constant {ssbuffer.core_type = "VECTOR"} 2 : i32
    %c1c = arith.constant {ssbuffer.core_type = "CUBE"} 1 : i64
    %0:2 = scf.for %i = %lb to %ub step %c1v iter_args(%state = %vec_init, %cube = %cube_init) -> (i32, i64) : i32 {
      %1:2 = scf.if %cond -> (i32, i64) {
        %2 = arith.addi %state, %c2v {ssbuffer.core_type = "VECTOR"} : i32
        %3:2 = scf.while (%iter = %state, %acc = %cube) : (i32, i64) -> (i32, i64) {
          %4 = arith.cmpi slt, %iter, %2 {ssbuffer.core_type = "VECTOR"} : i32
          scf.condition(%4) %iter, %acc : i32, i64
        } do {
        ^bb0(%iter_arg: i32, %acc_arg: i64):
          %4 = arith.addi %iter_arg, %c1v {ssbuffer.core_type = "VECTOR"} : i32
          %5 = arith.extsi %iter_arg {ssbuffer.core_type = "CUBE"} : i32 to i64
          %6 = arith.addi %acc_arg, %5 {ssbuffer.core_type = "CUBE"} : i64
          scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %4, %6 : i32, i64
        } attributes {ssbuffer.core_type = "VECTOR, CUBE"}
        scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %3#0, %3#1 : i32, i64
      } else {
        scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %state, %cube : i32, i64
      } {ssbuffer.core_type = "VECTOR, CUBE"}
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1#0, %1#1 : i32, i64
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    memref.store %0#0, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %0#1, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi64>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @memref_slot_neutralize_uses_alloc(
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: memref.load
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @memref_slot_neutralize_uses_alloc(%cond: i1, %src: memref<4xi32>, %vec_seed: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %cube_idx = arith.constant {ssbuffer.core_type = "CUBE"} 2 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %0:2 = scf.if %cond -> (i32, memref<4xi32>) {
      %1 = arith.addi %vec_seed, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %src : i32, memref<4xi32>
    } else {
      %1 = arith.subi %vec_seed, %c1v {ssbuffer.core_type = "VECTOR"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %1, %src : i32, memref<4xi32>
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %1 = arith.addi %0#0, %c1v {ssbuffer.core_type = "VECTOR"} : i32
    %2 = memref.load %0#1[%cube_idx] {ssbuffer.core_type = "CUBE"} : memref<4xi32>
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @static_tensor_slot_neutralize_uses_dense_zero(
// CHECK: scope.scope : () -> () {
// CHECK: arith.addf
// CHECK: bufferization.materialize_in_destination
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @static_tensor_slot_neutralize_uses_dense_zero(%cond: i1, %src: tensor<4xf32>, %cube_seed: i32, %outv: memref<4xf32>, %outc: memref<1xi32>) {
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0:2 = scf.if %cond -> (tensor<4xf32>, i32) {
      %1 = arith.addi %cube_seed, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %src, %1 : tensor<4xf32>, i32
    } else {
      %1 = arith.subi %cube_seed, %c1v {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %src, %1 : tensor<4xf32>, i32
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    %1 = arith.addf %0#0, %src {ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    %2 = arith.addi %0#1, %c2c {ssbuffer.core_type = "CUBE"} : i32
    bufferization.materialize_in_destination %1 in writable %outv {ssbuffer.core_type = "VECTOR"} : (tensor<4xf32>, memref<4xf32>) -> ()
    memref.store %2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @cross_scope_live_user_keeps_producer(
// CHECK: scope.scope : () -> () {
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: arith.muli
// CHECK-NOT: arith.addi
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @cross_scope_live_user_keeps_producer(%cube_seed: i32, %vec_seed: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0 = arith.muli %cube_seed, %c2c {ssbuffer.core_type = "CUBE"} : i32
    %1 = arith.addi %0, %vec_seed {ssbuffer.core_type = "VECTOR"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %0, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}

// -----

// CHECK-LABEL: func.func @short_core_type_list_falls_back_to_first(
// CHECK: scope.scope : () -> () {
// CHECK-NOT: scf.if
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: scf.yield
// CHECK: memref.store
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @short_core_type_list_falls_back_to_first(%cond: i1, %vec_seed: i32, %cube_seed0: i32, %cube_seed1: i32, %outv: memref<1xi32>, %outc0: memref<1xi32>, %outc1: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc0 = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %idxc1 = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c2c = arith.constant {ssbuffer.core_type = "CUBE"} 2 : i32
    %0:2 = scf.if %cond -> (i32, i32) {
      %1 = arith.addi %cube_seed0, %c2c {ssbuffer.core_type = "CUBE"} : i32
      %2 = arith.subi %cube_seed1, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "CUBE"} %1, %2 : i32, i32
    } else {
      %1 = arith.subi %cube_seed0, %c2c {ssbuffer.core_type = "CUBE"} : i32
      %2 = arith.addi %cube_seed1, %c2c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "CUBE"} %1, %2 : i32, i32
    } {ssbuffer.core_type = "CUBE"}
    %1 = arith.addi %vec_seed, %c1v {ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.addi %0#0, %c2c {ssbuffer.core_type = "CUBE"} : i32
    %3 = arith.addi %0#1, %c2c {ssbuffer.core_type = "CUBE"} : i32
    memref.store %1, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    memref.store %2, %outc0[%idxc0] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    memref.store %3, %outc1[%idxc1] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }
}
