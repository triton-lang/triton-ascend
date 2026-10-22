// RUN: triton-opt %s --separate-cv-scope | FileCheck %s

// Regression for the structural-user erase crash in SeparateCVScope:
// a VECTOR-only scf.if feeds the iter_args of a mixed VECTOR/CUBE scf.for.
// In the CUBE clone the scf.if shell becomes semantically dead before the loop
// is fully rewritten, so the pass must not erase the shell while the raw SSA
// use from the downstream scf.for still exists.

// CHECK-LABEL: func.func @if_results_feed_followup_loop_structural_user_repro(
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @if_results_feed_followup_loop_structural_user_repro(
      %lb: i32,
      %ub: i32,
      %cond: i1,
      %vec0: tensor<4xf32>,
      %vec1: tensor<4xf32>,
      %cube_init: i32,
      %outv: memref<1xf32>,
      %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c1c = arith.constant {ssbuffer.core_type = "CUBE"} 1 : i32

    %0:2 = scf.if %cond -> (tensor<4xf32>, tensor<4xf32>) {
      %1 = arith.addf %vec0, %vec1 {ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
      %2 = arith.mulf %1, %vec1 {ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
      scf.yield {ssbuffer.core_type = "VECTOR, VECTOR"} %1, %2 : tensor<4xf32>, tensor<4xf32>
    } else {
      scf.yield {ssbuffer.core_type = "VECTOR, VECTOR"} %vec0, %vec1 : tensor<4xf32>, tensor<4xf32>
    } {ssbuffer.core_type = "VECTOR, VECTOR"}

    %1:3 = scf.for %i = %lb to %ub step %c1v iter_args(%lhs = %0#0, %rhs = %0#1, %cube = %cube_init) -> (tensor<4xf32>, tensor<4xf32>, i32) : i32 {
      %2 = arith.addi %cube, %c1c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, VECTOR, CUBE"} %lhs, %rhs, %2 : tensor<4xf32>, tensor<4xf32>, i32
    } {ssbuffer.core_type = "VECTOR, VECTOR, CUBE"}

    %2 = tensor.extract %1#0[%idxv] {ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    memref.store %2, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xf32>
    memref.store %1#2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }

  // A while induction variable can belong to VECTOR while still controlling
  // the trip count of CUBE work. Both separated scopes must preserve the
  // predicate-carried value and its update.

  // CHECK-LABEL: func.func @while_predicate_controls_mixed_scope(
  // CHECK-SAME: %[[UB:.*]]: i32
  // CHECK: scope.scope : () -> () {
  // CHECK: scf.while (%[[V_IV:.*]] = %{{.*}}) : (i32) -> i32 {
  // CHECK: %[[V_COND:.*]] = arith.cmpi slt, %[[V_IV]], %[[UB]]
  // CHECK: scf.condition(%[[V_COND]]) %[[V_IV]] : i32
  // CHECK: ^bb0(%[[V_BODY_IV:.*]]: i32):
  // CHECK: %[[V_NEXT:.*]] = arith.addi %[[V_BODY_IV]], %{{.*}}
  // CHECK: scf.yield %[[V_NEXT]] : i32
  // CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  // CHECK: scope.scope : () -> () {
  // CHECK: scf.while (%[[C_IV:.*]] = %{{.*}}) : (i32) -> i32 {
  // CHECK: %[[C_COND:.*]] = arith.cmpi slt, %[[C_IV]], %[[UB]]
  // CHECK: scf.condition(%[[C_COND]]) %[[C_IV]] : i32
  // CHECK: ^bb0(%[[C_BODY_IV:.*]]: i32):
  // CHECK: memref.store
  // CHECK: %[[C_NEXT:.*]] = arith.addi %[[C_BODY_IV]], %{{.*}}
  // CHECK: scf.yield %[[C_NEXT]] : i32
  // CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
  func.func @while_predicate_controls_mixed_scope(
      %ub: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c0 = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : i32
    %c1 = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %cube_value = arith.constant {ssbuffer.core_type = "CUBE"} 7 : i32

    %result = scf.while (%iv = %c0) : (i32) -> i32 {
      %cond = arith.cmpi slt, %iv, %ub {ssbuffer.core_type = "VECTOR"} : i32
      scf.condition(%cond) {ssbuffer.core_type = "VECTOR"} %iv : i32
    } do {
    ^bb0(%iv: i32):
      memref.store %cube_value, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
      %next = arith.addi %iv, %c1 {ssbuffer.core_type = "VECTOR"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR"} %next : i32
    } attributes {ssbuffer.core_type = "VECTOR"}

    memref.store %result, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    func.return
  }

  // All loop result slots are VECTOR, but the scalar iter_arg is used by CUBE.
  // The if result updates that iter_arg, so its load must survive in CUBE too.
  // CUBE has no external result users: only the next iteration needs the carry.
  // CHECK-LABEL: func.func @for_scalar_carry_without_cube_sibling(
  // CHECK-SAME: %[[LB:[^:]+]]: index, %[[UB:[^:]+]]: index, %[[STEP:[^:]+]]: index, %[[COND:[^:]+]]: i1, %[[SRC:[^:]+]]: memref<?xi32>, %[[DST:[^:]+]]: memref<?xi32>
  // CHECK: hivm.tcore_type = #hivm.tcore_type<VECTOR>
  // CHECK: scope.scope
  // CHECK: scf.for %[[IV:[^ ]+]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ACC:[^ ]+]] = %{{[^)]+}})
  // CHECK: %[[NEXT:[^ ]+]] = scf.if %[[COND]] -> (i32) {
  // CHECK-NEXT: %[[LOAD:[^ ]+]] = memref.load %[[SRC]][%[[IV]]]
  // CHECK-NEXT: scf.yield{{.*}} %[[LOAD]] : i32
  // CHECK-NEXT: } else {
  // CHECK-NEXT: scf.yield{{.*}} %[[ACC]] : i32
  // CHECK: %[[SUM:[^ ]+]] = arith.addi %[[ACC]], %[[NEXT]]
  // CHECK: memref.store %[[ACC]], %[[DST]][%[[IV]]]
  // CHECK: scf.yield{{.*}} %[[SUM]] : i32
  // CHECK: hivm.tcore_type = #hivm.tcore_type<CUBE>
  func.func @for_scalar_carry_without_cube_sibling(
      %lb: index, %ub: index, %step: index, %cond: i1,
      %src: memref<?xi32>, %dst: memref<?xi32>, %outv: memref<1xi32>) {
    %idx = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %zero = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : i32
    %result = scf.for %i = %lb to %ub step %step
        iter_args(%acc = %zero) -> (i32) {
      %next = scf.if %cond -> (i32) {
        %value = memref.load %src[%i] {ssbuffer.core_type = "VECTOR"} : memref<?xi32>
        scf.yield %value : i32
      } else {
        scf.yield %acc : i32
      } {ssbuffer.core_type = "VECTOR"}
      %sum = arith.addi %acc, %next {ssbuffer.core_type = "VECTOR"} : i32
      memref.store %acc, %dst[%i] {ssbuffer.core_type = "CUBE"} : memref<?xi32>
      scf.yield %sum : i32
    } {ssbuffer.core_type = "VECTOR"}
    memref.store %result, %outv[%idx] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    return
  }

  // CHECK-LABEL: func.func @while_scalar_carry_without_cube_sibling(
  // CHECK-SAME: %[[COND:[^:]+]]: i1, %[[BRANCH:[^:]+]]: i1, %[[SRC:[^:]+]]: memref<1xindex>, %[[DST:[^:]+]]: memref<1xindex>
  // CHECK: hivm.tcore_type = #hivm.tcore_type<VECTOR>
  // CHECK: scope.scope
  // CHECK: scf.while (%[[BEFORE:[^ ]+]] = %{{[^)]+}}) : (index) -> index {
  // CHECK: scf.condition(%[[COND]]) %[[BEFORE]] : index
  // CHECK: ^bb0(%[[ACC:[^:]+]]: index):
  // CHECK: %[[NEXT:[^ ]+]] = scf.if %[[BRANCH]] -> (index) {
  // CHECK-NEXT: %[[LOAD:[^ ]+]] = memref.load %[[SRC]]
  // CHECK-NEXT: scf.yield{{.*}} %[[LOAD]] : index
  // CHECK-NEXT: } else {
  // CHECK-NEXT: scf.yield{{.*}} %[[ACC]] : index
  // CHECK: %[[SUM:[^ ]+]] = arith.addi %[[ACC]], %[[NEXT]]
  // CHECK: memref.store %[[ACC]], %[[DST]]
  // CHECK: scf.yield{{.*}} %[[SUM]] : index
  // CHECK: hivm.tcore_type = #hivm.tcore_type<CUBE>
  func.func @while_scalar_carry_without_cube_sibling(
      %cond: i1, %branch: i1, %src: memref<1xindex>, %dst: memref<1xindex>,
      %outv: memref<1xindex>) {
    %zero = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %result = scf.while (%acc = %zero) : (index) -> index {
      scf.condition(%cond) %acc : index
    } do {
    ^bb0(%acc: index):
      %next = scf.if %branch -> (index) {
        %value = memref.load %src[%zero] {ssbuffer.core_type = "VECTOR"} : memref<1xindex>
        scf.yield %value : index
      } else {
        scf.yield %acc : index
      } {ssbuffer.core_type = "VECTOR"}
      %sum = arith.addi %acc, %next {ssbuffer.core_type = "VECTOR"} : index
      memref.store %acc, %dst[%zero] {ssbuffer.core_type = "CUBE"} : memref<1xindex>
      scf.yield %sum : index
    } attributes {ssbuffer.core_type = "VECTOR"}
    memref.store %result, %outv[%zero] {ssbuffer.core_type = "VECTOR"} : memref<1xindex>
    return
  }
}
