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
}
