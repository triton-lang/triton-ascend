// RUN: triton-opt --clone-ops %s --allow-unregistered-dialect | FileCheck %s

// A value used by the main-loop yield must still be remapped at consumer uses.
// Its sibling results and nested yields must use the same block-local clone.
// CHECK-LABEL: func.func @clone_loop_yield_uses
// CHECK: %[[PRODUCER:.*]]:3 = scf.if
// CHECK: %[[CLONE:.*]]:3 = scf.if
// CHECK: %[[COND:.*]] = arith.cmpi eq, %[[CLONE]]#2, %c2_i32 {ssbuffer.block_id = 2 : i32} : i32
// CHECK: %[[CONSUMER:.*]] = scf.if %[[COND]] -> (i32)
// CHECK: scf.yield %[[CLONE]]#0 : i32
// CHECK: scf.yield %[[CLONE]]#0 : i32
// CHECK: scf.yield %[[PRODUCER]]#0 : i32
module {
  func.func @clone_loop_yield_uses() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %result = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%ki = %c0_i32) -> (i32) {
      %producer_cond = arith.cmpi eq, %ki, %c0_i32 {ssbuffer.block_id = 1 : i32} : i32
      %producer:3 = scf.if %producer_cond -> (i32, i32, i32) {
        %p0 = arith.addi %ki, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
        %p1 = arith.addi %p0, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
        %p2 = arith.addi %p1, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
        scf.yield %p0, %p1, %p2 : i32, i32, i32
      } else {
        scf.yield %ki, %ki, %ki : i32, i32, i32
      } {ssbuffer.block_id = 1 : i32}
      %consumer_cond = arith.cmpi eq, %producer#2, %c2_i32 {ssbuffer.block_id = 2 : i32} : i32
      %consumer = scf.if %consumer_cond -> (i32) {
        scf.yield %producer#0 : i32
      } else {
        scf.yield %producer#0 : i32
      } {ssbuffer.block_id = 2 : i32}
      scf.yield %producer#0 : i32
    } {ssbuffer.main_loop = 0 : i32}
    return
  }
}
