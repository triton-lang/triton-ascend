// RUN: triton-opt --process-args %s --allow-unregistered-dialect | FileCheck %s

// D072 regression: a producer block generates multiple K-state results, and
// a consumer block reads the result corresponding to its own tile.
// ProcessArgs must clone the complete producer chain and make both consumer
// conditions and writes use the cloned result instead of the producer's
// newly generated result for the current iteration.

// CHECK-LABEL: func.func @for_consumer_k_state
// CHECK: scf.for
// CHECK-SAME: iter_args(
// CHECK: %[[PRODUCER:.*]]:3 = scf.if
// CHECK: %[[CONSUMER_CLONE:.*]]:3 = scf.if
// CHECK: %[[CLONE_ARG0:.*]] = arith.addi %{{.*}}, %c1_i32 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 1 : i32} : i32
// CHECK: %[[CLONE_ARG1:.*]] = arith.addi %[[CLONE_ARG0]], %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
// CHECK: %[[CLONE_ARG2:.*]] = arith.addi %[[CLONE_ARG1]], %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
// CHECK: scf.yield %[[CLONE_ARG0]], %[[CLONE_ARG1]], %[[CLONE_ARG2]]
// CHECK: %[[CONSUMER_COND:.*]] = arith.cmpi eq, %[[CONSUMER_CLONE]]#2, %c2_i32 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 2 : i32} : i32
// CHECK: %[[CONSUMER_COND0:.*]] = arith.cmpi eq, %[[CONSUMER_CLONE]]#0, %c2_i32 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 2 : i32} : i32
// CHECK: %[[CONSUMER_IF:.*]] = scf.if %[[CONSUMER_COND]] -> (i32)
// CHECK: %[[CONSUMER_IF0:.*]] = scf.if %[[CONSUMER_COND0]] -> (i32)
// CHECK: scf.yield %[[PRODUCER]]#1 : i32
// CHECK-NOT: arith.cmpi eq, %[[PRODUCER]]#2
// CHECK-NOT: arith.cmpi eq, %[[PRODUCER]]#0
// CHECK: scf.yield %[[PRODUCER]]#0, %[[CONSUMER_CLONE]]#0 : i32, i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @for_consumer_k_state() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true

    %result = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%ki = %c0_i32) -> (i32) {
      %producer_cond = arith.cmpi eq, %ki, %c0_i32 {ssbuffer.block_id = 1 : i32} : i32
      %producer_if:3 = scf.if %producer_cond -> (i32, i32, i32) {
        %p0 = arith.addi %ki, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
        %p1 = arith.addi %p0, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
        %p2 = arith.addi %p1, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
        scf.yield %p0, %p1, %p2 : i32, i32, i32
      } else {
        scf.yield %ki, %ki, %ki : i32, i32, i32
      } {ssbuffer.block_id = 1 : i32, ssbuffer.if = 1 : i32}

      %consumer_cond = arith.cmpi eq, %producer_if#2, %c2_i32 {ssbuffer.block_id = 2 : i32} : i32
      %consumer_cond0 = arith.cmpi eq, %producer_if#0, %c2_i32 {ssbuffer.block_id = 2 : i32} : i32
      %consumer_if = scf.if %consumer_cond -> (i32) {
        scf.yield %producer_if#1 : i32
      } else {
        scf.yield %producer_if#1 : i32
      } {ssbuffer.block_id = 2 : i32, ssbuffer.if = 2 : i32}
      %consumer_if0 = scf.if %consumer_cond0 -> (i32) {
        scf.yield %producer_if#1 : i32
      } else {
        scf.yield %producer_if#1 : i32
      } {ssbuffer.block_id = 2 : i32, ssbuffer.if = 3 : i32}
      scf.yield %producer_if#0 : i32
    } {ssbuffer.main_loop = 0 : i32}

    return
  }
}
