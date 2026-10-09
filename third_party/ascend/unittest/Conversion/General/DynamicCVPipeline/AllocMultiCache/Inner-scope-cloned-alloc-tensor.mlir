// RUN: triton-opt --add_multi_buffer_inner_scope %s | FileCheck %s

// T-cloned-alloc-tensor: bufferization.alloc_tensor appearing as a cross-block
// dep must NOT be routed through the multi-buffer pipeline. Multi-buffering
// it would copy uninitialized memory into a ping/pong memref, and the
// consumer would read that uninitialized data (read-before-first-write).
// Instead, the pass clones the alloc into each cross-block consumer's block
// and rewires the consumer's uses to the cloned alloc's result.
//
// `tensor::EmptyOp` is intentionally NOT covered here — it is a shape
// placeholder that goes through the scalar-dep path (ssbuffer.dep_mark) so
// the SSA chain across loop iterations is preserved. See
// Inner-scope.mlir::test_t19_tensor_empty_dep_mark and
// test_t20_mixed_dependencies for the existing EmptyOp behavior.
//
// Case A (test_alloc_tensor_cross_block_clone):
//   %alloc = bufferization.alloc_tensor() at block_id = 9 is consumed by a
//   linalg.fill at block_id = 11 (different block). The pass must:
//     1) clone the alloc into block_id = 11 (before the consumer)
//     2) rewire the consumer to use the cloned alloc
//     3) NOT insert any multi-buffer (no remsi / scf.if / hivm.hir.copy
//        that targets the alloc's value)
//
// Case B (test_alloc_tensor_same_block_no_clone):
//   %alloc at block_id = 9 is used in the SAME block (block_id = 9). No
//   cross-block dep exists, so the pass must not clone or multi-buffer.
//   Guards against the clone being over-eagerly applied to same-block uses.
//
// Case C (test_alloc_tensor_cloned_to_multiple_consumers):
//   One producer-side alloc_tensor (block_id = 9) feeds two distinct
//   cross-block consumers (block_id = 11 and block_id = 13). Each consumer
//   must get its own fresh clone — and each clone must be inserted BEFORE
//   the consumer in its own block. Mirrors the real-world case where a
//   loop-invariant alloc is read in multiple downstream blocks.

// CHECK-LABEL: func.func @test_alloc_tensor_cross_block_clone
// Producer-side alloc stays in block_id = 9 (untouched).
// CHECK: %[[ORIG_ALLOC:.*]] = bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32} : tensor<f32>
// Cloned alloc lands in the consumer's block (block_id = 11), BEFORE the
// linalg.fill that consumes it.
// CHECK: %[[CLONE_ALLOC:.*]] = bufferization.alloc_tensor() {ssbuffer.block_id = 11 : i32} : tensor<f32>
// CHECK: linalg.fill {ssbuffer.block_id = 11 : i32} ins({{.*}}) outs(%[[CLONE_ALLOC]] : tensor<f32>)
// The buggy pattern — arith.remsi / scf.if / hivm.hir.copy selecting ping/pong
// memrefs for an uninitialized alloc_tensor — must NOT appear anywhere in
// this function. We assert with a CHECK-NOT scoped to the function body.
// CHECK-NOT: arith.remsi
// CHECK-NOT: arith.cmpi
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.copy

// CHECK-LABEL: func.func @test_alloc_tensor_same_block_no_clone
// Alloc and consumer are in the same block (block_id = 9). No cross-block
// dep exists, so the pass must not clone or multi-buffer.
// CHECK: %[[SAME_ALLOC:.*]] = bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32} : tensor<f32>
// CHECK: arith.addf %[[SAME_ALLOC]]
// CHECK-NOT: bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32}
// CHECK-NOT: arith.remsi
// CHECK-NOT: scf.if

// CHECK-LABEL: func.func @test_alloc_tensor_cloned_to_multiple_consumers
// Producer-side alloc (block_id = 9) is read by two distinct consumers:
// one at block_id = 11 (linalg.fill) and one at block_id = 13 (arith.addf).
// Each consumer block must get its own fresh clone.
// CHECK: %[[ORIG_MULTI:.*]] = bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32} : tensor<f32>
// Clone into block_id = 11.
// CHECK: %[[CLONE_AT_11:.*]] = bufferization.alloc_tensor() {ssbuffer.block_id = 11 : i32} : tensor<f32>
// CHECK: linalg.fill {ssbuffer.block_id = 11 : i32} ins({{.*}}) outs(%[[CLONE_AT_11]] : tensor<f32>)
// Clone into block_id = 13 (the other consumer).
// CHECK: %[[CLONE_AT_13:.*]] = bufferization.alloc_tensor() {ssbuffer.block_id = 13 : i32} : tensor<f32>
// CHECK: arith.addf %[[CLONE_AT_13]]
// CHECK-NOT: arith.remsi
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.copy

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Case A: bufferization.alloc_tensor at block_id=9, consumed at block_id=11.
  // Without the fix, the pass would insert a remsi + scf.if + hivm.hir.copy
  // chain that copies the uninitialized tensor into a ping/pong memref and
  // then to_tensor's it back — i.e. read-before-first-write on the consumer.
  func.func @test_alloc_tensor_cross_block_clone() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %cst = arith.constant 0.0 : f32
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
        %alloc = bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32} : tensor<f32>
        %alloc_2 = bufferization.alloc_tensor() {ssbuffer.block_id = 10 : i32} : tensor<f32>
        scf.for %j = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
          %fill = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst : f32) outs(%alloc : tensor<f32>) -> tensor<f32>
        } {Undefined, ssbuffer.block_id = 22 : i32}
      } {Undefined, ssbuffer.block_id = 23 : i32, ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // Case B: alloc and consumer share block_id=9 → no cross-block dep, so the
  // pass must not clone and must not multi-buffer. This guards against the
  // clone being over-eagerly applied to same-block uses.
  func.func @test_alloc_tensor_same_block_no_clone() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
        %alloc = bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32} : tensor<f32>
        %carry = bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32} : tensor<f32>
        %filled = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst : f32) outs(%carry : tensor<f32>) -> tensor<f32>
        %add = arith.addf %alloc, %filled {ssbuffer.block_id = 9 : i32} : tensor<f32>
      } {Undefined, ssbuffer.block_id = 23 : i32, ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // Case C: one producer-side alloc (block_id=9) feeds two distinct consumers
  // (block_id=11 and block_id=13). Each consumer must get its own clone.
  func.func @test_alloc_tensor_cloned_to_multiple_consumers() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %cst = arith.constant 0.0 : f32
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
        %alloc = bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32} : tensor<f32>
        scf.for %j = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
          %fill = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst : f32) outs(%alloc : tensor<f32>) -> tensor<f32>
        } {Undefined, ssbuffer.block_id = 22 : i32}
        %add = arith.addf %alloc, %alloc {ssbuffer.block_id = 13 : i32} : tensor<f32>
      } {Undefined, ssbuffer.block_id = 23 : i32, ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
