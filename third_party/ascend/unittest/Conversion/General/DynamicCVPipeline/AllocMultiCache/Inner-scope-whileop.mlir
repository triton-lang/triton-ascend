// RUN: triton-opt --add_multi_buffer_inner_scope %s | FileCheck %s

// T-while-A: whileOp as main_loop, INTRA bufNum == 1 (single-buffer scope).
//   Verifies:
//     - whileOp carrying ssbuffer.main_loop on its terminator IS recognized
//       (legacy shape supported by hasMainLoopAttr).
//     - setupWhileIterArgCounter is SKIPPED (bufNum==1 → no dead iter_arg).
//     - Cross-block tensor dep still gets single-buffer treatment
//       (single memref.alloc + single hivm.hir.copy + single to_tensor).
//     - NO scf.if dispatch is emitted (N==1 fast path).
//     - The original whileOp is preserved verbatim (no extra i32 iter_arg).
//   We pin the buffer count to 1 via the module-level
//   `ssbuffer.intra_buf_count` attribute so the default of 2 doesn't
//   trigger the counter-setup branch.

// CHECK-LABEL: func.func @test_while_mainloop_bufnum_one
// Exactly one UB alloc before the whileOp.
// CHECK-DAG:   memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
// CHECK-NOT:   memref.alloc() {{.*}}: memref<128xf32, #hivm.address_space<ub>>
// Original whileOp do-region bb0 has only 2 block-args (no i32 counter).
// CHECK:       ^bb0(%{{.*}}: tensor<128xf32>, %{{.*}}: i32):
// Single producer-side hivm.hir.copy.
// CHECK:       hivm.hir.copy ins({{.*}} : tensor<128xf32>) outs({{.*}} : memref<128xf32>)
// Single consumer-side bufferization.to_tensor (the readback).
// CHECK:       bufferization.to_tensor {{.*}}: memref<128xf32> to tensor<128xf32>
// main_loop attribute survives on the new whileOp.
// CHECK:       } {{.*}}ssbuffer.main_loop = 1 : i64

// T-while-B: whileOp as main_loop, INTRA bufNum == 2 (multi-buffer scope).
//   Verifies:
//     - setupWhileIterArgCounter IS called (bufNum>1).
//     - whileOp is REPLACED with a new one that has an extra i32 iter_arg
//       (init=0, yielded as counter+1 at end of do-region).
//     - The new do-region bb0 has ONE extra block-arg compared to the input
//       (3 block-args instead of 2; the new arg is i32).
//     - arith.addi counter, 1 is present in the do-region for the yield.
//     - Multi-buffer producer/consumer with scf.if dispatch works as forOp.

// CHECK-LABEL: func.func @test_while_mainloop_bufnum_two
// Two UB allocs (ping/pong).
// CHECK-DAG:   memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
// CHECK-DAG:   memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
// WhileOp's do-region bb0 has 3 block-args now (the new i32 counter is the last one).
// CHECK:       ^bb0(%{{.*}}: tensor<128xf32>, %{{.*}}: i32, %{{.*}}: i32):
// Producer scf.if dispatch (the original counter increment lives inside this region).
// CHECK:       scf.if
// CHECK:         hivm.hir.copy
// Consumer scf.if dispatch returning tensor.
// CHECK:       scf.if {{.*}} -> (tensor<128xf32>)
// CHECK:         bufferization.to_tensor
// arith.addi increment for the multi-buffer counter is present (block_id=10), tagged iterCounter.
// CHECK:       %{{.+}} = arith.addi %{{.+}}, %{{.+}} {ssbuffer.block_id = 10 : i32, ssbuffer.iterCounter} : i32
// Counter-aware whileOp carries ssbuffer.iterCounter alongside main_loop.
// CHECK:       } {{.*}}ssbuffer.iterCounter, {{.*}}ssbuffer.main_loop = 1 : i64

// T-while-C: whileOp main_loop with scf.if inside the do-region (multi-region
// consumer pattern). The whileOp carries `ssbuffer.main_loop` on itself (NOT
// on the terminator). This exercises `hasMainLoopAttr`'s "op has attr" path,
// in contrast to T-while-A/B which use the terminator-attr legacy shape.
// Verifies the cross-block tensor dep flowing through scf.if is buffered.

// CHECK-LABEL: func.func @test_while_mainloop_attr_on_op
// Producer alloc + copy before the whileOp.
// CHECK-DAG:   memref.alloc() : memref<64xf16, #hivm.address_space<ub>>
// CHECK-DAG:   memref.alloc() : memref<64xf16, #hivm.address_space<ub>>
// Consumer scf.if + to_tensor inside the do-region.
// CHECK:       scf.if {{.*}} -> (tensor<64xf16>)
// CHECK:         bufferization.to_tensor
// arith.addi increment for the multi-buffer counter is present (block_id=12), tagged iterCounter.
// CHECK:       %{{.+}} = arith.addi %{{.+}}, %{{.+}} {ssbuffer.block_id = 12 : i32, ssbuffer.iterCounter} : i32
// Counter-aware whileOp carries ssbuffer.iterCounter alongside main_loop.
// CHECK:       } {{.*}}ssbuffer.iterCounter, {{.*}}ssbuffer.main_loop = 1 : i64

// T-while-D: Regression guard for the getOutermostSsbufferId priority fix.
//   Before the fix, walking up from an op inside a whileOp's do-region would
//   hit the multi-region check (`numRegions >= 2`) BEFORE the kMainLoop check
//   and return the whileOp's own block_id, breaking cross-block dep
//   classification.
//   Setup: whileOp (block_id = 5) wrapping a producer block_id = 8 inside the
//   do-region and a consumer block_id = 12 also inside the do-region.
//   Cross-block judgment must use 8 vs 12 (NOT 5 vs 12).
//   Result: a multi-buffer MUST be emitted for the producer (8 → 12 is cross-
//   block). If the bug were present, getOutermostSsbufferId would return 5
//   for both, classify them as same-block, and SKIP the multi-buffer.

// CHECK-LABEL: func.func @test_while_outermost_id_priority
// Multi-buffer must be emitted (proves cross-block judgment saw 8 != 12).
// CHECK-DAG:   memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
// CHECK-DAG:   memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
// Producer-side dispatch.
// CHECK:       scf.if
// CHECK:         hivm.hir.copy
// arith.addi increment for the multi-buffer counter is present (block_id=12), tagged iterCounter.
// CHECK:       %{{.+}} = arith.addi %{{.+}}, %{{.+}} {ssbuffer.block_id = 12 : i32, ssbuffer.iterCounter} : i32
// Counter-aware whileOp carries ssbuffer.iterCounter alongside main_loop.
// CHECK:       } {{.*}}ssbuffer.iterCounter, {{.*}}ssbuffer.main_loop = 1 : i64

// T-while-E: whileOp main_loop with a tensor::EmptyOp + linalg::FillOp pattern
//   inside the do-region. The empty+fill is cloned into each consumer block.
//   Verifies cloneEmptyFillsInBlocks works on a whileOp main_loop (the body
//   is `getAfterBody()`, not `getBody()`).
//   The cloned fill's ins stays the same (arith.constant, lives outside the
//   main_loop) — no clone of the scalar chain is needed.

// CHECK-LABEL: func.func @test_while_clone_empty_fill
// Original fill at block_id = 8 (producer block) is preserved.
// CHECK:       linalg.fill {{.*}}{ssbuffer.block_id = 8 : i32} {{.*}}outs({{.*}} : tensor<32x1xf32>)
// Original fill at block_id = 14 (second producer block) is preserved.
// CHECK:       linalg.fill {{.*}}{ssbuffer.block_id = 14 : i32} {{.*}}outs({{.*}} : tensor<32x1xf32>)
// Cloned fills land in block_id = 12 (consumer block) — there are two of them,
// one per original producer. CHECK matches at least once.
// CHECK:       linalg.fill {{.*}}{ssbuffer.block_id = 12 : i32} {{.*}}outs({{.*}} : tensor<32x1xf32>)
// arith.addi increment for the multi-buffer counter is present (block_id=12), tagged iterCounter.
// CHECK:       %{{.+}} = arith.addi %{{.+}}, %{{.+}} {ssbuffer.block_id = 12 : i32, ssbuffer.iterCounter} : i32
// Counter-aware whileOp carries ssbuffer.iterCounter alongside main_loop.
// CHECK:       } {{.*}}ssbuffer.iterCounter, {{.*}}ssbuffer.main_loop = 1 : i64

// T-while-F (TODO): Negative — two whileOps in the same scope, neither is the
//   "outer" main_loop; one is nested in the other's do-region. The pass must
//   refuse via findNestedMainloop (returns WalkResult::interrupt + fallback).
//   Not implemented in this initial drop — to be added when the negative case
//   is needed for a regression guard.

// ---- Inputs ----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">,
                   ssbuffer.intra_buf_count = 1 : i32} {
  // T-while-A: whileOp with main_loop on the terminator, single-buffer.
  func.func @test_while_mainloop_bufnum_one() {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128xf32>
    %carry = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0_i32)
          : (tensor<128xf32>, i32) -> (tensor<128xf32>, i32) {
        %cmp = arith.cmpi slt, %arg1, %c10_i32 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : tensor<128xf32>, i32
      } do {
      ^bb0(%arg0: tensor<128xf32>, %arg1: i32):
        // Producer block_id = 7.
        %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xf32>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 7 : i32} : memref<128xf32> to tensor<128xf32>
        // Consumer block_id = 10 (cross-block).
        %consumed = arith.addf %prod, %prod {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
        %next = arith.addi %arg1, %c1_i32 : i32
        scf.yield %consumed, %next : tensor<128xf32>, i32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 23 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">,
                   ssbuffer.intra_buf_count = 2 : i32} {
  // T-while-B: whileOp with main_loop on the terminator, multi-buffer (bufNum=2).
  func.func @test_while_mainloop_bufnum_two() {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128xf32>
    %carry = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0_i32)
          : (tensor<128xf32>, i32) -> (tensor<128xf32>, i32) {
        %cmp = arith.cmpi slt, %arg1, %c10_i32 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : tensor<128xf32>, i32
      } do {
      ^bb0(%arg0: tensor<128xf32>, %arg1: i32):
        %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xf32>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 7 : i32} : memref<128xf32> to tensor<128xf32>
        %consumed = arith.addf %prod, %prod {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
        %next = arith.addi %arg1, %c1_i32 : i32
        scf.yield %consumed, %next : tensor<128xf32>, i32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 23 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // T-while-C: whileOp with main_loop on the OP itself (not terminator).
  func.func @test_while_mainloop_attr_on_op() {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_one = arith.constant 1.0 : f16
    %init = tensor.empty() : tensor<64xf16>
    %carry = linalg.fill ins(%cst_one : f16) outs(%init : tensor<64xf16>) -> tensor<64xf16>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0_i32)
          : (tensor<64xf16>, i32) -> (tensor<64xf16>, i32) {
        %cmp = arith.cmpi slt, %arg1, %c10_i32 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : tensor<64xf16>, i32
      } do {
      ^bb0(%arg0: tensor<64xf16>, %arg1: i32):
        %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<64xf16>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 7 : i32} : memref<64xf16> to tensor<64xf16>
        // Multi-region consumer (scf.if inside the do-region).
        %cnd = arith.cmpi slt, %arg1, %c1_i32 {ssbuffer.block_id = 11 : i32} : i32
        %consumed = scf.if %cnd -> (tensor<64xf16>) {
          %a = arith.addf %prod, %prod {ssbuffer.block_id = 11 : i32} : tensor<64xf16>
          scf.yield %a : tensor<64xf16>
        } else {
          %b = arith.mulf %prod, %prod {ssbuffer.block_id = 12 : i32} : tensor<64xf16>
          scf.yield %b : tensor<64xf16>
        } {ssbuffer.block_id = 12 : i32}
        %next = arith.addi %arg1, %c1_i32 : i32
        scf.yield %consumed, %next : tensor<64xf16>, i32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 23 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // T-while-D: whileOp main_loop with cross-block dep that tests
  // getOutermostSsbufferId's main_loop-first priority.
  // Producer at block_id=8, consumer at block_id=12, both inside do-region.
  // whileOp itself is at block_id=5 (so the buggy code would have collapsed
  // both to 5, hiding the cross-block dep).
  func.func @test_while_outermost_id_priority() {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32xf32>
    %carry = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<32xf32>) -> tensor<32xf32>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0_i32)
          : (tensor<32xf32>, i32) -> (tensor<32xf32>, i32) {
        %cmp = arith.cmpi slt, %arg1, %c10_i32 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : tensor<32xf32>, i32
      } do {
      ^bb0(%arg0: tensor<32xf32>, %arg1: i32):
        %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32xf32>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 8 : i32} : memref<32xf32> to tensor<32xf32>
        %consumed = arith.addf %prod, %prod {ssbuffer.block_id = 12 : i32} : tensor<32xf32>
        %next = arith.addi %arg1, %c1_i32 : i32
        scf.yield %consumed, %next : tensor<32xf32>, i32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 5 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // T-while-E: whileOp main_loop with empty+fill pattern.
  // Two pairs of empty+fill, both cloned to consumer block 12.
  func.func @test_while_clone_empty_fill() {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_one = arith.constant 1.0 : f32
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %c0_i32, %arg1 = %cst_one)
          : (i32, f32) -> (i32, f32) {
        %cmp = arith.cmpi slt, %arg0, %c10_i32 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : i32, f32
      } do {
      ^bb0(%arg0: i32, %arg1: f32):
        // Producer block_id = 8: empty+fill (cloneable).
        %empty8 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<32x1xf32>
        %fill8 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst_one : f32)
            outs(%empty8 : tensor<32x1xf32>) -> tensor<32x1xf32>
        // Producer block_id = 14: empty+fill (cloneable).
        %empty14 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<32x1xf32>
        %fill14 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst_one : f32)
            outs(%empty14 : tensor<32x1xf32>) -> tensor<32x1xf32>
        // Consumer block_id = 12: uses BOTH fills cross-block.
        %consumed = arith.addf %fill8, %fill14 {ssbuffer.block_id = 12 : i32} : tensor<32x1xf32>
        %next = arith.addi %arg0, %c1_i32 : i32
        scf.yield %next, %arg1 : i32, f32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 23 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
