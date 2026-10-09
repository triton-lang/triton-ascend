// RUN: triton-opt --sink-i1-producers-into-users %s | FileCheck %s

// ============================================================================
// SinkI1ProducersIntoUsersPass: sink i1-producing ops next to their i1 uses.
//
// Covers two fixes from commit 52b239ea:
//   1. Removal of HasRecursiveMemoryEffects early-reject in isPureAndRegionless.
//      Ops that carry the RecursiveMemoryEffects trait (e.g. scope.scope) are
//      now evaluated via MemoryEffectOpInterface.getEffects() instead of being
//      blanket-rejected. If the region body is pure, the op is treated as a
//      valid i1 producer and sunk like any other.
//   2. Missing seenBlockIds.insert(consumerBlockId) after moving the producer
//      to the first consumer's block. Without this insert, the subsequent
//      consumer loop would re-clone the producer in the same block, producing
//      a duplicate. With the fix, the already-moved producer is reused.
//
// Covered scenarios:
//   1. @sink_scope_i1_producer          – Fix 1: scope.scope (RecursiveMemory-
//                                          Effects) with pure body is sunk.
//   2. @sink_i1_same_block_no_clone     – Fix 2: two consumers in the same
//                                          block → one move, zero clones.
//   3. @sink_scope_i1_same_block_no_clone – Fix 1 + Fix 2 combined.
//   4. @sink_i1_diff_block_clones       – Regression: consumers in different
//                                          blocks still get separate clones.
// ============================================================================

module {
  // ---- Fix 1: scope.scope with RecursiveMemoryEffects is sunk ------------
  // scope.scope has the RecursiveMemoryEffects trait. Before the fix the
  // HasRecursiveMemoryEffects guard in isPureAndRegionless rejected it
  // outright. After the fix getEffects() is called; since the body only
  // contains a pure arith.cmpi, effects are empty and the op is sunk from
  // block 110 to block 120 (the consumer's block).
  // CHECK-LABEL: func.func @sink_scope_i1_producer
  func.func @sink_scope_i1_producer(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>, %arg2: tensor<8xi1>) -> tensor<8xi1> {
    // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<8xi1> {
    %mask = scope.scope : () -> tensor<8xi1> {
      // CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : tensor<8xi32>
      %cmp = arith.cmpi slt, %arg0, %arg1 : tensor<8xi32>
      // CHECK: scope.return %{{.*}} : tensor<8xi1>
      scope.return %cmp : tensor<8xi1>
    // CHECK: } {ssbuffer.block_id = 120
    } {ssbuffer.block_id = 110 : i32, ssbuffer.core_type = "VECTOR"}
    // CHECK: arith.andi %[[SCOPE]], %{{.*}} {ssbuffer.block_id = 120
    %result = arith.andi %mask, %arg2 {ssbuffer.block_id = 120 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
    return %result : tensor<8xi1>
  }

  // ---- Fix 2: two consumers in the same block → no duplicate clone -------
  // Producer arith.cmpi is in block 210; both arith.andi consumers are in
  // block 220. After moving the producer to block 220 the fix inserts
  // consumerBlockId into seenBlockIds, so the second consumer reuses the
  // already-moved producer instead of cloning a duplicate.
  // CHECK-LABEL: func.func @sink_i1_same_block_no_clone
  func.func @sink_i1_same_block_no_clone(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>, %arg2: tensor<8xi1>, %arg3: tensor<8xi1>) -> (tensor<8xi1>, tensor<8xi1>) {
    // CHECK: %[[PROD:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} {ssbuffer.block_id = 220
    %mask = arith.cmpi slt, %arg0, %arg1 {ssbuffer.block_id = 210 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi32>
    // CHECK: arith.andi %[[PROD]], %{{.*}} {ssbuffer.block_id = 220
    %c1 = arith.andi %mask, %arg2 {ssbuffer.block_id = 220 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
    // CHECK: arith.andi %[[PROD]], %{{.*}} {ssbuffer.block_id = 220
    %c2 = arith.andi %mask, %arg3 {ssbuffer.block_id = 220 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
    // CHECK-NOT: arith.cmpi
    return %c1, %c2 : tensor<8xi1>, tensor<8xi1>
  }

  // ---- Fix 1 + Fix 2 combined: scope.scope with two same-block consumers --
  // The scope.scope producer (RecursiveMemoryEffects) is sunk to block 320
  // (Fix 1) and, because both consumers share block 320, no duplicate clone
  // is created (Fix 2).
  // CHECK-LABEL: func.func @sink_scope_i1_same_block_no_clone
  func.func @sink_scope_i1_same_block_no_clone(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>, %arg2: tensor<8xi1>, %arg3: tensor<8xi1>) -> (tensor<8xi1>, tensor<8xi1>) {
    // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<8xi1> {
    %mask = scope.scope : () -> tensor<8xi1> {
      %cmp = arith.cmpi slt, %arg0, %arg1 : tensor<8xi32>
      scope.return %cmp : tensor<8xi1>
    // CHECK: } {ssbuffer.block_id = 320
    } {ssbuffer.block_id = 310 : i32, ssbuffer.core_type = "VECTOR"}
    // CHECK: arith.andi %[[SCOPE]], %{{.*}} {ssbuffer.block_id = 320
    %c1 = arith.andi %mask, %arg2 {ssbuffer.block_id = 320 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
    // CHECK: arith.andi %[[SCOPE]], %{{.*}} {ssbuffer.block_id = 320
    %c2 = arith.andi %mask, %arg3 {ssbuffer.block_id = 320 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
    // CHECK-NOT: scope.scope
    return %c1, %c2 : tensor<8xi1>, tensor<8xi1>
  }

  // ---- Regression: consumers in different blocks still get clones ---------
  // Producer in block 410 has consumers in block 420 and block 430. The pass
  // should clone the producer for the second distinct block (430), producing
  // two arith.cmpi ops.
  // CHECK-LABEL: func.func @sink_i1_diff_block_clones
  func.func @sink_i1_diff_block_clones(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>, %arg2: tensor<8xi1>, %arg3: tensor<8xi1>) -> (tensor<8xi1>, tensor<8xi1>) {
    // CHECK: %[[P1:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} {ssbuffer.block_id = 420
    %mask = arith.cmpi slt, %arg0, %arg1 {ssbuffer.block_id = 410 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi32>
    // CHECK: arith.andi %[[P1]], %{{.*}} {ssbuffer.block_id = 420
    %c1 = arith.andi %mask, %arg2 {ssbuffer.block_id = 420 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
    // CHECK: %[[P2:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} {ssbuffer.block_id = 430
    // CHECK: arith.andi %[[P2]], %{{.*}} {ssbuffer.block_id = 430
    %c2 = arith.andi %mask, %arg3 {ssbuffer.block_id = 430 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xi1>
    return %c1, %c2 : tensor<8xi1>, tensor<8xi1>
  }
}
