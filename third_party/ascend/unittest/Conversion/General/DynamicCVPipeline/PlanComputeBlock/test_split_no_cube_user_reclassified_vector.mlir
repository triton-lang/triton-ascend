// RUN: triton-opt --op-classifier %s | FileCheck %s

// Regression test for `OpClassifierPass` Step 7 Phase 0 (PD-934):
//
//   A pure dataflow op colored CUBE by a seed pattern while every one of its
//   users is VECTOR used to be split unconditionally. The split redirected
//   all users to the VECTOR clone and left a dead CUBE original, which
//   downstream passes (CloneCubeDepInIf / DataDependencyAnalysis) could not
//   tell from a real CUBE block — it materialized duplicate cross-core
//   transfer/sync chains and deadlocked the K>16 cross-core handshake.
//
//   Phase 0 now censuses the users first: no OP_CUBE_ONLY user means the
//   op is pure VECTOR dataflow — reclassify in place, never split.
//
// Seed path exercised here: matchExtractSlicePattern marks
// tensor.extract_slice CUBE because it is the sole user of a linalg.matmul
// result (which requires the matmul result to have exactly one use), while
// the vector upstream BFS (Step 4) ORs VECTOR in through the VECTOR addf
// consumer -> CUBE_AND_VECTOR.

module {

  // The CUBE-seeded extract_slice has only a VECTOR user: it must NOT be
  // split. It is reclassified VECTOR in place, keeps its single user and
  // leaves no dead CUBE copy for the block planners.
  // CHECK-LABEL: func.func @cube_seed_dataflow_no_cube_user_reclassified
  // CHECK: %[[EMPTY:.*]] = tensor.empty() {ssbuffer.core_type = "CUBE"}
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.core_type = "CUBE"} ins({{.*}}, {{.*}} : tensor<16x16xf32>, tensor<16x16xf32>) outs(%[[EMPTY]]
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[MM]][0, 0] [16, 16] [1, 1] {ssbuffer.core_type = "VECTOR"}
  // No dead CUBE copy may follow (pre-fix the split left a user-less CUBE
  // original right here, between the clone and its consumer).
  // CHECK-NOT: tensor.extract_slice
  // CHECK: %[[ADD:.*]] = arith.addf %[[SLICE]], %{{.*}} {ssbuffer.core_type = "VECTOR"}
  // CHECK: return %[[ADD]]
  // Negative guard: no CUBE_AND_VECTOR residue (pre-fix the op was stamped
  // with the latter before splitting).
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: core_type = "CUBE_AND_VECTOR"
  func.func @cube_seed_dataflow_no_cube_user_reclassified(
      %a: tensor<16x16xf32>, %b: tensor<16x16xf32>,
      %other: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %init = tensor.empty() : tensor<16x16xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    %slice = tensor.extract_slice %mm[0, 0] [16, 16] [1, 1] : tensor<16x16xf32> to tensor<16x16xf32>
    %add = arith.addf %slice, %other : tensor<16x16xf32>
    return %add : tensor<16x16xf32>
  }

  // Control: the census must not suppress legitimate splits. This is the
  // canonical shared-producer pattern documented on the splitter: one
  // producer (tensor.empty, CUBE-seeded by matchEmptyPattern through the
  // matmul `outs` chain, VECTOR-colored by the Step 4 BFS through the addf)
  // feeds both a CUBE consumer (linalg.matmul reading it as `outs`) and a
  // VECTOR consumer (arith.addf). The census finds the CUBE user, so the
  // split must still fire: CUBE original keeps the matmul, VECTOR clone
  // takes the addf. (The extract_slice seed above cannot serve here: the
  // one-use chain gate that drives the seeding already forces its sole
  // user to be the VECTOR one — which is exactly why the dead-original
  // bug only reproduced through that path.)
  // CHECK-LABEL: func.func @shared_producer_with_cube_user_still_split
  // The VECTOR clone is inserted before the original: the addf reads the
  // clone, the matmul keeps the CUBE original as `outs`.
  // CHECK: %[[EMPTY_V:.*]] = tensor.empty() {ssbuffer.core_type = "VECTOR"}
  // CHECK: %[[EMPTY_C:.*]] = tensor.empty() {ssbuffer.core_type = "CUBE"}
  // CHECK: %[[MM:.*]] = linalg.matmul {ssbuffer.core_type = "CUBE"} ins({{.*}}, {{.*}} : tensor<16x16xf32>, tensor<16x16xf32>) outs(%[[EMPTY_C]]
  // CHECK: %[[ADD:.*]] = arith.addf %[[EMPTY_V]], %{{.*}} {ssbuffer.core_type = "VECTOR"}
  // CHECK: return %[[ADD]], %[[MM]]
  func.func @shared_producer_with_cube_user_still_split(
      %a: tensor<16x16xf32>, %b: tensor<16x16xf32>,
      %other: tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
    %init = tensor.empty() : tensor<16x16xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    %add = arith.addf %init, %other : tensor<16x16xf32>
    return %add, %mm : tensor<16x16xf32>, tensor<16x16xf32>
  }
}
