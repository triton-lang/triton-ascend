// RUN: triton-opt --split-input-file --merge-cube-block %s | FileCheck %s --check-prefix=MERGE
// RUN: triton-opt --split-input-file --merge-cube-block --reorder-ops-by-block-id %s | FileCheck %s --check-prefix=REORDER

// A later loop without a merge must not clear an earlier loop's merge state.
// Merging blocks 2 and 3 leaves block 2 split by independent block 4, so the
// following reorder must run and make the merged block contiguous.
// MERGE-LABEL: module attributes
// MERGE-NOT: ssbuffer.merge_compute_block_applied
// MERGE: func.func @merge_then_no_merge
// MERGE: linalg.matmul {ssbuffer.block_id = 2 : i32
// MERGE: linalg.matmul {ssbuffer.block_id = 4 : i32
// MERGE: linalg.matmul {ssbuffer.block_id = 2 : i32
// REORDER-LABEL: module attributes
// REORDER-NOT: ssbuffer.merge_compute_block_applied
// REORDER: func.func @merge_then_no_merge
// REORDER: linalg.matmul {ssbuffer.block_id = 2 : i32
// REORDER-NEXT: %{{.*}} = linalg.matmul {ssbuffer.block_id = 2 : i32
// REORDER-NOT: ssbuffer.block_id = 2 : i32
// REORDER: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @merge_then_no_merge(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %init: tensor<16x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %v = arith.addf %a, %b {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
      %first = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%v, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %independent = linalg.matmul {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %second = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%v, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %out = arith.addf %first, %second {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
    }
    scf.for %j = %c0 to %c16 step %c1 {
      %v = arith.addf %a, %b {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
      %only = linalg.matmul {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} ins(%v, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %out = arith.addf %only, %a {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
    }
    return
  }
}

// -----

// Reversing the loop order must still request reorder after a merge.
// MERGE-LABEL: module attributes
// MERGE-NOT: ssbuffer.merge_compute_block_applied
// MERGE: func.func @no_merge_then_merge
// MERGE: linalg.matmul {ssbuffer.block_id = 2 : i32
// MERGE: linalg.matmul {ssbuffer.block_id = 4 : i32
// MERGE: linalg.matmul {ssbuffer.block_id = 2 : i32
// REORDER-LABEL: module attributes
// REORDER-NOT: ssbuffer.merge_compute_block_applied
// REORDER: func.func @no_merge_then_merge
// REORDER: linalg.matmul {ssbuffer.block_id = 2 : i32
// REORDER-NEXT: %{{.*}} = linalg.matmul {ssbuffer.block_id = 2 : i32
// REORDER-NOT: ssbuffer.block_id = 2 : i32
// REORDER: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @no_merge_then_merge(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %init: tensor<16x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %j = %c0 to %c16 step %c1 {
      %v = arith.addf %a, %b {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
      %only = linalg.matmul {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} ins(%v, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %out = arith.addf %only, %a {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
    }
    scf.for %i = %c0 to %c16 step %c1 {
      %v = arith.addf %a, %b {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
      %first = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%v, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %independent = linalg.matmul {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %second = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%v, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %out = arith.addf %first, %second {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
    }
    return
  }
}

// -----

// When no loop merges, retain the false marker and let reorder consume it.
// MERGE-LABEL: module attributes
// MERGE-SAME: ssbuffer.merge_compute_block_applied = false
// MERGE: func.func @no_loop_merges
// REORDER-LABEL: module attributes
// REORDER-NOT: ssbuffer.merge_compute_block_applied
// REORDER: func.func @no_loop_merges
// REORDER: scf.for
// REORDER-NEXT: %{{.*}} = arith.addf {{.*}}ssbuffer.block_id = 1 : i32
// REORDER-NEXT: %{{.*}} = linalg.matmul {ssbuffer.block_id = 2 : i32
// REORDER-NEXT: %{{.*}} = arith.addf {{.*}}ssbuffer.block_id = 3 : i32
// REORDER: scf.for
// REORDER-NEXT: %{{.*}} = arith.addf {{.*}}ssbuffer.block_id = 11 : i32
// REORDER-NEXT: %{{.*}} = linalg.matmul {ssbuffer.block_id = 12 : i32
// REORDER-NEXT: %{{.*}} = arith.addf {{.*}}ssbuffer.block_id = 13 : i32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @no_loop_merges(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %init: tensor<16x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %v = arith.addf %a, %b {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
      %only = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%v, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %out = arith.addf %only, %a {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
    }
    scf.for %j = %c0 to %c16 step %c1 {
      %v = arith.addf %a, %b {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
      %only = linalg.matmul {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} ins(%v, %b : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %out = arith.addf %only, %a {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
    }
    return
  }
}
