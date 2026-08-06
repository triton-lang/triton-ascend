// RUN: triton-opt --merge-same-source-axis %s | FileCheck %s

// Guards: convergence op's same-block VECTOR_ONLY direct downstream must
// be pulled along with the merge chain, otherwise the moved convergence
// op would leave its tail in the old block and create a cross-block
// dependency inconsistency.

// CHECK-LABEL: func.func @downstream_follows_convergence
func.func @downstream_follows_convergence(%arg0: f32, %arg1: f32) {
  // Source at block 28.
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 28
  %src = arith.addf %arg0, %arg1 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : f32

  // Two consumers in block 29 doing different "axis" processing.
  // CHECK: arith.mulf {{.*}}ssbuffer.block_id = 28
  %a = arith.mulf %src, %arg1 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : f32
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 28
  %b = arith.addf %src, %arg1 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : f32

  // Convergence op at block 29 — moves to 28.
  // CHECK: arith.subf {{.*}}ssbuffer.block_id = 28
  %k = arith.subf %a, %b {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : f32

  // Direct downstream VECTOR_ONLY user at the same block_id (29) as %k.
  // It must follow %k to block 28 — otherwise %k ends up in 28 while
  // %down stays in 29, splitting the chain.
  // CHECK: arith.mulf {{.*}}ssbuffer.block_id = 28
  %down = arith.mulf %k, %arg0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : f32

  return
}

// Downstream that is at a DIFFERENT block_id from the convergence op is
// not touched — its placement is independent and may have been decided by
// other passes.
func.func @downstream_at_different_block_left_alone(%arg0: f32, %arg1: f32) {
  %src = arith.addf %arg0, %arg1 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : f32
  %a = arith.mulf %src, %arg1 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : f32
  %b = arith.addf %src, %arg1 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : f32
  // CHECK: arith.subf {{.*}}ssbuffer.block_id = 28
  %k = arith.subf %a, %b {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : f32

  // Downstream at a different block (50) — must stay put.
  // CHECK: arith.mulf {{.*}}ssbuffer.block_id = 50
  %down = arith.mulf %k, %arg0 {ssbuffer.block_id = 50 : i32, ssbuffer.core_type = "VECTOR"} : f32

  return
}
