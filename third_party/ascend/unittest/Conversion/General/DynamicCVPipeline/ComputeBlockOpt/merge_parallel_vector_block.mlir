// RUN: triton-opt --merge-parallel-vector-block %s | FileCheck %s

// The loop contains four compute blocks:
//   block 21 (VECTOR): independent (all operands are loop-external values) and
//                      its only compute-block user is block 11  -> A
//   block 4  (CUBE)  : independent                                -> C
//   block 11 (VECTOR): all ops elementwise, depends only on A and C -> B
//   block 6  (CUBE)  : consumes B
// Exactly one (A, B) pair exists, so B's block_id 11 must be rewritten to A's
// block_id 21.

// CHECK-LABEL: func.func @merge_same_stage_vector
func.func @merge_same_stage_vector(
    %x: tensor<32x32xf32>, %y: tensor<32x32xf32>,
    %lhs: tensor<32x128xbf16>, %rhs: tensor<128x32xbf16>,
    %rhs2: tensor<32x32xbf16>, %init: tensor<32x32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    // A: independent VECTOR block 21.
    // CHECK: arith.addf %{{.*}}, %{{.*}} {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %a = arith.addf %x, %y {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    // C: independent CUBE block 4.
    // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"
    %cube = linalg.matmul {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%lhs, %rhs : tensor<32x128xbf16>, tensor<128x32xbf16>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    // B: elementwise VECTOR block 11 depending on {21, 4}; rewritten to 21.
    // CHECK: arith.mulf %{{.*}}, %{{.*}} {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %b0 = arith.mulf %cube, %a {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    // CHECK: math.exp %{{.*}} {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %b1 = math.exp %b0 {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xbf16>
    %b2 = arith.truncf %b1 {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> to tensor<32x32xbf16>
    // CUBE block consuming B stays at 6.
    // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"
    %c6 = linalg.matmul {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} ins(%b2, %rhs2 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  }
  return
}

// Block 11 must not survive anywhere.
// CHECK-NOT: ssbuffer.block_id = 11 : i32
