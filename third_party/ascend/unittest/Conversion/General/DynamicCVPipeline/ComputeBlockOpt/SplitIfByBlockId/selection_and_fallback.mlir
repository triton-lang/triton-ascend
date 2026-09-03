// RUN: triton-opt --split-input-file --split-if-by-block-id %s | FileCheck %s

// VCV: then side has VECTOR -> CUBE(matmul) -> VECTOR, should split.
// CHECK-LABEL: func.func @split_then_vcv
// CHECK: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// CHECK: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: scf.if
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 96
func.func @split_then_vcv(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: f32, %d: f32, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : f32
    scf.if %cond {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : f32
      %m = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v2 = arith.mulf %c, %d {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : f32
    }
  }
  return
}

// -----

// CVC: else side has CUBE(matmul) -> VECTOR -> CUBE(matmul), should split.
// Split ifs are re-ordered by memory dependencies (matmuls alias on %a),
// final order is 97 -> 96 -> 95.
// CHECK-LABEL: func.func @split_else_cvc
// CHECK: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 97
// CHECK: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 96
// CHECK: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
func.func @split_else_cvc(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: f32, %d: f32, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : f32
    scf.if %cond {
      %x = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : f32
    } else {
      %m1 = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : f32
      %m2 = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 97 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
    }
  }
  return
}

// -----

// Non-CVC/VCV: else side has only 2 groups (CUBE+VECTOR), not alternating 3 groups, should NOT split.
// CHECK-LABEL: func.func @skip_two_groups
// CHECK-COUNT-1: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
func.func @skip_two_groups(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: f32, %d: f32, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : f32
    scf.if %cond {
      %x = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : f32
    } else {
      %y = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
    }
  }
  return
}
