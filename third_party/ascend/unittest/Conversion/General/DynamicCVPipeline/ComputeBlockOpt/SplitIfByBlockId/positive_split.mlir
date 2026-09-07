// RUN: triton-opt --split-input-file --split-if-by-block-id %s | FileCheck %s

// VCV: then side has VECTOR -> CUBE -> VECTOR with cross-core data flow
// (%m consumes %v1, %v2 consumes %m), should split.
// CHECK-LABEL: func.func @split_then_vcv
// CHECK: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// CHECK: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: scf.if
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 96
func.func @split_then_vcv(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    scf.if %cond {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m = linalg.matmul ins(%v1, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v2 = arith.mulf %m, %b {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    }
  }
  return
}

// -----

// CVC: else side has CUBE -> VECTOR -> CUBE with cross-core data flow
// (%v1 consumes %m1, %m2 consumes %v1), should split. Split order follows
// the SSA data chain 95 -> 96 -> 97.
// CHECK-LABEL: func.func @split_else_cvc
// CHECK: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 96
// CHECK: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 97
func.func @split_else_cvc(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: f32, %d: f32, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : f32
    scf.if %cond {
      %x = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : f32
    } else {
      %m1 = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v1 = arith.addf %m1, %m1 {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m2 = linalg.matmul ins(%a, %v1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 97 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
    }
  }
  return
}

// -----

// ForOp with only CUBE ops inside: groupIsCube classifies it as CUBE via
// getCoreTypeOfSimpleOpOrCf body walk. VCV pattern should split.
// CHECK-LABEL: func.func @split_then_vcv_for_cube_only
// CHECK: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// CHECK: scf.if
// CHECK: scf.for
// CHECK: linalg.matmul {{.*}}CUBE
// CHECK: scf.if
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 96
func.func @split_then_vcv_for_cube_only(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    scf.if %cond {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %for_res = scf.for %i = %c0 to %c1 step %c1 iter_args(%arg = %v1) -> (tensor<2x2xf32>) {
        %m = linalg.matmul ins(%arg, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
        scf.yield %m : tensor<2x2xf32>
      } {ssbuffer.block_id = 95 : i32}
      %v2 = arith.mulf %for_res, %b {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    }
  }
  return
}