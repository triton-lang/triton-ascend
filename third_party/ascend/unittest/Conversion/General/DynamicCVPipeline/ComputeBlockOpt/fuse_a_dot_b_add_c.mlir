// RUN: triton-opt -fuse-adotbaddc %s | FileCheck %s

module {
// CHECK: module {
// Case 1: C is a block argument — canFuse returns false, no fusion
// CHECK: func.func @case1_no_defining_op
// CHECK: linalg.matmul
// CHECK: arith.addf
// CHECK: return
  func.func @case1_no_defining_op(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>, %C: tensor<32x32xf32>, %init_D: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %D = linalg.matmul {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%init_D : tensor<32x32xf32>) -> tensor<32x32xf32>
    %E = arith.addf %D, %C {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    return %E : tensor<32x32xf32>
  }

// Case 2: addf result is used by another matmul — canFuse returns false, no fusion
// CHECK: func.func @case2_addf_used_by_matmul
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: arith.addf
// CHECK: linalg.matmul
// CHECK: arith.addf
// CHECK: return
  func.func @case2_addf_used_by_matmul(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>, %B2: tensor<32x16xf32>, %init_D: tensor<32x32xf32>, %init_F: tensor<32x16xf32>) -> tensor<32x16xf32> {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f32
    %empty_C = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %C = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%empty_C : tensor<32x32xf32>) -> tensor<32x32xf32>
    %D = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%init_D : tensor<32x32xf32>) -> tensor<32x32xf32>
    %E = arith.addf %D, %C {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %F = linalg.matmul {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%E, %B2 : tensor<32x32xf32>, tensor<32x16xf32>) outs(%init_F : tensor<32x16xf32>) -> tensor<32x16xf32>
    %ret = arith.addf %F, %B2 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x16xf32>
    return %ret : tensor<32x16xf32>
  }


// Case 3: C is defined inside scf.if, does not dominate matmul — canFuse returns false, no fusion
// CHECK: func.func @case3_no_dominance
// CHECK: linalg.matmul
// CHECK: scf.if
// CHECK: arith.addf
// CHECK: return
  func.func @case3_no_dominance(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>, %cond: i1, %init_D: tensor<32x32xf32>, %fallback: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %D = linalg.matmul {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%init_D : tensor<32x32xf32>) -> tensor<32x32xf32>
    %C = scf.if %cond -> (tensor<32x32xf32>) {
      %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f32
      %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
      %filled = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %filled : tensor<32x32xf32>
    } else {
      scf.yield %fallback : tensor<32x32xf32>
    }
    %E = arith.addf %D, %C {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    return %E : tensor<32x32xf32>
  }

// Case 4: C = fill(0.0) dominates matmul — canFuse returns true, fusion applied
// CHECK: func.func @case4_fill_zero_fuse
// CHECK: linalg.fill
// CHECK: linalg.matmul {{.*}} outs(%{{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK-NOT: arith.addf
// CHECK: return %{{.*}} : tensor<32x32xf32>
  func.func @case4_fill_zero_fuse(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>, %init_D: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f32
    %empty_C = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %C = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%empty_C : tensor<32x32xf32>) -> tensor<32x32xf32>
    %D = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%init_D : tensor<32x32xf32>) -> tensor<32x32xf32>
    %E = arith.addf %D, %C {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    return %E : tensor<32x32xf32>
  }

// Case 5: C = broadcast(bias: tensor<32xf32> -> tensor<32x32xf32>), rank 1->2 — canFuse returns true, fusion applied
// CHECK: func.func @case5_broadcast_fuse
// CHECK: linalg.broadcast
// CHECK: linalg.matmul {{.*}} outs(%{{.*}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK-NOT: arith.addf
// CHECK: return %{{.*}} : tensor<32x32xf32>
  func.func @case5_broadcast_fuse(%A: tensor<32x64xf32>, %B: tensor<64x32xf32>, %bias: tensor<32xf32>, %init_D: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %empty_C = tensor.empty() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %C = linalg.broadcast {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} ins(%bias : tensor<32xf32>) outs(%empty_C : tensor<32x32xf32>) dimensions = [0]
    %D = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<32x64xf32>, tensor<64x32xf32>) outs(%init_D : tensor<32x32xf32>) -> tensor<32x32xf32>
    %E = arith.addf %D, %C {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    return %E : tensor<32x32xf32>
  }

// Integer-add variants (arith.addi) for the same scenarios
// Case 1 (addi): C is a block argument — canFuse returns false, no fusion
// CHECK: func.func @case1_no_defining_op_addi
// CHECK: linalg.matmul
// CHECK: arith.addi
// CHECK: return
  func.func @case1_no_defining_op_addi(%A: tensor<32x64xi32>, %B: tensor<64x32xi32>, %C: tensor<32x32xi32>, %init_D: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %D = linalg.matmul {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<32x64xi32>, tensor<64x32xi32>) outs(%init_D : tensor<32x32xi32>) -> tensor<32x32xi32>
    %E = arith.addi %D, %C {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xi32>
    return %E : tensor<32x32xi32>
  }

// Case 2 (addi): addi result is used by another matmul — canFuse returns false, no fusion
// CHECK: func.func @case2_addi_used_by_matmul
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: arith.addi
// CHECK: linalg.matmul
// CHECK: arith.addi
// CHECK: return
  func.func @case2_addi_used_by_matmul(%A: tensor<32x64xi32>, %B: tensor<64x32xi32>, %B2: tensor<32x16xi32>, %init_D: tensor<32x32xi32>, %init_F: tensor<32x16xi32>) -> tensor<32x16xi32> {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0 : i32
    %empty_C = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xi32>
    %C = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : i32) outs(%empty_C : tensor<32x32xi32>) -> tensor<32x32xi32>
    %D = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<32x64xi32>, tensor<64x32xi32>) outs(%init_D : tensor<32x32xi32>) -> tensor<32x32xi32>
    %E = arith.addi %D, %C {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xi32>
    %F = linalg.matmul {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%E, %B2 : tensor<32x32xi32>, tensor<32x16xi32>) outs(%init_F : tensor<32x16xi32>) -> tensor<32x16xi32>
    %ret = arith.addi %F, %B2 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x16xi32>
    return %ret : tensor<32x16xi32>
  }


// Case 3 (addi): C is defined inside scf.if, does not dominate matmul — canFuse returns false, no fusion
// CHECK: func.func @case3_no_dominance_addi
// CHECK: linalg.matmul
// CHECK: scf.if
// CHECK: arith.addi
// CHECK: return
  func.func @case3_no_dominance_addi(%A: tensor<32x64xi32>, %B: tensor<64x32xi32>, %cond: i1, %init_D: tensor<32x32xi32>, %fallback: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %D = linalg.matmul {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<32x64xi32>, tensor<64x32xi32>) outs(%init_D : tensor<32x32xi32>) -> tensor<32x32xi32>
    %C = scf.if %cond -> (tensor<32x32xi32>) {
      %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0 : i32
      %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xi32>
      %filled = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : i32) outs(%empty : tensor<32x32xi32>) -> tensor<32x32xi32>
      scf.yield %filled : tensor<32x32xi32>
    } else {
      scf.yield %fallback : tensor<32x32xi32>
    }
    %E = arith.addi %D, %C {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xi32>
    return %E : tensor<32x32xi32>
  }

// Case 4 (addi): C = fill(0) dominates matmul — canFuse returns true, fusion applied
// CHECK: func.func @case4_fill_zero_fuse_addi
// CHECK: linalg.fill
// CHECK: linalg.matmul {{.*}} outs(%{{.*}} : tensor<32x32xi32>) -> tensor<32x32xi32>
// CHECK-NOT: arith.addi
// CHECK: return %{{.*}} : tensor<32x32xi32>
  func.func @case4_fill_zero_fuse_addi(%A: tensor<32x64xi32>, %B: tensor<64x32xi32>, %init_D: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0 : i32
    %empty_C = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xi32>
    %C = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : i32) outs(%empty_C : tensor<32x32xi32>) -> tensor<32x32xi32>
    %D = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<32x64xi32>, tensor<64x32xi32>) outs(%init_D : tensor<32x32xi32>) -> tensor<32x32xi32>
    %E = arith.addi %D, %C {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xi32>
    return %E : tensor<32x32xi32>
  }

// Case 5 (addi): C = broadcast(bias: tensor<32xi32> -> tensor<32x32xi32>), rank 1->2 — canFuse returns true, fusion applied
// CHECK: func.func @case5_broadcast_fuse_addi
// CHECK: linalg.broadcast
// CHECK: linalg.matmul {{.*}} outs(%{{.*}} : tensor<32x32xi32>) -> tensor<32x32xi32>
// CHECK-NOT: arith.addi
// CHECK: return %{{.*}} : tensor<32x32xi32>
  func.func @case5_broadcast_fuse_addi(%A: tensor<32x64xi32>, %B: tensor<64x32xi32>, %bias: tensor<32xi32>, %init_D: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %empty_C = tensor.empty() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xi32>
    %C = linalg.broadcast {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} ins(%bias : tensor<32xi32>) outs(%empty_C : tensor<32x32xi32>) dimensions = [0]
    %D = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<32x64xi32>, tensor<64x32xi32>) outs(%init_D : tensor<32x32xi32>) -> tensor<32x32xi32>
    %E = arith.addi %D, %C {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xi32>
    return %E : tensor<32x32xi32>
  }
}
