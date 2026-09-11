// RUN: triton-opt --split-input-file --split-if-by-block-id %s | FileCheck %s

// Else-side tensor.empty placeholders are wrapped as the DPS init of a
// linalg.fill: a bare tensor.empty is only referenced by scf.yield, so the
// bufferized alloc has no real consumer and no lifetime in regbase
// PlanMemory. The fill op, the zero constant, and the empty all carry the
// consuming group's block_id.

// V-C-V-C 4-group split: both VECTOR else blocks build the fill chain
// (tensor.empty + zero + linalg.fill) and yield the fill result; the CUBE
// group traces to the dominating outs root and needs no placeholder.
// CHECK-LABEL: func.func @split_then_vcvc_fill_placeholder
// CHECK: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// First VECTOR else: placeholder filled with zero.
// CHECK: tensor.empty() {ssbuffer.block_id = 94 : i32} : tensor<2x2xf32>
// CHECK: arith.constant {{.*}}ssbuffer.block_id = 94 : i32} 0.000000e+00 : f32
// CHECK: linalg.fill {ssbuffer.block_id = 94 : i32} ins({{.*}} : f32) outs({{.*}} : tensor<2x2xf32>)
// CHECK: scf.yield {ssbuffer.block_id = 94 : i32} {{.*}} : tensor<2x2xf32>
// CHECK: scf.if
// CUBE group 95: matmul traces to the outs root, no placeholder.
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK-NOT: linalg.fill {ssbuffer.block_id = 95
// CHECK: scf.yield {ssbuffer.block_id = 95 : i32} %arg0 : tensor<2x2xf32>
// CHECK: scf.if
// Second VECTOR else: placeholder filled with zero again.
// CHECK: tensor.empty() {ssbuffer.block_id = 96 : i32} : tensor<2x2xf32>
// CHECK: arith.constant {{.*}}ssbuffer.block_id = 96 : i32} 0.000000e+00 : f32
// CHECK: linalg.fill {ssbuffer.block_id = 96 : i32} ins({{.*}} : f32) outs({{.*}} : tensor<2x2xf32>)
// CHECK: scf.yield {ssbuffer.block_id = 96 : i32} {{.*}} : tensor<2x2xf32>
// CHECK: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 97
func.func @split_then_vcvc_fill_placeholder(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    scf.if %cond {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m = linalg.matmul ins(%v1, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v2 = arith.mulf %m, %b {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m2 = linalg.matmul ins(%v2, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 97 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
    }
  }
  return
}

// -----

// C-V-C groups inside the else branch: the middle VECTOR group (tensor
// addf) gets a filled placeholder in its else, the CUBE groups trace to
// the outs root %a.
// CHECK-LABEL: func.func @split_else_cvc_fill_placeholder
// CHECK: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: scf.yield {ssbuffer.block_id = 95 : i32} %arg0 : tensor<2x2xf32>
// CHECK: scf.if
// CHECK: tensor.empty() {ssbuffer.block_id = 96 : i32} : tensor<2x2xf32>
// CHECK: arith.constant {{.*}}ssbuffer.block_id = 96 : i32} 0.000000e+00 : f32
// CHECK: linalg.fill {ssbuffer.block_id = 96 : i32} ins({{.*}} : f32) outs({{.*}} : tensor<2x2xf32>)
// CHECK: scf.yield {ssbuffer.block_id = 96 : i32} {{.*}} : tensor<2x2xf32>
// CHECK: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 97
func.func @split_else_cvc_fill_placeholder(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: f32, %d: f32, %cond: i1) {
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

// CUBE group's matmul outs is an OpResult from fill+empty inside the if,
// so the CUBE else-block creates a filled placeholder instead of reusing
// the outs root.
// C-V-C groups: CUBE(95,matmul) -> VECTOR(96) -> CUBE(97)
// CHECK-LABEL: func.func @split_cube_matmul_outs_op_result
// CUBE 95 then: fill+empty+matmul; else: fill placeholder (OpResult path)
// CHECK: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: scf.yield {ssbuffer.block_id = 95 : i32}
// CHECK: } else {
// CHECK: tensor.empty() {ssbuffer.block_id = 95 : i32} : tensor<2x2xf32>
// CHECK: arith.constant {{.*}}ssbuffer.block_id = 95 : i32} 0.000000e+00 : f32
// CHECK: linalg.fill {ssbuffer.block_id = 95 : i32}
// CHECK: scf.yield {ssbuffer.block_id = 95 : i32}
// VECTOR 96
// CHECK: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 96
// CUBE 97: last group, void if (no else, no yield)
// CHECK: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 97
func.func @split_cube_matmul_outs_op_result(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    scf.if %cond {
      %empty = tensor.empty() : tensor<2x2xf32>
      %z = arith.constant 0.000000e+00 : f32
      %filled = linalg.fill ins(%z : f32) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
      %m = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%filled : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v = arith.addf %m, %m {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m2 = linalg.matmul ins(%a, %v : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 97 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
    }
  }
  return
}
