// RUN: triton-opt --split-input-file --split-if-by-block-id %s | FileCheck %s

// VCV then-side split: VECTOR placeholder (block_id=94) in else block
// produces linalg.fill + arith.constant 0; CUBE placeholder (block_id=95)
// reuses the matmul source tensor without fill.
// CHECK-LABEL: func.func @vector_placeholder_has_fill
// VECTOR placeholder has fill in else block.
// CHECK: tensor.empty() {ssbuffer.block_id = 94 : i32}
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: linalg.fill {ssbuffer.block_id = 94 : i32}
// CUBE placeholder reuses matmul source, no fill.
// CHECK-NOT: linalg.fill {ssbuffer.block_id = 95 : i32}
func.func @vector_placeholder_has_fill(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) {
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

// CVC else-side split: VECTOR placeholder (block_id=96) in then block
// also produces linalg.fill.
// CHECK-LABEL: func.func @vector_placeholder_has_fill_else
// CHECK: tensor.empty() {ssbuffer.block_id = 96 : i32}
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: linalg.fill {ssbuffer.block_id = 96 : i32}
func.func @vector_placeholder_has_fill_else(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: f32, %d: f32, %cond: i1) {
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

// Matmul bias comes from tensor.empty + linalg.fill chain (outside the if).
// The CUBE placeholder traces through createMatmulPlaceHolderValue and
// produces only tensor::EmptyOp without linalg.fill.
// CHECK-LABEL: func.func @matmul_placeholder_no_fill
// VECTOR placeholder: fill + constant present.
// CHECK: tensor.empty() {ssbuffer.block_id = 94 : i32}
// CHECK: linalg.fill {ssbuffer.block_id = 94 : i32}
// CUBE placeholder: only tensor.empty, no linalg.fill.
// CHECK: tensor.empty() {ssbuffer.block_id = 95 : i32}
// CHECK-NOT: linalg.fill {ssbuffer.block_id = 95 : i32}
func.func @matmul_placeholder_no_fill(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x2xf32>
  %zero = linalg.fill ins(%cst : f32) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    scf.if %cond {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m = linalg.matmul ins(%v1, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%zero : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v2 = arith.mulf %m, %b {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    }
  }
  return
}

// -----

// A rectangular PV result needs the same storage in both branches. Keeping
// an unwritten empty in the inactive branch creates an L0C alias without a
// producer after bufferization.
// CHECK-LABEL: func.func @outer_matmul_shared_empty
// CHECK: %[[EMPTY:.*]] = tensor.empty() {{.*}} : tensor<128x64xf32>
// CHECK: scf.if {{.*}} -> (tensor<128x64xf32>) {
// CHECK: linalg.fill {{.*}} outs(%[[EMPTY]] : tensor<128x64xf32>)
// CHECK: linalg.matmul
// CHECK: } else {
// CHECK-NEXT: scf.yield {{.*}}%[[EMPTY]] : tensor<128x64xf32>
func.func @outer_matmul_shared_empty(%a: tensor<128x128xbf16>, %b: tensor<128x64xbf16>, %out: tensor<128x64xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  scf.for %iv = %c0 to %c1 step %c1 {
    scf.if %cond {
      %v = arith.addf %a, %a {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xbf16>
      %empty = tensor.empty() {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x64xf32>
      %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x64xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<128x64xf32>
      %m = linalg.matmul ins(%v, %b : tensor<128x128xbf16>, tensor<128x64xbf16>) outs(%init : tensor<128x64xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<128x64xf32>
      %result = arith.addf %m, %out {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32>
    }
  } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
  return
}
