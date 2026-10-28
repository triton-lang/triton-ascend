// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// A missing left scale is an identity, including K below one scale block.
// CHECK-LABEL: func.func @fp16_missing_both_small_k
// CHECK: arith.constant 127 : i8
// CHECK: tensor.empty() : tensor<16x1xi8>
// CHECK: linalg.matmul
// CHECK-SAME: outs(%arg2 : tensor<16x32xf32>)
func.func @fp16_missing_both_small_k(%a: tensor<16x16xf16>, %b: tensor<16x32xf16>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = true} : tensor<16x16xf16> * tensor<16x32xf16> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// The generated identity scale also follows the non-fast-math path.
// CHECK-LABEL: func.func @fp16_missing_both
// CHECK: arith.constant 127 : i8
// CHECK: tensor.empty() : tensor<32x1xi8>
// CHECK: linalg.matmul
// CHECK-SAME: outs(%arg2 : tensor<32x16xf32>)
func.func @fp16_missing_both(%a: tensor<32x32xf16>, %b: tensor<32x16xf16>, %c: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<32x32xf16> * tensor<32x16xf16> -> tensor<32x16xf32>
  return %d : tensor<32x16xf32>
}

// -----

// The right scale retains its original handling when the left scale is absent.
// CHECK-LABEL: func.func @fp16_missing_lhs
// CHECK: arith.constant 127 : i8
// CHECK: tensor.empty() : tensor<16x2xi8>
// CHECK: linalg.transpose ins(%arg2 : tensor<32x2xi8>)
// CHECK: linalg.matmul
// CHECK-SAME: outs(%arg3 : tensor<16x32xf32>)
func.func @fp16_missing_lhs(%a: tensor<16x64xf16>, %b: tensor<64x32xf16>, %sb: tensor<32x2xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = fp16 rhs = fp16 {fastMath = true} : tensor<16x64xf16> * tensor<64x32xf16>, tensor<32x2xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @fp16_missing_lhs_strict
// CHECK: arith.constant 127 : i8
// CHECK: tensor.empty() : tensor<32x1xi8>
// CHECK: linalg.transpose ins(%arg2 : tensor<16x1xi8>)
// CHECK: linalg.matmul
// CHECK-SAME: outs(%arg3 : tensor<32x16xf32>)
func.func @fp16_missing_lhs_strict(%a: tensor<32x32xf16>, %b: tensor<32x16xf16>, %sb: tensor<16x1xi8>, %c: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<32x32xf16> * tensor<32x16xf16>, tensor<16x1xi8> -> tensor<32x16xf32>
  return %d : tensor<32x16xf32>
}

// -----

// Explicit left scales continue through their existing E8M0 decoder.
// CHECK-LABEL: func.func @fp16_explicit_lhs
// CHECK-NOT: arith.constant 127 : i8
// CHECK: arith.extui %arg2 : tensor<16x2xi8> to tensor<16x2xi16>
// CHECK: linalg.matmul
// CHECK-SAME: outs(%arg3 : tensor<16x32xf32>)
func.func @fp16_explicit_lhs(%a: tensor<16x64xf16>, %b: tensor<64x32xf16>, %sa: tensor<16x2xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b, %c lhs = fp16 rhs = fp16 {fastMath = true} : tensor<16x64xf16>, tensor<16x2xi8> * tensor<64x32xf16> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}
