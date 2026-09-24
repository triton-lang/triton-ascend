// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// Packed FP4 has two logical K elements per byte. M and N intentionally differ.
// CHECK-LABEL: func.func @fp4_missing_lhs
// CHECK: arith.constant 127 : i8
// CHECK: tensor.empty() : tensor<16x2xi8>
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x32xi8>, tensor<32x64xi8>, tensor<16x2xi8>, tensor<64x2xi8>
func.func @fp4_missing_lhs(%a: tensor<16x32xi8>, %b: tensor<32x64xi8>, %sb: tensor<64x2xi8>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x32xi8> * tensor<32x64xi8>, tensor<64x2xi8> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// FP8 K is not packed; a missing scale uses the E8M0 encoding of one.
// CHECK-LABEL: func.func @fp8_missing_lhs
// CHECK: arith.constant 127 : i8
// CHECK: tensor.empty() : tensor<16x2xi8>
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x64xf8E4M3FN>, tensor<64x32xf8E5M2>, tensor<16x2xi8>, tensor<32x2xi8>
func.func @fp8_missing_lhs(%a: tensor<16x64xf8E4M3FN>, %b: tensor<64x32xf8E5M2>, %sb: tensor<32x2xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = e4m3 rhs = e5m2 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x32xf8E5M2>, tensor<32x2xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// The ordinary floating-point path represents scales as signed exponents.
// CHECK-LABEL: func.func @fp16_missing_lhs
// CHECK: arith.constant 0 : i8
// CHECK: tensor.empty() : tensor<16x2xi8>
// CHECK: linalg.matmul
func.func @fp16_missing_lhs(%a: tensor<16x64xf16>, %b: tensor<64x32xf16>, %sb: tensor<32x2xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x32xf16>, tensor<32x2xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_missing_lhs
// CHECK: arith.constant 0 : i8
// CHECK: tensor.empty() : tensor<32x1xi8>
// CHECK: linalg.matmul
func.func @bf16_missing_lhs(%a: tensor<32x32xbf16>, %b: tensor<32x16xbf16>, %sb: tensor<16x1xi8>, %c: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = bf16 rhs = bf16 {fastMath = false} : tensor<32x32xbf16> * tensor<32x16xbf16>, tensor<16x1xi8> -> tensor<32x16xf32>
  return %d : tensor<32x16xf32>
}

// -----

// Neither side needs scaling; K below one scale block must remain nonempty.
// CHECK-LABEL: func.func @fp16_missing_both_small_k
// CHECK: arith.constant 0 : i8
// CHECK: tensor.empty() : tensor<16x1xi8>
// CHECK: linalg.matmul
func.func @fp16_missing_both_small_k(%a: tensor<16x16xf16>, %b: tensor<16x32xf16>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = true} : tensor<16x16xf16> * tensor<16x32xf16> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// Explicit scales must still be passed through unchanged.
// CHECK-LABEL: func.func @fp4_explicit_scales
// CHECK-NOT: linalg.fill
// CHECK: hfusion.matmul_mx
// CHECK-SAME: ins(%arg0, %arg1, %arg2, %arg3 :
func.func @fp4_explicit_scales(%a: tensor<16x32xi8>, %b: tensor<32x64xi8>, %sa: tensor<16x2xi8>, %sb: tensor<64x2xi8>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x32xi8>, tensor<16x2xi8> * tensor<32x64xi8>, tensor<64x2xi8> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}
