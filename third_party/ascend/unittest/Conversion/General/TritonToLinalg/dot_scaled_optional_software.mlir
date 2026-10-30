// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s


// A missing scale in the fallback path is an identity; no scale tensor is needed.
// CHECK-LABEL: func.func @fp16_missing_lhs
// CHECK-NOT: arith.constant 127 : i8
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
func.func @fp16_missing_lhs(%a: tensor<16x64xf16>, %b: tensor<64x32xf16>, %sb: tensor<32x2xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x32xf16>, tensor<32x2xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_missing_lhs
// CHECK-NOT: arith.constant 127 : i8
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
func.func @bf16_missing_lhs(%a: tensor<32x32xbf16>, %b: tensor<32x16xbf16>, %sb: tensor<16x1xi8>, %c: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = bf16 rhs = bf16 {fastMath = false} : tensor<32x32xbf16> * tensor<32x16xbf16>, tensor<16x1xi8> -> tensor<32x16xf32>
  return %d : tensor<32x16xf32>
}

// -----

// Neither side needs scaling; K below one scale block must remain nonempty.
// CHECK-LABEL: func.func @fp16_missing_both_small_k
// CHECK-NOT: arith.constant 127 : i8
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
func.func @fp16_missing_both_small_k(%a: tensor<16x16xf16>, %b: tensor<16x32xf16>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = true} : tensor<16x16xf16> * tensor<16x32xf16> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}
