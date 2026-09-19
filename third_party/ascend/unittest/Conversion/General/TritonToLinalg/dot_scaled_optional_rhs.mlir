// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @fp4_missing_rhs
// CHECK: %[[ONE:.*]] = arith.constant 127 : i8
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<64x2xi8>
// CHECK: %[[SCALE:.*]] = linalg.fill ins(%[[ONE]] : i8) outs(%[[EMPTY]] : tensor<64x2xi8>) -> tensor<64x2xi8>
// CHECK: hfusion.matmul_mx {{.*}}%[[SCALE]]
// CHECK-NOT: tt.dot_scaled
func.func @fp4_missing_rhs(%a: tensor<16x32xi8>, %sa: tensor<16x2xi8>, %b: tensor<32x64xi8>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %d = tt.dot_scaled %a scale %sa, %b, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x32xi8>, tensor<16x2xi8> * tensor<32x64xi8> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @fp4_missing_rhs_multiple_blocks
// CHECK: %[[ONE:.*]] = arith.constant 127 : i8
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16x4xi8>
// CHECK: %[[SCALE:.*]] = linalg.fill ins(%[[ONE]] : i8) outs(%[[EMPTY]] : tensor<16x4xi8>) -> tensor<16x4xi8>
// CHECK: hfusion.matmul_mx {{.*}}%[[SCALE]]
// CHECK-NOT: tt.dot_scaled
func.func @fp4_missing_rhs_multiple_blocks(%a: tensor<32x64xi8>, %sa: tensor<32x4xi8>, %b: tensor<64x16xi8>, %c: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %d = tt.dot_scaled %a scale %sa, %b, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x64xi8>, tensor<32x4xi8> * tensor<64x16xi8> -> tensor<32x16xf32>
  return %d : tensor<32x16xf32>
}

// -----

// CHECK-LABEL: func.func @fp8_missing_rhs
// CHECK: %[[ONE:.*]] = arith.constant 127 : i8
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<64x1xi8>
// CHECK: %[[SCALE:.*]] = linalg.fill ins(%[[ONE]] : i8) outs(%[[EMPTY]] : tensor<64x1xi8>) -> tensor<64x1xi8>
// CHECK: hfusion.matmul_mx {{.*}}%[[SCALE]]
// CHECK-NOT: tt.dot_scaled
func.func @fp8_missing_rhs(%a: tensor<16x32xf8E4M3FN>, %sa: tensor<16x1xi8>, %b: tensor<32x64xf8E5M2>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %d = tt.dot_scaled %a scale %sa, %b, %c lhs = e4m3 rhs = e5m2 {fastMath = false} : tensor<16x32xf8E4M3FN>, tensor<16x1xi8> * tensor<32x64xf8E5M2> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @fp8_missing_rhs_multiple_blocks
// CHECK: %[[ONE:.*]] = arith.constant 127 : i8
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16x2xi8>
// CHECK: %[[SCALE:.*]] = linalg.fill ins(%[[ONE]] : i8) outs(%[[EMPTY]] : tensor<16x2xi8>) -> tensor<16x2xi8>
// CHECK: hfusion.matmul_mx {{.*}}%[[SCALE]]
// CHECK-NOT: tt.dot_scaled
func.func @fp8_missing_rhs_multiple_blocks(%a: tensor<32x64xf8E5M2>, %sa: tensor<32x2xi8>, %b: tensor<64x16xf8E4M3FN>, %c: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %d = tt.dot_scaled %a scale %sa, %b, %c lhs = e5m2 rhs = e4m3 {fastMath = false} : tensor<32x64xf8E5M2>, tensor<32x2xi8> * tensor<64x16xf8E4M3FN> -> tensor<32x16xf32>
  return %d : tensor<32x16xf32>
}

// -----

// CHECK-LABEL: func.func @fp4_explicit_scales
// CHECK-NOT: linalg.fill
// CHECK: hfusion.matmul_mx
// CHECK-NOT: tt.dot_scaled
func.func @fp4_explicit_scales(%a: tensor<16x32xi8>, %sa: tensor<16x2xi8>, %b: tensor<32x64xi8>, %sb: tensor<64x2xi8>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x32xi8>, tensor<16x2xi8> * tensor<32x64xi8>, tensor<64x2xi8> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}
