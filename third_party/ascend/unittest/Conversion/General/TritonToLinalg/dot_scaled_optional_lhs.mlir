// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// Packed FP4 has two logical K elements per byte. M and N intentionally differ.
// CHECK-LABEL: func.func @fp4_missing_lhs
// CHECK-NOT: scf.for
// CHECK: %[[ONE:[^ ]+]] = arith.constant 127 : i8
// CHECK: %[[EMPTY:[^ ]+]] = tensor.empty() : tensor<16x4xi8>
// CHECK: %[[SCALE:[^ ]+]] = linalg.fill ins(%[[ONE]] : i8) outs(%[[EMPTY]] : tensor<16x4xi8>) -> tensor<16x4xi8>
// CHECK: hfusion.matmul_mx {{.*}}%[[SCALE]]
// CHECK-SAME: tensor<16x64xi8>, tensor<64x64xi8>, tensor<16x4xi8>, tensor<64x4xi8>
// CHECK-SAME: outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: scf.for
// CHECK: return {{.*}} : tensor<16x64xf32>
// CHECK-NOT: tt.dot_scaled
func.func @fp4_missing_lhs(%a: tensor<16x64xi8>, %b: tensor<64x64xi8>, %sb: tensor<64x4xi8>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x64xi8> * tensor<64x64xi8>, tensor<64x4xi8> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}

// -----

// FP8 K is not packed; a missing scale uses the E8M0 encoding of one.
// CHECK-LABEL: func.func @fp8_missing_lhs
// CHECK-NOT: scf.for
// CHECK: %[[ONE:[^ ]+]] = arith.constant 127 : i8
// CHECK: %[[EMPTY:[^ ]+]] = tensor.empty() : tensor<16x2xi8>
// CHECK: %[[SCALE:[^ ]+]] = linalg.fill ins(%[[ONE]] : i8) outs(%[[EMPTY]] : tensor<16x2xi8>) -> tensor<16x2xi8>
// CHECK: hfusion.matmul_mx {{.*}}%[[SCALE]]
// CHECK-SAME: tensor<16x64xf8E4M3FN>, tensor<64x32xf8E5M2>, tensor<16x2xi8>, tensor<32x2xi8>
// CHECK-SAME: outs({{.*}} : tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: scf.for
// CHECK: return {{.*}} : tensor<16x32xf32>
// CHECK-NOT: tt.dot_scaled
func.func @fp8_missing_lhs(%a: tensor<16x64xf8E4M3FN>, %b: tensor<64x32xf8E5M2>, %sb: tensor<32x2xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = e4m3 rhs = e5m2 {fastMath = false} : tensor<16x64xf8E4M3FN> * tensor<64x32xf8E5M2>, tensor<32x2xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// Explicit scales must still be passed through unchanged.
// CHECK-LABEL: func.func @fp4_explicit_scales
// CHECK-NOT: scf.for
// CHECK-NOT: linalg.fill
// CHECK: hfusion.matmul_mx
// CHECK-SAME: tensor<16x64xi8>, tensor<64x64xi8>, tensor<16x4xi8>, tensor<64x4xi8>
// CHECK-SAME: outs({{.*}} : tensor<16x64xf32>) -> tensor<16x64xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: scf.for
// CHECK: return {{.*}} : tensor<16x64xf32>
// CHECK-NOT: tt.dot_scaled
func.func @fp4_explicit_scales(%a: tensor<16x64xi8>, %b: tensor<64x64xi8>, %sa: tensor<16x4xi8>, %sb: tensor<64x4xi8>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x64xi8>, tensor<16x4xi8> * tensor<64x64xi8>, tensor<64x4xi8> -> tensor<16x64xf32>
  return %d : tensor<16x64xf32>
}
