// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=true" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=false" --split-input-file %s | FileCheck %s

// M/N packing doubles the non-K axes. Missing scales are identities.
// CHECK-LABEL: func.func @fp4_non_k_packed
// CHECK-NOT: hfusion.matmul_mx
// CHECK: tensor.collapse_shape {{.*}} : tensor<8x2x32xbf16> into tensor<16x32xbf16>
// CHECK: tensor.collapse_shape {{.*}} : tensor<32x16x2xbf16> into tensor<32x32xbf16>
// CHECK: linalg.matmul
// CHECK-SAME: tensor<16x32xbf16>, tensor<32x32xbf16>
func.func @fp4_non_k_packed(%a: tensor<8x32xi8>, %b: tensor<32x16xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = e2m1 rhs = e2m1 {fastMath = false, lhs_k_pack = false, rhs_k_pack = false} : tensor<8x32xi8> * tensor<32x16xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// As upstream Fp4ToFpOp does, decode directly to FP16 for a mixed FP16 dot.
// CHECK-LABEL: func.func @fp4_lhs_fp16_rhs
// CHECK-NOT: hfusion.matmul_mx
// CHECK-NOT: arith.extf
// CHECK: arith.bitcast {{.*}} : tensor<16x16xi16> to tensor<16x16xf16>
// CHECK: arith.bitcast {{.*}} : tensor<16x16xi16> to tensor<16x16xf16>
// CHECK: tensor.collapse_shape {{.*}} : tensor<16x16x2xf16> into tensor<16x32xf16>
// CHECK: linalg.matmul
// CHECK-SAME: tensor<16x32xf16>, tensor<32x32xf16>
func.func @fp4_lhs_fp16_rhs(%a: tensor<16x16xi8>, %b: tensor<32x32xf16>, %sa: tensor<16x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b, %c lhs = e2m1 rhs = fp16 {fastMath = false} : tensor<16x16xi8>, tensor<16x1xi8> * tensor<32x32xf16> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @fp16_lhs_fp4_rhs
// CHECK-NOT: hfusion.matmul_mx
// CHECK-NOT: arith.extf
// CHECK: arith.bitcast {{.*}} : tensor<16x32xi16> to tensor<16x32xf16>
// CHECK: arith.bitcast {{.*}} : tensor<16x32xi16> to tensor<16x32xf16>
// CHECK: tensor.collapse_shape {{.*}} : tensor<16x2x32xf16> into tensor<32x32xf16>
// CHECK: linalg.matmul
// CHECK-SAME: tensor<16x32xf16>, tensor<32x32xf16>
func.func @fp16_lhs_fp4_rhs(%a: tensor<16x32xf16>, %b: tensor<16x32xi8>, %sa: tensor<16x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b, %c lhs = fp16 rhs = e2m1 {fastMath = false} : tensor<16x32xf16>, tensor<16x1xi8> * tensor<16x32xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}
