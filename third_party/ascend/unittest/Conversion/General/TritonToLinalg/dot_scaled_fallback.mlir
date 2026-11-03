// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=true" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=false" --split-input-file %s | FileCheck %s

// Decode FP4 before multiplying mixed-format operands.
// CHECK-LABEL: func.func @mixed_fp4_fp8
// CHECK-NOT: hfusion.matmul_mx
// CHECK: tensor.collapse_shape {{.*}} : tensor<16x16x2xf32> into tensor<16x32xf32>
// CHECK: arith.extf {{.*}} : tensor<32x32xf8E5M2> to tensor<32x32xf32>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK-SAME: tensor<16x32xf32>, tensor<32x32xf32>
func.func @mixed_fp4_fp8(%a: tensor<16x16xi8>, %b: tensor<32x32xf8E5M2>, %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e5m2 {fastMath = false} : tensor<16x16xi8>, tensor<16x1xi8> * tensor<32x32xf8E5M2>, tensor<32x1xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// M/N packing doubles the non-K axes. Missing scales are identities.
// CHECK-LABEL: func.func @fp4_non_k_packed
// CHECK-NOT: hfusion.matmul_mx
// CHECK: tensor.collapse_shape {{.*}} : tensor<8x2x32xf32> into tensor<16x32xf32>
// CHECK: tensor.collapse_shape {{.*}} : tensor<32x16x2xf32> into tensor<32x32xf32>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK-SAME: tensor<16x32xf32>, tensor<32x32xf32>
func.func @fp4_non_k_packed(%a: tensor<8x32xi8>, %b: tensor<32x16xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b, %c lhs = e2m1 rhs = e2m1 {fastMath = false, lhs_k_pack = false, rhs_k_pack = false} : tensor<8x32xi8> * tensor<32x16xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// Floating E4M3 scales use groups of 16 and cannot use the E8M0 MX ABI.
// CHECK-LABEL: func.func @nvfp4
// CHECK-NOT: hfusion.matmul_mx
// CHECK: arith.extf {{.*}} : tensor<16x2xf8E4M3FN> to tensor<16x2xf32>
// CHECK: tensor.collapse_shape {{.*}} : tensor<16x2x16xf32> into tensor<16x32xf32>
// CHECK: arith.extf {{.*}} : tensor<32x2xf8E4M3FN> to tensor<32x2xf32>
// CHECK: tensor.collapse_shape {{.*}} : tensor<32x2x16xf32> into tensor<32x32xf32>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
func.func @nvfp4(%a: tensor<16x16xi8>, %b: tensor<16x32xi8>, %sa: tensor<16x2xf8E4M3FN>, %sb: tensor<32x2xf8E4M3FN>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x16xi8>, tensor<16x2xf8E4M3FN> * tensor<16x32xi8>, tensor<32x2xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// E8M0 0/255 are finite 2^-127/NaN. Scale and accumulate in FP32, then round.
// CHECK-LABEL: func.func @mixed_half_scales
// CHECK-DAG: arith.constant 4194304 : i32
// CHECK-DAG: arith.constant 2143289344 : i32
// CHECK: arith.extf {{.*}} : tensor<16x32xf16> to tensor<16x32xf32>
// CHECK: arith.mulf {{.*}} : tensor<16x32xf32>
// CHECK: arith.extf {{.*}} : tensor<32x32xbf16> to tensor<32x32xf32>
// CHECK: arith.mulf {{.*}} : tensor<32x32xf32>
// CHECK: arith.extf {{.*}} : tensor<16x32xbf16> to tensor<16x32xf32>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: arith.truncf {{.*}} {round_mode = #hfusion.round_mode<rint>} : tensor<16x32xf32> to tensor<16x32xbf16>
func.func @mixed_half_scales(%a: tensor<16x32xf16>, %b: tensor<32x32xbf16>, %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xbf16>) -> tensor<16x32xbf16> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = fp16 rhs = bf16 {fastMath = false} : tensor<16x32xf16>, tensor<16x1xi8> * tensor<32x32xbf16>, tensor<32x1xi8> -> tensor<16x32xbf16>
  return %d : tensor<16x32xbf16>
}

// -----

// A missing software-path scale does not require constructing an MX scale.
// CHECK-LABEL: func.func @fp16_missing_lhs
// CHECK-NOT: hfusion.matmul_mx
// CHECK: arith.extf {{.*}} : tensor<16x32xf16> to tensor<16x32xf32>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
func.func @fp16_missing_lhs(%a: tensor<16x32xf16>, %b: tensor<32x32xf16>, %sb: tensor<32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x32xf16> * tensor<32x32xf16>, tensor<32x1xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}
