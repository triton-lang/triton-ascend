// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=true" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=false" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @e8m0_boundaries
// E8M0 code 0 is the BF16 subnormal 2^-127 (bits 64).
// CHECK-DAG: arith.constant{{.*}}64{{.*}}i16
// CHECK: arith.cmpi eq
// CHECK: arith.select
// CHECK: arith.bitcast {{.*}} : tensor<16x1xi16> to tensor<16x1xbf16>
// Code 255 is masked to NaN after scaling, as in upstream maskNan.
// CHECK: arith.mulf
// CHECK: arith.cmpi eq
// CHECK: arith.select
// CHECK: linalg.matmul
func.func @e8m0_boundaries(%a: tensor<16x32xbf16>, %b: tensor<32x32xbf16>, %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = bf16 rhs = bf16 {fastMath = false} : tensor<16x32xbf16>, tensor<16x1xi8> * tensor<32x32xbf16>, tensor<32x1xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// Fast math keeps the upstream exponent-bit decode for both operands.
// In particular, code 0 must not get the strict-mode subnormal correction.
// CHECK-LABEL: func.func @e8m0_fast_math
// CHECK: arith.extui {{.*}} : tensor<16x1xi8> to tensor<16x1xi16>
// CHECK: arith.shli
// CHECK-NEXT: {{.*}}arith.bitcast {{.*}} : tensor<16x1xi16> to tensor<16x1xbf16>
// CHECK: arith.extui {{.*}} : tensor<1x32xi8> to tensor<1x32xi16>
// CHECK: arith.shli
// CHECK-NEXT: {{.*}}arith.bitcast {{.*}} : tensor<1x32xi16> to tensor<1x32xbf16>
// CHECK-NOT: arith.select
// CHECK: linalg.matmul
func.func @e8m0_fast_math(%a: tensor<16x32xbf16>, %b: tensor<32x32xbf16>, %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = bf16 rhs = bf16 {fastMath = true} : tensor<16x32xbf16>, tensor<16x1xi8> * tensor<32x32xbf16>, tensor<32x1xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// FP16 scales must take the upstream integer -> FP32 -> FP16 path.
// CHECK-LABEL: func.func @e8m0_fp16
// CHECK-NOT: bf16
// CHECK: arith.extui {{.*}} : tensor<16x1xi8> to tensor<16x1xi32>
// CHECK-NOT: bf16
// CHECK: arith.bitcast {{.*}} : tensor<16x1xi32> to tensor<16x1xf32>
// CHECK-NEXT: {{.*}}arith.truncf {{.*}} : tensor<16x1xf32> to tensor<16x1xf16>
// CHECK-NOT: bf16
// CHECK: arith.extui {{.*}} : tensor<1x32xi8> to tensor<1x32xi32>
// CHECK: arith.bitcast {{.*}} : tensor<1x32xi32> to tensor<1x32xf32>
// CHECK-NEXT: {{.*}}arith.truncf {{.*}} : tensor<1x32xf32> to tensor<1x32xf16>
// CHECK-NOT: bf16
// CHECK: linalg.matmul
func.func @e8m0_fp16(%a: tensor<16x32xf16>, %b: tensor<32x32xf16>, %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x32xf16>, tensor<16x1xi8> * tensor<32x32xf16>, tensor<32x1xi8> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: func.func @floating_rhs_scale
// CHECK: arith.extf {{.*}} : tensor<2x32xf8E4M3FN> to tensor<2x32xbf16>
// CHECK: arith.mulf {{.*}} : tensor<32x32xbf16>
// CHECK: arith.cmpf une
// CHECK: linalg.matmul
func.func @floating_rhs_scale(%a: tensor<16x32xbf16>, %b: tensor<32x32xbf16>, %sb: tensor<32x2xf8E4M3FN>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a, %b scale %sb, %c lhs = bf16 rhs = bf16 {fastMath = false} : tensor<16x32xbf16> * tensor<32x32xbf16>, tensor<32x2xf8E4M3FN> -> tensor<16x32xf32>
  return %d : tensor<16x32xf32>
}
