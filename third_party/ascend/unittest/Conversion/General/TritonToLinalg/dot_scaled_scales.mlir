// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=true" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=false" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @e8m0_boundaries
// BF16 encodings: 2^-127 = 64, quiet NaN = 32704.
// CHECK-DAG: arith.constant{{.*}}64{{.*}}i16
// CHECK-DAG: arith.constant{{.*}}32704{{.*}}i16
// CHECK: linalg.matmul
func.func @e8m0_boundaries(%a: tensor<16x32xbf16>, %b: tensor<32x32xbf16>, %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = bf16 rhs = bf16 {fastMath = false} : tensor<16x32xbf16>, tensor<16x1xi8> * tensor<32x32xbf16>, tensor<32x1xi8> -> tensor<16x32xf32>
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
