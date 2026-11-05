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
