// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=true" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=false" --split-input-file %s | FileCheck %s

// Scale and accumulate in FP32, then round the final result.
// CHECK-LABEL: func.func @mixed_half_scales
// CHECK-DAG: arith.extf {{.*}} : tensor<16x32xf16> to tensor<16x32xf32>
// CHECK-DAG: arith.mulf {{.*}} : tensor<16x32xf32>
// CHECK-DAG: arith.extf {{.*}} : tensor<32x32xbf16> to tensor<32x32xf32>
// CHECK-DAG: arith.mulf {{.*}} : tensor<32x32xf32>
// CHECK-DAG: arith.extf {{.*}} : tensor<16x32xbf16> to tensor<16x32xf32>
// CHECK: linalg.matmul {{.*}}input_precision = "ieee"
// CHECK: arith.truncf {{.*}} {round_mode = #hfusion.round_mode<rint>} : tensor<16x32xf32> to tensor<16x32xbf16>
func.func @mixed_half_scales(%a: tensor<16x32xf16>, %b: tensor<32x32xbf16>, %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %c: tensor<16x32xbf16>) -> tensor<16x32xbf16> {
  %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = fp16 rhs = bf16 {fastMath = false} : tensor<16x32xf16>, tensor<16x1xi8> * tensor<32x32xbf16>, tensor<32x1xi8> -> tensor<16x32xbf16>
  return %d : tensor<16x32xbf16>
}
