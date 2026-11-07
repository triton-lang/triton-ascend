// RUN: triton-opt %s --triton-to-hfusion=compile-on-910-95=true | FileCheck %s --check-prefixes=REG,COMMON
// RUN: triton-opt %s --triton-to-hfusion | FileCheck %s --check-prefixes=DEFAULT,COMMON

// REG-LABEL: tt.func @precise_divf_f32_tensor
// REG-SAME: (%[[LHS:.*]]: tensor<64xf32>, %[[RHS:.*]]: tensor<64xf32>)
// REG: %[[EMPTY:.*]] = tensor.empty() : tensor<64xf32>
// REG: %[[QUOTIENT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<divfhp>} ins(%[[LHS]], %[[RHS]] : tensor<64xf32>, tensor<64xf32>) outs(%[[EMPTY]] : tensor<64xf32>) -> tensor<64xf32>
// REG: tt.return %[[QUOTIENT]] : tensor<64xf32>
// DEFAULT-LABEL: tt.func @precise_divf_f32_tensor
// DEFAULT: tt.precise_divf {{.*}} : tensor<64xf32>
tt.func @precise_divf_f32_tensor(%lhs: tensor<64xf32>, %rhs: tensor<64xf32>) -> tensor<64xf32> {
  %0 = tt.precise_divf %lhs, %rhs : tensor<64xf32>
  tt.return %0 : tensor<64xf32>
}

// COMMON-LABEL: tt.func @precise_divf_f16_tensor
// COMMON: tt.precise_divf {{.*}} : tensor<64xf16>
tt.func @precise_divf_f16_tensor(%lhs: tensor<64xf16>, %rhs: tensor<64xf16>) -> tensor<64xf16> {
  %0 = tt.precise_divf %lhs, %rhs : tensor<64xf16>
  tt.return %0 : tensor<64xf16>
}

// COMMON-LABEL: tt.func @precise_divf_f32_scalar
// COMMON: tt.precise_divf {{.*}} : f32
tt.func @precise_divf_f32_scalar(%lhs: f32, %rhs: f32) -> f32 {
  %0 = tt.precise_divf %lhs, %rhs : f32
  tt.return %0 : f32
}
