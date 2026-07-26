// RUN: triton-opt --triton-to-linalg="named-ops=True" %s | FileCheck %s

// CHECK-LABEL: func.func @cos_even_range_reduction(
// CHECK-SAME: %[[ARG:.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT: %[[ABS:.*]] = math.absf %[[ARG]] : tensor<2xf32>
// CHECK-NEXT: %[[COS:.*]] = math.cos %[[ABS]] : tensor<2xf32>
// CHECK-NEXT: return %[[COS]] : tensor<2xf32>
func.func @cos_even_range_reduction(%arg: tensor<2xf32>) -> tensor<2xf32> {
  %cos = math.cos %arg : tensor<2xf32>
  return %cos : tensor<2xf32>
}

// CHECK-LABEL: func.func @cos_even_range_reduction_fastmath(
// CHECK-SAME: %[[FAST_ARG:.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT: %[[FAST_ABS:.*]] = math.absf %[[FAST_ARG]] fastmath<fast> : tensor<2xf32>
// CHECK-NEXT: %[[FAST_COS:.*]] = math.cos %[[FAST_ABS]] fastmath<fast> : tensor<2xf32>
// CHECK-NEXT: return %[[FAST_COS]] : tensor<2xf32>
func.func @cos_even_range_reduction_fastmath(%arg: tensor<2xf32>) -> tensor<2xf32> {
  %cos = math.cos %arg fastmath<fast> : tensor<2xf32>
  return %cos : tensor<2xf32>
}

// CHECK-LABEL: func.func @cos_even_range_reduction_already_abs(
// CHECK-SAME: %[[ABS_ARG:.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT: %[[EXISTING_ABS:.*]] = math.absf %[[ABS_ARG]] : tensor<2xf32>
// CHECK-NEXT: %[[ABS_COS:.*]] = math.cos %[[EXISTING_ABS]] : tensor<2xf32>
// CHECK-NEXT: return %[[ABS_COS]] : tensor<2xf32>
func.func @cos_even_range_reduction_already_abs(%arg: tensor<2xf32>) -> tensor<2xf32> {
  %abs = math.absf %arg : tensor<2xf32>
  %cos = math.cos %abs : tensor<2xf32>
  return %cos : tensor<2xf32>
}

// A greedy rewrite would loop after folding abs(constant) while leaving the
// f16 cos unfolded. This case also checks that the folded input is positive.
// CHECK-LABEL: func.func @cos_even_range_reduction_f16_constant() -> tensor<2xf16> {
// CHECK-NEXT: %[[F16_ONE:.*]] = arith.constant 1.000000e+00 : f16
// CHECK-NEXT: %[[F16_EMPTY:.*]] = tensor.empty() : tensor<2xf16>
// CHECK-NEXT: %[[F16_INPUT:.*]] = linalg.fill
// CHECK-SAME: ins(%[[F16_ONE]] : f16)
// CHECK-SAME: outs(%[[F16_EMPTY]] : tensor<2xf16>) -> tensor<2xf16>
// CHECK-NEXT: %[[F16_COS:.*]] = math.cos %[[F16_INPUT]] : tensor<2xf16>
// CHECK-NEXT: return %[[F16_COS]] : tensor<2xf16>
func.func @cos_even_range_reduction_f16_constant() -> tensor<2xf16> {
  %cst = arith.constant dense<[-1.0, 1.0]> : tensor<2xf16>
  %cos = math.cos %cst : tensor<2xf16>
  return %cos : tensor<2xf16>
}
