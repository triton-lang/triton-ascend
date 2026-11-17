// RUN: triton-opt --triton-to-hfusion --triton-to-linalg %s | FileCheck %s --implicit-check-not=arith.cmpf --implicit-check-not=arith.select

// CHECK-LABEL: func.func @saturate_f32_to_f8E4M3FN(
// CHECK-SAME: %[[INPUT:.*]]: f32
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[CAST]] : f8E4M3FN
tt.func @saturate_f32_to_f8E4M3FN(%input: f32) -> f8E4M3FN {
  %result = tt.fp_to_fp %input, rounding = rtne : f32 -> f8E4M3FN
  tt.return %result : f8E4M3FN
}

// CHECK-LABEL: func.func @saturate_f32_to_f8E5M2(
// CHECK-SAME: %[[INPUT:.*]]: f32
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E5M2
// CHECK: return %[[CAST]] : f8E5M2
tt.func @saturate_f32_to_f8E5M2(%input: f32) -> f8E5M2 {
  %result = tt.fp_to_fp %input, rounding = rtne : f32 -> f8E5M2
  tt.return %result : f8E5M2
}

// CHECK-LABEL: func.func @saturate_f16_to_f8E4M3FN(
// CHECK-SAME: %[[INPUT:.*]]: f16
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f16 to f8E4M3FN
// CHECK: return %[[CAST]] : f8E4M3FN
tt.func @saturate_f16_to_f8E4M3FN(%input: f16) -> f8E4M3FN {
  %result = tt.fp_to_fp %input, rounding = rtne : f16 -> f8E4M3FN
  tt.return %result : f8E4M3FN
}

// CHECK-LABEL: func.func @saturate_f16_to_f8E5M2(
// CHECK-SAME: %[[INPUT:.*]]: f16
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f16 to f8E5M2
// CHECK: return %[[CAST]] : f8E5M2
tt.func @saturate_f16_to_f8E5M2(%input: f16) -> f8E5M2 {
  %result = tt.fp_to_fp %input, rounding = rtne : f16 -> f8E5M2
  tt.return %result : f8E5M2
}

// CHECK-LABEL: func.func @saturate_bf16_to_f8E4M3FN(
// CHECK-SAME: %[[INPUT:.*]]: bf16
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : bf16 to f8E4M3FN
// CHECK: return %[[CAST]] : f8E4M3FN
tt.func @saturate_bf16_to_f8E4M3FN(%input: bf16) -> f8E4M3FN {
  %result = tt.fp_to_fp %input, rounding = rtne : bf16 -> f8E4M3FN
  tt.return %result : f8E4M3FN
}

// CHECK-LABEL: func.func @saturate_bf16_to_f8E5M2(
// CHECK-SAME: %[[INPUT:.*]]: bf16
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : bf16 to f8E5M2
// CHECK: return %[[CAST]] : f8E5M2
tt.func @saturate_bf16_to_f8E5M2(%input: bf16) -> f8E5M2 {
  %result = tt.fp_to_fp %input, rounding = rtne : bf16 -> f8E5M2
  tt.return %result : f8E5M2
}

// CHECK-LABEL: func.func @saturate_tensor_f32_to_f8E4M3FN(
// CHECK-SAME: %[[INPUT:.*]]: tensor<4x8xf32>
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4x8xf32> to tensor<4x8xf8E4M3FN>
// CHECK: return %[[CAST]] : tensor<4x8xf8E4M3FN>
func.func @saturate_tensor_f32_to_f8E4M3FN(%input: tensor<4x8xf32>) -> tensor<4x8xf8E4M3FN> {
  %result = tt.fp_to_fp %input, rounding = rtne : tensor<4x8xf32> -> tensor<4x8xf8E4M3FN>
  return %result : tensor<4x8xf8E4M3FN>
}

// CHECK-LABEL: func.func @saturate_tensor_f32_to_f8E5M2(
// CHECK-SAME: %[[INPUT:.*]]: tensor<4x8xf32>
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4x8xf32> to tensor<4x8xf8E5M2>
// CHECK: return %[[CAST]] : tensor<4x8xf8E5M2>
func.func @saturate_tensor_f32_to_f8E5M2(%input: tensor<4x8xf32>) -> tensor<4x8xf8E5M2> {
  %result = tt.fp_to_fp %input, rounding = rtne : tensor<4x8xf32> -> tensor<4x8xf8E5M2>
  return %result : tensor<4x8xf8E5M2>
}

// CHECK-LABEL: func.func @saturate_tensor_f16_to_f8E4M3FN(
// CHECK-SAME: %[[INPUT:.*]]: tensor<4x8xf16>
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4x8xf16> to tensor<4x8xf8E4M3FN>
// CHECK: return %[[CAST]] : tensor<4x8xf8E4M3FN>
func.func @saturate_tensor_f16_to_f8E4M3FN(%input: tensor<4x8xf16>) -> tensor<4x8xf8E4M3FN> {
  %result = tt.fp_to_fp %input, rounding = rtne : tensor<4x8xf16> -> tensor<4x8xf8E4M3FN>
  return %result : tensor<4x8xf8E4M3FN>
}

// CHECK-LABEL: func.func @saturate_tensor_f16_to_f8E5M2(
// CHECK-SAME: %[[INPUT:.*]]: tensor<4x8xf16>
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4x8xf16> to tensor<4x8xf8E5M2>
// CHECK: return %[[CAST]] : tensor<4x8xf8E5M2>
func.func @saturate_tensor_f16_to_f8E5M2(%input: tensor<4x8xf16>) -> tensor<4x8xf8E5M2> {
  %result = tt.fp_to_fp %input, rounding = rtne : tensor<4x8xf16> -> tensor<4x8xf8E5M2>
  return %result : tensor<4x8xf8E5M2>
}

// CHECK-LABEL: func.func @saturate_tensor_bf16_to_f8E4M3FN(
// CHECK-SAME: %[[INPUT:.*]]: tensor<4x8xbf16>
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4x8xbf16> to tensor<4x8xf8E4M3FN>
// CHECK: return %[[CAST]] : tensor<4x8xf8E4M3FN>
func.func @saturate_tensor_bf16_to_f8E4M3FN(%input: tensor<4x8xbf16>) -> tensor<4x8xf8E4M3FN> {
  %result = tt.fp_to_fp %input, rounding = rtne : tensor<4x8xbf16> -> tensor<4x8xf8E4M3FN>
  return %result : tensor<4x8xf8E4M3FN>
}

// CHECK-LABEL: func.func @saturate_tensor_bf16_to_f8E5M2(
// CHECK-SAME: %[[INPUT:.*]]: tensor<4x8xbf16>
// CHECK: %[[CAST:.*]] = arith.truncf %[[INPUT]] {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4x8xbf16> to tensor<4x8xf8E5M2>
// CHECK: return %[[CAST]] : tensor<4x8xf8E5M2>
func.func @saturate_tensor_bf16_to_f8E5M2(%input: tensor<4x8xbf16>) -> tensor<4x8xf8E5M2> {
  %result = tt.fp_to_fp %input, rounding = rtne : tensor<4x8xbf16> -> tensor<4x8xf8E5M2>
  return %result : tensor<4x8xf8E5M2>
}
