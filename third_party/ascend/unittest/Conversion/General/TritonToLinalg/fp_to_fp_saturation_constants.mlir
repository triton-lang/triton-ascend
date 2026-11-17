// RUN: triton-opt --triton-to-hfusion --triton-to-linalg --canonicalize --split-input-file %s | FileCheck %s

// Constant folding must honor saturation. It may preserve the attributed cast
// or produce the signed finite limit, but must not fold infinity to FP8 infinity.
// CHECK-LABEL: func.func @positive_infinity_to_e5m2
// CHECK: %[[RESULT:.*]] = arith.{{(truncf .*enable_saturate = true.*round_mode = #hfusion.round_mode<rint>.* : f32 to f8E5M2|constant 5.734400e\+04 : f8E5M2)}}
// CHECK: return %[[RESULT]] : f8E5M2
func.func @positive_infinity_to_e5m2() -> f8E5M2 {
  %input = arith.constant 0x7F800000 : f32
  %result = tt.fp_to_fp %input, rounding = rtne : f32 -> f8E5M2
  return %result : f8E5M2
}

// -----

// CHECK-LABEL: func.func @negative_infinity_to_e5m2
// CHECK: %[[RESULT:.*]] = arith.{{(truncf .*enable_saturate = true.*round_mode = #hfusion.round_mode<rint>.* : f32 to f8E5M2|constant -5.734400e\+04 : f8E5M2)}}
// CHECK: return %[[RESULT]] : f8E5M2
func.func @negative_infinity_to_e5m2() -> f8E5M2 {
  %input = arith.constant 0xFF800000 : f32
  %result = tt.fp_to_fp %input, rounding = rtne : f32 -> f8E5M2
  return %result : f8E5M2
}

// -----

// A mixed constant must saturate both signs without changing either signed zero.
// CHECK-LABEL: func.func @mixed_dense_to_e5m2
// CHECK: %[[RESULT:.*]] = arith.{{(truncf .*enable_saturate = true.*round_mode = #hfusion.round_mode<rint>.* : tensor<4xf32> to tensor<4xf8E5M2>|constant dense<\[5.734400e\+04, -5.734400e\+04, 0.000000e\+00, -0.000000e\+00\]> : tensor<4xf8E5M2>)}}
// CHECK: return %[[RESULT]] : tensor<4xf8E5M2>
func.func @mixed_dense_to_e5m2() -> tensor<4xf8E5M2> {
  %input = arith.constant dense<[0x7F800000, 0xFF800000, 0.0, -0.0]> : tensor<4xf32>
  %result = tt.fp_to_fp %input, rounding = rtne : tensor<4xf32> -> tensor<4xf8E5M2>
  return %result : tensor<4xf8E5M2>
}

// -----

// Splat lowering may retain a tensor cast or fill with a saturated scalar.
// CHECK-LABEL: func.func @positive_infinity_splat_to_e5m2
// CHECK: arith.{{(truncf .*enable_saturate = true.*round_mode = #hfusion.round_mode<rint>.* to (tensor<4xf8E5M2>|f8E5M2)|constant (dense<5.734400e\+04> : tensor<4xf8E5M2>|5.734400e\+04 : f8E5M2))}}
// CHECK: return {{.*}} : tensor<4xf8E5M2>
func.func @positive_infinity_splat_to_e5m2() -> tensor<4xf8E5M2> {
  %input = arith.constant 0x7F800000 : f32
  %splat = tt.splat %input : f32 -> tensor<4xf32>
  %result = tt.fp_to_fp %splat, rounding = rtne : tensor<4xf32> -> tensor<4xf8E5M2>
  return %result : tensor<4xf8E5M2>
}

// -----

// A constant fill has the same contract as a splat.
// CHECK-LABEL: func.func @negative_infinity_fill_to_e5m2
// CHECK: arith.{{(truncf .*enable_saturate = true.*round_mode = #hfusion.round_mode<rint>.* to (tensor<4xf8E5M2>|f8E5M2)|constant (dense<-5.734400e\+04> : tensor<4xf8E5M2>|-5.734400e\+04 : f8E5M2))}}
// CHECK: return {{.*}} : tensor<4xf8E5M2>
func.func @negative_infinity_fill_to_e5m2() -> tensor<4xf8E5M2> {
  %input = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<4xf32>
  %fill = linalg.fill ins(%input : f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  %result = tt.fp_to_fp %fill, rounding = rtne : tensor<4xf32> -> tensor<4xf8E5M2>
  return %result : tensor<4xf8E5M2>
}
