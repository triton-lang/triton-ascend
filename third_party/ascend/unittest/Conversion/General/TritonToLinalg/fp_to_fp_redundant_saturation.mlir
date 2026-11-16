// RUN: triton-opt --triton-to-hfusion --triton-to-linalg --split-input-file %s | FileCheck %s

// Keep the original clamp, including its treatment of NaNs and signed zeros.
// CHECK-LABEL: func.func @scalar_clamped_e4m3
// CHECK: arith.minnumf
// CHECK: %[[CLAMP:.*]] = arith.maxnumf
// CHECK-NOT: arith.cmpf
// CHECK-NOT: arith.select
// CHECK: arith.truncf %[[CLAMP]] {{.*}} : f32 to f8E4M3FN
func.func @scalar_clamped_e4m3(%input: f32) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : f32
  %result = tt.fp_to_fp %clamp, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}

// -----

// The SwiGLU path reshapes an already-clamped tensor before the FP8 cast.
// The proof must also accept NaNs propagated by the existing clamp.
// CHECK-LABEL: func.func @reshape_clamped_e4m3
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK-NOT: arith.cmpf
// CHECK-NOT: arith.select
// CHECK: arith.truncf {{.*}} : tensor<1536xf32> to tensor<1536xf8E4M3FN>
func.func @reshape_clamped_e4m3(%input: tensor<48x32xf32>) -> tensor<1536xf8E4M3FN> {
  %lo_scalar = arith.constant -448.0 : f32
  %hi_scalar = arith.constant 448.0 : f32
  %lo = tt.splat %lo_scalar : f32 -> tensor<48x32xf32>
  %hi = tt.splat %hi_scalar : f32 -> tensor<48x32xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = all : tensor<48x32xf32>
  %reshape = tt.reshape %clamp : tensor<48x32xf32> -> tensor<6x256xf32>
  %flat = tt.reshape %reshape : tensor<6x256xf32> -> tensor<1536xf32>
  %result = tt.fp_to_fp %flat, rounding = rtne : tensor<1536xf32> -> tensor<1536xf8E4M3FN>
  return %result : tensor<1536xf8E4M3FN>
}

// -----

// Bounds are compared in the source type, using the destination format's limit.
// CHECK-LABEL: func.func @clamped_f16_e5m2
// CHECK: arith.minnumf
// CHECK: %[[CLAMP:.*]] = arith.maxnumf
// CHECK-NOT: arith.cmpf
// CHECK-NOT: arith.select
// CHECK: arith.truncf %[[CLAMP]] {{.*}} : tensor<4xf16> to tensor<4xf8E5M2>
func.func @clamped_f16_e5m2(%input: tensor<4xf16>) -> tensor<4xf8E5M2> {
  %lo = arith.constant dense<-57344.0> : tensor<4xf16>
  %hi = arith.constant dense<57344.0> : tensor<4xf16>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf16>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf16> -> tensor<4xf8E5M2>
  return %result : tensor<4xf8E5M2>
}

// -----

// A strictly narrower interval also makes the added saturation redundant.
// CHECK-LABEL: func.func @narrow_clamped_bf16_e4m3
// CHECK: arith.minnumf
// CHECK: %[[CLAMP:.*]] = arith.maxnumf
// CHECK-NOT: arith.cmpf
// CHECK-NOT: arith.select
// CHECK: arith.truncf %[[CLAMP]] {{.*}} : tensor<4xbf16> to tensor<4xf8E4M3FN>
func.func @narrow_clamped_bf16_e4m3(%input: tensor<4xbf16>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-224.0> : tensor<4xbf16>
  %hi = arith.constant dense<224.0> : tensor<4xbf16>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xbf16>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xbf16> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}

// -----

// Unknown bounds cannot establish a finite interval.
// CHECK-LABEL: func.func @dynamic_bound
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: arith.truncf
func.func @dynamic_bound(%input: tensor<4xf32>, %hi: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-448.0> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}

// -----

// A clamp for a wider interval cannot replace FP8 saturation.
// CHECK-LABEL: func.func @wide_upper_bound
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: arith.truncf
func.func @wide_upper_bound(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-448.0> : tensor<4xf32>
  %hi = arith.constant dense<480.0> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}

// -----

// Both ends of the interval must be covered by the range proof.
// CHECK-LABEL: func.func @wide_lower_bound
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: arith.truncf
func.func @wide_lower_bound(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-480.0> : tensor<4xf32>
  %hi = arith.constant dense<448.0> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}

// -----

// NaN bounds do not prove a range: unordered APFloat comparisons must not
// accidentally satisfy the bounds check. Clamp's NaN bounds are unspecified.
// CHECK-LABEL: func.func @nan_bound
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: arith.truncf
func.func @nan_bound(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-448.0> : tensor<4xf32>
  %hi = arith.constant dense<0x7FC00000> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}

// -----

// An infinite bound still permits an infinite value through the original clamp.
// CHECK-LABEL: func.func @infinite_bound
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: arith.truncf
func.func @infinite_bound(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<0xFF800000> : tensor<4xf32>
  %hi = arith.constant dense<448.0> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}

// -----

// Do not infer a tensor-wide bound from just one lane of a non-splat constant.
// CHECK-LABEL: func.func @non_splat_bound
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: arith.truncf
func.func @non_splat_bound(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-448.0> : tensor<4xf32>
  %hi = arith.constant dense<[448.0, 448.0, 480.0, 448.0]> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}

// -----

// Only value-preserving reshapes may be crossed; arithmetic can overflow again.
// CHECK-LABEL: func.func @arithmetic_after_clamp
// CHECK: arith.cmpf ogt
// CHECK: arith.select
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: arith.truncf
func.func @arithmetic_after_clamp(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-448.0> : tensor<4xf32>
  %hi = arith.constant dense<448.0> : tensor<4xf32>
  %two = arith.constant dense<2.0> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %scaled = arith.mulf %clamp, %two : tensor<4xf32>
  %result = tt.fp_to_fp %scaled, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}
