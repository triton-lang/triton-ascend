// RUN: triton-opt --triton-to-hfusion --triton-to-linalg --split-input-file %s | FileCheck %s --implicit-check-not=arith.cmpf --implicit-check-not=arith.select

// Saturation belongs to the conversion, independently of its producer.
// Only the original select operations are permitted; no range guards are added.

// Keep the original clamp, including its treatment of NaNs and signed zeros.
// CHECK-LABEL: func.func @scalar_clamped_e4m3
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @scalar_clamped_e4m3(%input: f32) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : f32
  %result = tt.fp_to_fp %clamp, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// The SwiGLU clamp and reshape retain the same saturating conversion contract.
// CHECK-LABEL: func.func @reshape_clamped_e4m3
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: tensor.reshape
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<1536xf32> to tensor<1536xf8E4M3FN>
// CHECK: return %[[RESULT]] : tensor<1536xf8E4M3FN>
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

// The original f16 clamp is preserved before the E5M2 conversion.
// CHECK-LABEL: func.func @clamped_f16_e5m2
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4xf16> to tensor<4xf8E5M2>
// CHECK: return %[[RESULT]] : tensor<4xf8E5M2>
func.func @clamped_f16_e5m2(%input: tensor<4xf16>) -> tensor<4xf8E5M2> {
  %lo = arith.constant dense<-57344.0> : tensor<4xf16>
  %hi = arith.constant dense<57344.0> : tensor<4xf16>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf16>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf16> -> tensor<4xf8E5M2>
  return %result : tensor<4xf8E5M2>
}
// -----

// A narrower bf16 clamp also uses the same saturating conversion.
// CHECK-LABEL: func.func @narrow_clamped_bf16_e4m3
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4xbf16> to tensor<4xf8E4M3FN>
// CHECK: return %[[RESULT]] : tensor<4xf8E4M3FN>
func.func @narrow_clamped_bf16_e4m3(%input: tensor<4xbf16>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-224.0> : tensor<4xbf16>
  %hi = arith.constant dense<224.0> : tensor<4xbf16>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xbf16>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xbf16> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}
// -----

// Dynamic bounds do not require an extra software saturation sequence.
// CHECK-LABEL: func.func @dynamic_bound
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4xf32> to tensor<4xf8E4M3FN>
// CHECK: return %[[RESULT]] : tensor<4xf8E4M3FN>
func.func @dynamic_bound(%input: tensor<4xf32>, %hi: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-448.0> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}
// -----

// The conversion saturates values above the destination range.
// CHECK-LABEL: func.func @wide_upper_bound
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4xf32> to tensor<4xf8E4M3FN>
// CHECK: return %[[RESULT]] : tensor<4xf8E4M3FN>
func.func @wide_upper_bound(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-448.0> : tensor<4xf32>
  %hi = arith.constant dense<480.0> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}
// -----

// The conversion saturates values below the destination range.
// CHECK-LABEL: func.func @wide_lower_bound
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4xf32> to tensor<4xf8E4M3FN>
// CHECK: return %[[RESULT]] : tensor<4xf8E4M3FN>
func.func @wide_lower_bound(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-480.0> : tensor<4xf32>
  %hi = arith.constant dense<448.0> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}
// -----

// Preserve the original clamp with a NaN bound and the saturation contract.
// CHECK-LABEL: func.func @nan_bound
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4xf32> to tensor<4xf8E4M3FN>
// CHECK: return %[[RESULT]] : tensor<4xf8E4M3FN>
func.func @nan_bound(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-448.0> : tensor<4xf32>
  %hi = arith.constant dense<0x7FC00000> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}
// -----

// An infinite bound still permits infinity through the original clamp.
// CHECK-LABEL: func.func @infinite_bound
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4xf32> to tensor<4xf8E4M3FN>
// CHECK: return %[[RESULT]] : tensor<4xf8E4M3FN>
func.func @infinite_bound(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<0xFF800000> : tensor<4xf32>
  %hi = arith.constant dense<448.0> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}
// -----

// Different bounds in different lanes use the same saturating conversion.
// CHECK-LABEL: func.func @non_splat_bound
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4xf32> to tensor<4xf8E4M3FN>
// CHECK: return %[[RESULT]] : tensor<4xf8E4M3FN>
func.func @non_splat_bound(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-448.0> : tensor<4xf32>
  %hi = arith.constant dense<[448.0, 448.0, 480.0, 448.0]> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %result = tt.fp_to_fp %clamp, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}
// -----

// Multiplication can overflow the clamp range before the conversion.
// CHECK-LABEL: func.func @arithmetic_after_clamp
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: arith.mulf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4xf32> to tensor<4xf8E4M3FN>
// CHECK: return %[[RESULT]] : tensor<4xf8E4M3FN>
func.func @arithmetic_after_clamp(%input: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant dense<-448.0> : tensor<4xf32>
  %hi = arith.constant dense<448.0> : tensor<4xf32>
  %two = arith.constant dense<2.0> : tensor<4xf32>
  %clamp = tt.clampf %input, %lo, %hi, propagateNan = none : tensor<4xf32>
  %scaled = arith.mulf %clamp, %two : tensor<4xf32>
  %result = tt.fp_to_fp %scaled, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @minimumf_maximumf_bounded
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @minimumf_maximumf_bounded(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = arith.minimumf %x, %hi : f32
  %v = arith.maximumf %lo, %a : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @maximumf_minimumf_bounded
// CHECK: arith.maximumf
// CHECK: arith.minimumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @maximumf_minimumf_bounded(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = arith.maximumf %x, %lo : f32
  %v = arith.minimumf %hi, %a : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @minnumf_maxnumf_bounded
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @minnumf_maxnumf_bounded(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = arith.minnumf %x, %hi : f32
  %v = arith.maxnumf %lo, %a : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @maxnumf_minnumf_bounded
// CHECK: arith.maxnumf
// CHECK: arith.minnumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @maxnumf_minnumf_bounded(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = arith.maxnumf %x, %lo : f32
  %v = arith.minnumf %hi, %a : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @one_sided_minimumf
// CHECK: arith.minimumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @one_sided_minimumf(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %v = arith.minimumf %x, %hi : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @one_sided_maximumf
// CHECK: arith.maximumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @one_sided_maximumf(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %v = arith.maximumf %x, %lo : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @bounded_negf
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: arith.negf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @bounded_negf(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = tt.clampf %x, %lo, %hi, propagateNan = all : f32
  %v = arith.negf %a : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @bounded_absf
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: math.absf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @bounded_absf(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = tt.clampf %x, %lo, %hi, propagateNan = all : f32
  %v = math.absf %a : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @unbounded_abs
// CHECK: math.absf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @unbounded_abs(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %v = math.absf %x : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @abs_negative_interval
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: math.absf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @abs_negative_interval(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -224.0 : f32
  %hi = arith.constant -16.0 : f32
  %a = tt.clampf %x, %lo, %hi, propagateNan = all : f32
  %v = math.absf %a : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @abs_positive_interval
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: math.absf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @abs_positive_interval(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant 16.0 : f32
  %hi = arith.constant 224.0 : f32
  %a = tt.clampf %x, %lo, %hi, propagateNan = all : f32
  %v = math.absf %a : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the original select and saturate its result.
// CHECK-LABEL: func.func @bounded_select
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: arith.select
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @bounded_select(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = tt.clampf %x, %lo, %hi, propagateNan = all : f32
  %v = arith.select %cond, %a, %hi : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// An unbounded select arm has the same saturating conversion contract.
// CHECK-LABEL: func.func @unbounded_select
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: arith.select
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @unbounded_select(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = tt.clampf %x, %lo, %hi, propagateNan = all : f32
  %v = arith.select %cond, %a, %y : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @minnumf_nan_all
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: arith.minnumf
// CHECK: arith.maximumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @minnumf_nan_all(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = tt.clampf %x, %lo, %hi, propagateNan = all : f32
  %b = arith.minnumf %a, %y : f32
  %v = arith.maximumf %b, %lo : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @minnumf_nan_none
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: arith.minnumf
// CHECK: arith.maximumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @minnumf_nan_none(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = tt.clampf %x, %lo, %hi, propagateNan = none : f32
  %b = arith.minnumf %a, %y : f32
  %v = arith.maximumf %b, %lo : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @maxnumf_nan_all
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: arith.maxnumf
// CHECK: arith.minimumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @maxnumf_nan_all(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = tt.clampf %x, %lo, %hi, propagateNan = all : f32
  %b = arith.maxnumf %a, %y : f32
  %v = arith.minimumf %b, %hi : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @maxnumf_nan_none
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: arith.maxnumf
// CHECK: arith.minimumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @maxnumf_nan_none(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = tt.clampf %x, %lo, %hi, propagateNan = none : f32
  %b = arith.maxnumf %a, %y : f32
  %v = arith.minimumf %b, %hi : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Transpose, broadcast and reshape preserve the original producer chain.
// CHECK-LABEL: func.func @bounded_shape_chain
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: linalg.transpose
// CHECK: linalg.broadcast
// CHECK: tensor.reshape
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<64xf32> to tensor<64xf8E4M3FN>
// CHECK: return %[[RESULT]] : tensor<64xf8E4M3FN>
func.func @bounded_shape_chain(%x: tensor<4x8xf32>) -> tensor<64xf8E4M3FN> {
  %lo = arith.constant dense<-448.0> : tensor<4x8xf32>
  %hi = arith.constant dense<448.0> : tensor<4x8xf32>
  %a = tt.clampf %x, %lo, %hi, propagateNan = all : tensor<4x8xf32>
  %b = tt.trans %a {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  %c = tt.expand_dims %b {axis = 0 : i32} : tensor<8x4xf32> -> tensor<1x8x4xf32>
  %d = tt.broadcast %c : tensor<1x8x4xf32> -> tensor<2x8x4xf32>
  %v = tt.reshape %d : tensor<2x8x4xf32> -> tensor<64xf32>
  %result = tt.fp_to_fp %v, rounding = rtne : tensor<64xf32> -> tensor<64xf8E4M3FN>
  return %result : tensor<64xf8E4M3FN>
}
// -----

// A scalar clamp becomes a fill before the saturating tensor conversion.
// CHECK-LABEL: func.func @bounded_splat
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: linalg.fill
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : tensor<4xf32> to tensor<4xf8E4M3FN>
// CHECK: return %[[RESULT]] : tensor<4xf8E4M3FN>
func.func @bounded_splat(%x: f32) -> tensor<4xf8E4M3FN> {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = tt.clampf %x, %lo, %hi, propagateNan = none : f32
  %v = tt.splat %a : f32 -> tensor<4xf32>
  %result = tt.fp_to_fp %v, rounding = rtne : tensor<4xf32> -> tensor<4xf8E4M3FN>
  return %result : tensor<4xf8E4M3FN>
}
// -----

// Preserve the computed bounds and both clamps.
// CHECK-LABEL: func.func @computed_clamp_bounds
// CHECK: arith.minnumf
// CHECK: arith.maxnumf
// CHECK: arith.negf
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @computed_clamp_bounds(%x: f32, %y: f32, %cond: i1) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %zero = arith.constant 0.0 : f32
  %bound = tt.clampf %y, %zero, %hi, propagateNan = none : f32
  %lower = arith.negf %bound : f32
  %v = tt.clampf %x, %lower, %bound, propagateNan = all : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @minnumf_possible_snan
// CHECK: arith.minnumf
// CHECK: arith.minnumf
// CHECK: arith.maximumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @minnumf_possible_snan(%x: f32, %y: f32) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = arith.minnumf %x, %hi : f32
  %b = arith.minnumf %a, %y : f32
  %v = arith.maximumf %b, %lo : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// The existing sNaN minnum fold may simplify the producer; saturation remains.
// CHECK-LABEL: func.func @minnumf_constant_snan
// CHECK: arith.maximumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @minnumf_constant_snan(%x: f32, %y: f32) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %snan = arith.constant 0x7F800001 : f32
  %a = arith.minnumf %snan, %hi : f32
  %b = arith.minnumf %a, %y : f32
  %v = arith.maximumf %b, %lo : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// Preserve the producer operations and use the uniform saturation contract.
// CHECK-LABEL: func.func @maxnumf_possible_snan
// CHECK: arith.maxnumf
// CHECK: arith.maxnumf
// CHECK: arith.minimumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @maxnumf_possible_snan(%x: f32, %y: f32) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %a = arith.maxnumf %x, %lo : f32
  %b = arith.maxnumf %a, %y : f32
  %v = arith.minimumf %b, %hi : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
// -----

// The existing sNaN maxnum fold may simplify the producer; saturation remains.
// CHECK-LABEL: func.func @maxnumf_constant_snan
// CHECK: arith.minimumf
// CHECK: %[[RESULT:.*]] = arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[RESULT]] : f8E4M3FN
func.func @maxnumf_constant_snan(%x: f32, %y: f32) -> f8E4M3FN {
  %lo = arith.constant -448.0 : f32
  %hi = arith.constant 448.0 : f32
  %snan = arith.constant 0x7F800001 : f32
  %a = arith.maxnumf %snan, %lo : f32
  %b = arith.maxnumf %a, %y : f32
  %v = arith.minimumf %b, %hi : f32
  %result = tt.fp_to_fp %v, rounding = rtne : f32 -> f8E4M3FN
  return %result : f8E4M3FN
}
