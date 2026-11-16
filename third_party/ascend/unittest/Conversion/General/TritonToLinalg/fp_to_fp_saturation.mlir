// RUN: triton-opt --triton-to-hfusion --triton-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @saturate_f32_to_f8E4M3FN(
// CHECK-SAME: %[[INPUT:.*]]: f32
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 4.480000e+02 : f32
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -4.480000e+02 : f32
// CHECK: %[[ABOVE:.*]] = arith.cmpf ogt, %[[INPUT]], %[[UPPER]] : f32
// CHECK: %[[HIGH:.*]] = arith.select %[[ABOVE]], %[[UPPER]], %[[INPUT]] : f32
// CHECK: %[[BELOW:.*]] = arith.cmpf olt, %[[HIGH]], %[[LOWER]] : f32
// CHECK: %[[CLAMPED:.*]] = arith.select %[[BELOW]], %[[LOWER]], %[[HIGH]] : f32
// CHECK: %[[CAST:.*]] = arith.truncf %[[CLAMPED]] {round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[CAST]] : f8E4M3FN
tt.func @saturate_f32_to_f8E4M3FN(%input: f32) -> f8E4M3FN {
  %result = tt.fp_to_fp %input, rounding = rtne : f32 -> f8E4M3FN
  tt.return %result : f8E4M3FN
}

// CHECK-LABEL: func.func @saturate_f32_to_f8E5M2(
// CHECK-SAME: %[[INPUT:.*]]: f32
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 5.734400e+04 : f32
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -5.734400e+04 : f32
// CHECK: %[[ABOVE:.*]] = arith.cmpf ogt, %[[INPUT]], %[[UPPER]] : f32
// CHECK: %[[HIGH:.*]] = arith.select %[[ABOVE]], %[[UPPER]], %[[INPUT]] : f32
// CHECK: %[[BELOW:.*]] = arith.cmpf olt, %[[HIGH]], %[[LOWER]] : f32
// CHECK: %[[CLAMPED:.*]] = arith.select %[[BELOW]], %[[LOWER]], %[[HIGH]] : f32
// CHECK: %[[CAST:.*]] = arith.truncf %[[CLAMPED]] {round_mode = #hfusion.round_mode<rint>} : f32 to f8E5M2
// CHECK: return %[[CAST]] : f8E5M2
tt.func @saturate_f32_to_f8E5M2(%input: f32) -> f8E5M2 {
  %result = tt.fp_to_fp %input, rounding = rtne : f32 -> f8E5M2
  tt.return %result : f8E5M2
}

// CHECK-LABEL: func.func @saturate_f16_to_f8E4M3FN(
// CHECK-SAME: %[[INPUT:.*]]: f16
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 4.480000e+02 : f16
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -4.480000e+02 : f16
// CHECK: %[[ABOVE:.*]] = arith.cmpf ogt, %[[INPUT]], %[[UPPER]] : f16
// CHECK: %[[HIGH:.*]] = arith.select %[[ABOVE]], %[[UPPER]], %[[INPUT]] : f16
// CHECK: %[[BELOW:.*]] = arith.cmpf olt, %[[HIGH]], %[[LOWER]] : f16
// CHECK: %[[CLAMPED:.*]] = arith.select %[[BELOW]], %[[LOWER]], %[[HIGH]] : f16
// CHECK: %[[CAST:.*]] = arith.truncf %[[CLAMPED]] {round_mode = #hfusion.round_mode<rint>} : f16 to f8E4M3FN
// CHECK: return %[[CAST]] : f8E4M3FN
tt.func @saturate_f16_to_f8E4M3FN(%input: f16) -> f8E4M3FN {
  %result = tt.fp_to_fp %input, rounding = rtne : f16 -> f8E4M3FN
  tt.return %result : f8E4M3FN
}

// CHECK-LABEL: func.func @saturate_f16_to_f8E5M2(
// CHECK-SAME: %[[INPUT:.*]]: f16
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 5.734400e+04 : f16
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -5.734400e+04 : f16
// CHECK: %[[ABOVE:.*]] = arith.cmpf ogt, %[[INPUT]], %[[UPPER]] : f16
// CHECK: %[[HIGH:.*]] = arith.select %[[ABOVE]], %[[UPPER]], %[[INPUT]] : f16
// CHECK: %[[BELOW:.*]] = arith.cmpf olt, %[[HIGH]], %[[LOWER]] : f16
// CHECK: %[[CLAMPED:.*]] = arith.select %[[BELOW]], %[[LOWER]], %[[HIGH]] : f16
// CHECK: %[[CAST:.*]] = arith.truncf %[[CLAMPED]] {round_mode = #hfusion.round_mode<rint>} : f16 to f8E5M2
// CHECK: return %[[CAST]] : f8E5M2
tt.func @saturate_f16_to_f8E5M2(%input: f16) -> f8E5M2 {
  %result = tt.fp_to_fp %input, rounding = rtne : f16 -> f8E5M2
  tt.return %result : f8E5M2
}

// CHECK-LABEL: func.func @saturate_bf16_to_f8E4M3FN(
// CHECK-SAME: %[[INPUT:.*]]: bf16
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 4.480000e+02 : bf16
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -4.480000e+02 : bf16
// CHECK: %[[ABOVE:.*]] = arith.cmpf ogt, %[[INPUT]], %[[UPPER]] : bf16
// CHECK: %[[HIGH:.*]] = arith.select %[[ABOVE]], %[[UPPER]], %[[INPUT]] : bf16
// CHECK: %[[BELOW:.*]] = arith.cmpf olt, %[[HIGH]], %[[LOWER]] : bf16
// CHECK: %[[CLAMPED:.*]] = arith.select %[[BELOW]], %[[LOWER]], %[[HIGH]] : bf16
// CHECK: %[[CAST:.*]] = arith.truncf %[[CLAMPED]] {round_mode = #hfusion.round_mode<rint>} : bf16 to f8E4M3FN
// CHECK: return %[[CAST]] : f8E4M3FN
tt.func @saturate_bf16_to_f8E4M3FN(%input: bf16) -> f8E4M3FN {
  %result = tt.fp_to_fp %input, rounding = rtne : bf16 -> f8E4M3FN
  tt.return %result : f8E4M3FN
}

// CHECK-LABEL: func.func @saturate_bf16_to_f8E5M2(
// CHECK-SAME: %[[INPUT:.*]]: bf16
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 5.734400e+04 : bf16
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -5.734400e+04 : bf16
// CHECK: %[[ABOVE:.*]] = arith.cmpf ogt, %[[INPUT]], %[[UPPER]] : bf16
// CHECK: %[[HIGH:.*]] = arith.select %[[ABOVE]], %[[UPPER]], %[[INPUT]] : bf16
// CHECK: %[[BELOW:.*]] = arith.cmpf olt, %[[HIGH]], %[[LOWER]] : bf16
// CHECK: %[[CLAMPED:.*]] = arith.select %[[BELOW]], %[[LOWER]], %[[HIGH]] : bf16
// CHECK: %[[CAST:.*]] = arith.truncf %[[CLAMPED]] {round_mode = #hfusion.round_mode<rint>} : bf16 to f8E5M2
// CHECK: return %[[CAST]] : f8E5M2
tt.func @saturate_bf16_to_f8E5M2(%input: bf16) -> f8E5M2 {
  %result = tt.fp_to_fp %input, rounding = rtne : bf16 -> f8E5M2
  tt.return %result : f8E5M2
}
