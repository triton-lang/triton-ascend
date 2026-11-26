// RUN: triton-opt %s -o %t
// RUN: triton-opt %t --canonicalize | FileCheck %s

// Preserve conversions involving HiFloat8 when canonicalizing constants.
// CHECK-LABEL: func.func @hifloat8_casts
// CHECK-DAG: %[[FP32:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-DAG: %[[BITS:.*]] = arith.constant dense<0> : tensor<4xi8>
// CHECK: %[[NARROW:.*]] = arith.truncf %[[FP32]] : tensor<4xf32> to tensor<4xf8HIF>
// CHECK-NEXT: %[[WIDE:.*]] = arith.extf %[[NARROW]] : tensor<4xf8HIF> to tensor<4xf32>
// CHECK-NEXT: %[[VALUE:.*]] = arith.bitcast %[[BITS]] : tensor<4xi8> to tensor<4xf8HIF>
// CHECK-NEXT: return %[[WIDE]], %[[VALUE]] : tensor<4xf32>, tensor<4xf8HIF>
func.func @hifloat8_casts() -> (tensor<4xf32>, tensor<4xf8HIF>) {
  %f32 = arith.constant dense<1.0> : tensor<4xf32>
  %bits = arith.constant dense<0> : tensor<4xi8>
  %narrow = arith.truncf %f32 : tensor<4xf32> to tensor<4xf8HIF>
  %wide = arith.extf %narrow : tensor<4xf8HIF> to tensor<4xf32>
  %value = arith.bitcast %bits : tensor<4xi8> to tensor<4xf8HIF>
  return %wide, %value : tensor<4xf32>, tensor<4xf8HIF>
}

// CHECK-LABEL: func.func @fp16_cast_folds
// CHECK-NEXT: %[[FP16:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf16>
// CHECK-NEXT: return %[[FP16]] : tensor<4xf16>
func.func @fp16_cast_folds() -> tensor<4xf16> {
  %f32 = arith.constant dense<1.0> : tensor<4xf32>
  %narrow = arith.truncf %f32 : tensor<4xf32> to tensor<4xf16>
  return %narrow : tensor<4xf16>
}
