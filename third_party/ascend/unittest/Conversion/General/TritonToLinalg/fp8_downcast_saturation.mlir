// RUN: triton-opt --triton-to-hfusion --triton-to-linalg %s | FileCheck %s

// Ordered comparisons must preserve NaNs, while infinities saturate to the
// destination's finite range. Cover scalar/tensor inputs and all source types.

// CHECK-LABEL: func.func @scalar_f32_to_e4m3
// CHECK-DAG: %[[HI:.*]] = arith.constant 4.480000e+02 : f32
// CHECK-DAG: %[[LO:.*]] = arith.constant -4.480000e+02 : f32
// CHECK: %[[GT:.*]] = arith.cmpf ogt, %arg0, %[[HI]] : f32
// CHECK: %[[UPPER:.*]] = arith.select %[[GT]], %[[HI]], %arg0 : f32
// CHECK: %[[LT:.*]] = arith.cmpf olt, %[[UPPER]], %[[LO]] : f32
// CHECK: %[[CLAMPED:.*]] = arith.select %[[LT]], %[[LO]], %[[UPPER]] : f32
// CHECK: arith.truncf %[[CLAMPED]] {round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
func.func @scalar_f32_to_e4m3(%arg0: f32) -> f8E4M3FN {
  %0 = tt.fp_to_fp %arg0, rounding = rtne : f32 -> f8E4M3FN
  return %0 : f8E4M3FN
}

// CHECK-LABEL: func.func @tensor_f16_to_e4m3
// CHECK-DAG: %[[HI_CONST:.*]] = arith.constant 4.480000e+02 : f16
// CHECK-DAG: %[[LO_CONST:.*]] = arith.constant -4.480000e+02 : f16
// CHECK-DAG: %[[HI:.*]] = linalg.fill ins(%[[HI_CONST]] : f16)
// CHECK-DAG: %[[LO:.*]] = linalg.fill ins(%[[LO_CONST]] : f16)
// CHECK: %[[GT:.*]] = arith.cmpf ogt, %arg0, %[[HI]] : tensor<8xf16>
// CHECK: %[[UPPER:.*]] = arith.select %[[GT]], %[[HI]], %arg0 : tensor<8xi1>, tensor<8xf16>
// CHECK: %[[LT:.*]] = arith.cmpf olt, %[[UPPER]], %[[LO]] : tensor<8xf16>
// CHECK: %[[CLAMPED:.*]] = arith.select %[[LT]], %[[LO]], %[[UPPER]] : tensor<8xi1>, tensor<8xf16>
// CHECK: arith.truncf %[[CLAMPED]] {round_mode = #hfusion.round_mode<rint>} : tensor<8xf16> to tensor<8xf8E4M3FN>
func.func @tensor_f16_to_e4m3(%arg0: tensor<8xf16>) -> tensor<8xf8E4M3FN> {
  %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8xf16> -> tensor<8xf8E4M3FN>
  return %0 : tensor<8xf8E4M3FN>
}

// CHECK-LABEL: func.func @tensor_bf16_to_e5m2
// CHECK-DAG: %[[HI_CONST:.*]] = arith.constant 5.734400e+04 : bf16
// CHECK-DAG: %[[LO_CONST:.*]] = arith.constant -5.734400e+04 : bf16
// CHECK-DAG: %[[HI:.*]] = linalg.fill ins(%[[HI_CONST]] : bf16)
// CHECK-DAG: %[[LO:.*]] = linalg.fill ins(%[[LO_CONST]] : bf16)
// CHECK: %[[GT:.*]] = arith.cmpf ogt, %arg0, %[[HI]] : tensor<8xbf16>
// CHECK: %[[UPPER:.*]] = arith.select %[[GT]], %[[HI]], %arg0 : tensor<8xi1>, tensor<8xbf16>
// CHECK: %[[LT:.*]] = arith.cmpf olt, %[[UPPER]], %[[LO]] : tensor<8xbf16>
// CHECK: %[[CLAMPED:.*]] = arith.select %[[LT]], %[[LO]], %[[UPPER]] : tensor<8xi1>, tensor<8xbf16>
// CHECK: arith.truncf %[[CLAMPED]] {round_mode = #hfusion.round_mode<rint>} : tensor<8xbf16> to tensor<8xf8E5M2>
func.func @tensor_bf16_to_e5m2(%arg0: tensor<8xbf16>) -> tensor<8xf8E5M2> {
  %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8xbf16> -> tensor<8xf8E5M2>
  return %0 : tensor<8xf8E5M2>
}

// CHECK-LABEL: func.func @non_fp8_downcast
// CHECK-NOT: arith.cmpf
// CHECK-NOT: arith.select
// CHECK: arith.truncf %arg0 {round_mode = #hfusion.round_mode<rint>} : f32 to f16
func.func @non_fp8_downcast(%arg0: f32) -> f16 {
  %0 = tt.fp_to_fp %arg0, rounding = rtne : f32 -> f16
  return %0 : f16
}
