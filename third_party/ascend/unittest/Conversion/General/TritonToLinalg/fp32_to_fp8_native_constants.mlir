// RUN: triton-opt --triton-to-hfusion --triton-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @infinity_to_e5m2
// CHECK: %[[VALUE:.*]] = arith.constant -5.734400e+04 : f8E5M2
// CHECK: return %[[VALUE]]
func.func @infinity_to_e5m2() -> f8E5M2 {
  %c = arith.constant 0xFF800000 : f32
  %0 = tt.fp_to_fp %c, rounding = rtne : f32 -> f8E5M2
  return %0 : f8E5M2
}

// The input includes overflow, a NaN, signed zero, and two halfway values.
// CHECK-LABEL: func.func @dense_to_e4m3
// CHECK: %[[VALUE:.*]] = arith.constant dense<[4.480000e+02, -4.480000e+02, 0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e+00]> : tensor<6xf8E4M3FN>
// CHECK: return %[[VALUE]]
func.func @dense_to_e4m3() -> tensor<6xf8E4M3FN> {
  %c = arith.constant dense<[1.000000e+10, 0xFF800000, 0x7FC00000, -0.0, 1.0625, 1.1875]> : tensor<6xf32>
  %0 = tt.fp_to_fp %c, rounding = rtne : tensor<6xf32> -> tensor<6xf8E4M3FN>
  return %0 : tensor<6xf8E4M3FN>
}

// CHECK-LABEL: func.func @splat_to_e5m2
// CHECK: arith.constant 5.734400e+04 : f8E5M2
func.func @splat_to_e5m2() -> tensor<4xf8E5M2> {
  %c = arith.constant dense<0x7F800000> : tensor<4xf32>
  %0 = tt.fp_to_fp %c, rounding = rtne : tensor<4xf32> -> tensor<4xf8E5M2>
  return %0 : tensor<4xf8E5M2>
}
