// RUN: triton-opt --triton-to-hfusion --triton-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @scalar_fp32_to_fp8_rtne
// CHECK: %[[CAST:.*]] = arith.truncf %{{.*}} {round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK: return %[[CAST]] : f8E4M3FN
module {
  tt.func @scalar_fp32_to_fp8_rtne(%arg0: f32) -> f8E4M3FN {
    %0 = tt.fp_to_fp %arg0, rounding = rtne : f32 -> f8E4M3FN
    tt.return %0 : f8E4M3FN
  }
}
