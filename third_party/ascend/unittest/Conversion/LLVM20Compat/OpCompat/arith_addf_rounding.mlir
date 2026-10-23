// RUN: triton-opt %s | FileCheck %s
// Input: generic + {roundingmode}. Output: LLVM 20 pretty, no rounding keyword.
// CHECK-LABEL: func.func @addf_rounding
// CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK: return %{{.*}} : f32
// CHECK-NOT: roundingmode
// CHECK-NOT: to_nearest_even

func.func @addf_rounding(%a: f32, %b: f32) -> f32 {
  %0 = "arith.addf"(%a, %b) {roundingmode = 0 : i32} : (f32, f32) -> f32
  return %0 : f32
}
