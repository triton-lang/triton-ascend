// RUN: triton-opt %s | FileCheck %s
// Input: generic + {roundingmode}. Output: LLVM 20 pretty, no rounding keyword.
// CHECK-LABEL: func.func @divf_rounding
// CHECK: %{{.*}} = arith.divf %{{.*}}, %{{.*}} : f32
// CHECK: return %{{.*}} : f32
// CHECK-NOT: roundingmode
// CHECK-NOT: toward_zero

func.func @divf_rounding(%a: f32, %b: f32) -> f32 {
  %0 = "arith.divf"(%a, %b) {roundingmode = 3 : i32} : (f32, f32) -> f32
  return %0 : f32
}
