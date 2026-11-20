// RUN: triton-opt %s | FileCheck %s
// Input: generic + {roundingmode}. Output: LLVM 20 pretty, no rounding keyword.
// CHECK-LABEL: func.func @subf_rounding
// CHECK: %{{.*}} = arith.subf %{{.*}}, %{{.*}} : f32
// CHECK: return %{{.*}} : f32
// CHECK-NOT: roundingmode
// CHECK-NOT: downward

func.func @subf_rounding(%a: f32, %b: f32) -> f32 {
  %0 = "arith.subf"(%a, %b) {roundingmode = 1 : i32} : (f32, f32) -> f32
  return %0 : f32
}
