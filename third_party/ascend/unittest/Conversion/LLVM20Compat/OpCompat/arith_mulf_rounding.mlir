// RUN: triton-opt %s | FileCheck %s
// Input: generic + {roundingmode}. Output: LLVM 20 pretty, no rounding keyword.
// CHECK-LABEL: func.func @mulf_rounding
// CHECK: %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK: return %{{.*}} : f32
// CHECK-NOT: roundingmode
// CHECK-NOT: upward

func.func @mulf_rounding(%a: f32, %b: f32) -> f32 {
  %0 = "arith.mulf"(%a, %b) {roundingmode = 2 : i32} : (f32, f32) -> f32
  return %0 : f32
}
