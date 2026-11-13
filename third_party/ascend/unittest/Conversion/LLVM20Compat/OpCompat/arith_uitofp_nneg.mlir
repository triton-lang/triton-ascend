// RUN: triton-opt %s | FileCheck %s
// Input: generic + {nonNeg}. Output: LLVM 20 pretty, no `nneg`.
// CHECK-LABEL: func.func @uitofp_nneg
// CHECK: %{{.*}} = arith.uitofp %{{.*}} : i32 to f32
// CHECK: return %{{.*}} : f32
// CHECK-NOT: nneg
// CHECK-NOT: nonNeg

func.func @uitofp_nneg(%a: i32) -> f32 {
  %0 = "arith.uitofp"(%a) {nonNeg} : (i32) -> f32
  return %0 : f32
}
