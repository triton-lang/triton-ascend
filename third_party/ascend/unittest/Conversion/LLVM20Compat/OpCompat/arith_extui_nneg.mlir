// RUN: triton-opt %s | FileCheck %s
// Input: generic + {nonNeg}. Output: LLVM 20 pretty, no `nneg`.
// CHECK-LABEL: func.func @extui_nneg
// CHECK: %{{.*}} = arith.extui %{{.*}} : i8 to i32
// CHECK: return %{{.*}} : i32
// CHECK-NOT: nneg
// CHECK-NOT: nonNeg

func.func @extui_nneg(%a: i8) -> i32 {
  %0 = "arith.extui"(%a) {nonNeg} : (i8) -> i32
  return %0 : i32
}
