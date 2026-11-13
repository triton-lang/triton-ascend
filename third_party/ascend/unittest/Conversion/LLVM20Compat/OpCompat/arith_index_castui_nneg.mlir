// RUN: triton-opt %s | FileCheck %s
// Input: generic + {nonNeg}. Output: LLVM 20 pretty, no `nneg`.
// CHECK-LABEL: func.func @index_castui_nneg
// CHECK: %{{.*}} = arith.index_castui %{{.*}} : i32 to index
// CHECK: return %{{.*}} : index
// CHECK-NOT: nneg
// CHECK-NOT: nonNeg

func.func @index_castui_nneg(%a: i32) -> index {
  %0 = "arith.index_castui"(%a) {nonNeg} : (i32) -> index
  return %0 : index
}
