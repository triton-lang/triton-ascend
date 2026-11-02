// RUN: triton-opt %s | FileCheck %s
// LLVM 22 has no isExact property; no compatibility patch is needed here.
// CHECK-LABEL: func.func @divsi
// CHECK: %{{.*}} = arith.divsi %{{.*}}, %{{.*}} : i32
// CHECK: return %{{.*}} : i32
// CHECK-NOT: exact
// CHECK-NOT: isExact

func.func @divsi(%a: i32, %b: i32) -> i32 {
  %0 = arith.divsi %a, %b : i32
  return %0 : i32
}
