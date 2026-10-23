// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @divui
// CHECK: %{{.*}} = arith.divui %{{.*}}, %{{.*}} : i32
// CHECK: return %{{.*}} : i32
// CHECK-NOT: exact
// CHECK-NOT: isExact

func.func @divui(%a: i32, %b: i32) -> i32 {
  %0 = "arith.divui"(%a, %b) {isExact} : (i32, i32) -> i32
  return %0 : i32
}
