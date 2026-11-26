// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @shrsi
// CHECK: %{{.*}} = arith.shrsi %{{.*}}, %{{.*}} : i32
// CHECK: return %{{.*}} : i32
// CHECK-NOT: exact
// CHECK-NOT: isExact

func.func @shrsi(%a: i32, %b: i32) -> i32 {
  %0 = "arith.shrsi"(%a, %b) {isExact} : (i32, i32) -> i32
  return %0 : i32
}
