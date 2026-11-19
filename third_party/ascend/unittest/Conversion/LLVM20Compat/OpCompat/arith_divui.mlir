// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @divui
// CHECK: %{{.*}} = arith.divui %{{.*}}, %{{.*}} : i32
// CHECK: return %{{.*}} : i32
// CHECK-NOT: exact
// CHECK-NOT: isExact

func.func @divui(%a: i32, %b: i32) -> i32 {
<<<<<<< HEAD
  %0 = "arith.divui"(%a, %b) {isExact} : (i32, i32) -> i32
=======
  %0 = arith.divui %a, %b : i32
>>>>>>> upstream/main
  return %0 : i32
}
