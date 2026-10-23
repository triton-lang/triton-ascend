// RUN: triton-opt %s | FileCheck %s
// Input: generic + {isExact} (23 memory). Output: LLVM 20 pretty, no `exact`.
// CHECK-LABEL: func.func @divsi
// CHECK: %{{.*}} = arith.divsi %{{.*}}, %{{.*}} : i32
// CHECK: return %{{.*}} : i32
// CHECK-NOT: exact
// CHECK-NOT: isExact

func.func @divsi(%a: i32, %b: i32) -> i32 {
  %0 = "arith.divsi"(%a, %b) {isExact} : (i32, i32) -> i32
  return %0 : i32
}
