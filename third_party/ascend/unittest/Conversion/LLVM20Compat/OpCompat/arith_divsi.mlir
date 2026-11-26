// RUN: triton-opt %s | FileCheck %s
<<<<<<< HEAD
// Input: generic + {isExact} (23 memory). Output: LLVM 20 pretty, no `exact`.
=======
// LLVM 22 has no isExact property; no compatibility patch is needed here.
>>>>>>> upstream/main
// CHECK-LABEL: func.func @divsi
// CHECK: %{{.*}} = arith.divsi %{{.*}}, %{{.*}} : i32
// CHECK: return %{{.*}} : i32
// CHECK-NOT: exact
// CHECK-NOT: isExact

func.func @divsi(%a: i32, %b: i32) -> i32 {
<<<<<<< HEAD
  %0 = "arith.divsi"(%a, %b) {isExact} : (i32, i32) -> i32
=======
  %0 = arith.divsi %a, %b : i32
>>>>>>> upstream/main
  return %0 : i32
}
