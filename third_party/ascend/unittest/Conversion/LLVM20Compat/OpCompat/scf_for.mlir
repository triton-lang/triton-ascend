// RUN: triton-opt %s | FileCheck %s
// Input: 23 pretty `scf.for unsigned`. Output: LLVM 20 (no `unsigned` keyword).
// CHECK-LABEL: func.func @for_unsigned
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: return
// CHECK-NOT: scf.for unsigned
// CHECK-NOT: unsignedCmp

func.func @for_unsigned(%lb: index, %ub: index, %st: index) {
  scf.for unsigned %i = %lb to %ub step %st {
  }
  return
}
