// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @trunci
// CHECK: %{{.*}} = arith.trunci %{{.*}} : i32 to i16
// CHECK: return %{{.*}} : i16
// CHECK-NOT: overflow<

func.func @trunci(%a: i32) -> i16 {
  %0 = "arith.trunci"(%a) {overflowFlags = #arith.overflow<nsw>} : (i32) -> i16
  return %0 : i16
}
