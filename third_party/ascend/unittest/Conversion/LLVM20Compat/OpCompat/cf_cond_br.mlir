// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @cond_br
// CHECK: cf.cond_br %{{.*}}, ^{{.*}}(%{{.*}} : i32), ^{{.*}}(%{{.*}} : i32)
// CHECK: return %{{.*}} : i32
// CHECK-NOT: weights(

func.func @cond_br(%flag: i1, %a: i32, %b: i32) -> i32 {
  cf.cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32) {branch_weights = array<i32: 1, 2>}
^bb1(%x: i32):
  return %x : i32
}
