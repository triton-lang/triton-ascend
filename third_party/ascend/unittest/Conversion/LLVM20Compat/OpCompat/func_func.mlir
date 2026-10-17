// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @f(%{{.*}}: i32) -> i32
// CHECK: return %{{.*}} : i32
// CHECK-NOT: no_inline

func.func @f(%x: i32) -> i32 attributes {no_inline} {
  return %x : i32
}
