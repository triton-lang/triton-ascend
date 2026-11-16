// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @callee
// CHECK-LABEL: func.func @call
// CHECK: %{{.*}} = call @callee(%{{.*}}) : (i32) -> i32
// CHECK: return %{{.*}} : i32
// CHECK-NOT: no_inline

func.func @callee(%x: i32) -> i32 {
  return %x : i32
}

func.func @call(%x: i32) -> i32 {
  %y = func.call @callee(%x) {no_inline} : (i32) -> i32
  return %y : i32
}
