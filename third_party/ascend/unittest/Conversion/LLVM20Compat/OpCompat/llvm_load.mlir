// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @load
// CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK: return %{{.*}} : i32
// CHECK-NOT: invariant_group
// CHECK-NOT: dereferenceable

func.func @load(%p: !llvm.ptr) -> i32 {
  %0 = "llvm.load"(%p) {invariantGroup} : (!llvm.ptr) -> i32
  return %0 : i32
}
