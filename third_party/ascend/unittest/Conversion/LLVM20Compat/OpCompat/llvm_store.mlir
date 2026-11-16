// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @store
// CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr
// CHECK: return
// CHECK-NOT: invariant_group

func.func @store(%v: i32, %p: !llvm.ptr) {
  "llvm.store"(%v, %p) {invariantGroup} : (i32, !llvm.ptr) -> ()
  return
}
