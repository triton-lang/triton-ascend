// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @inttoptr
// CHECK: %{{.*}} = llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
// CHECK: return %{{.*}} : !llvm.ptr
// CHECK-NOT: dereferenceable

func.func @inttoptr(%x: i64) -> !llvm.ptr {
  %0 = llvm.inttoptr %x : i64 to !llvm.ptr
  return %0 : !llvm.ptr
}
