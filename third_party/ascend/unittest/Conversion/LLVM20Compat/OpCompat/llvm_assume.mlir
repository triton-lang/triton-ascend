// RUN: triton-opt %s | FileCheck %s
// Input needs empty op_bundle_sizes (LLVM 23 property). Print must drop it.
// CHECK-LABEL: func.func @assume
// CHECK: "llvm.intr.assume"(%{{.*}}) : (i1) -> ()
// CHECK: return
// CHECK-NOT: op_bundle_sizes

func.func @assume(%c: i1) {
  "llvm.intr.assume"(%c) <{op_bundle_sizes = array<i32>}> : (i1) -> ()
  return
}
