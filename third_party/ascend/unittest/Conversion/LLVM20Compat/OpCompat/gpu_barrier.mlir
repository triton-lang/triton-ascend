// RUN: triton-opt %s | FileCheck %s
// 3.8: oilist(named / memfence / scope). Print must stay LLVM 20: bare `gpu.barrier`.
// CHECK-LABEL: gpu.module @m
// CHECK: gpu.func @barrier() kernel
// CHECK: gpu.barrier
// CHECK: gpu.return
// CHECK-NOT: memfence
// CHECK-NOT: address_spaces
// CHECK-NOT: scope
// CHECK-NOT: named
// CHECK-NOT: barrier_scope

gpu.module @m {
  gpu.func @barrier() kernel {
    "gpu.barrier"() <{
      named = 1 : i64,
      scope = #gpu.barrier_scope<workgroup>,
      address_spaces = [#gpu.address_space<workgroup>]
    }> : () -> ()
    gpu.return
  }
}
