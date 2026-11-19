// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: gpu.module @m
// CHECK: gpu.func @barrier() kernel
// CHECK: gpu.barrier
// CHECK: gpu.return
// CHECK-NOT: memfence

gpu.module @m {
  gpu.func @barrier() kernel {
    "gpu.barrier"() {address_spaces = [#gpu.address_space<workgroup>]} : () -> ()
    gpu.return
  }
}
