// RUN: triton-opt %s '--add_dynamic_cv_pipeline=compile-on-910-95=True' -verify-each | FileCheck %s --implicit-check-not=triton_ascend.dynamic_cv_pipeline.rc

// Two matmul results form two double-buffered Cube-to-Vector queues.
// CHECK: module attributes {{.*}}hivm.dynamic_plan = true
// CHECK-LABEL: func.func @tcb_pool_eligible(
// CHECK-DAG: hivm.tcore_type = #hivm.tcore_type<VECTOR>
// CHECK-DAG: hivm.tcore_type = #hivm.tcore_type<CUBE>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 2 : i32, ssbuffer.inter_core_buf_count = 2 : i32} {
  func.func @tcb_pool_eligible(%q: tensor<16x16xf16>, %k: tensor<16x16xf16>,
                             %v: tensor<16x16xf16>, %out: memref<16x16xf32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<16x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %init) -> tensor<16x16xf32> : i32 {
      %qk = linalg.matmul ins(%q, %k : tensor<16x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p32 = math.exp %qk : tensor<16x16xf32>
      %p = arith.truncf %p32 : tensor<16x16xf32> to tensor<16x16xf16>
      %pv = linalg.matmul ins(%p, %v : tensor<16x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %next = arith.addf %acc, %pv : tensor<16x16xf32>
      scf.yield %next : tensor<16x16xf32>
    }
    bufferization.materialize_in_destination %result in restrict writable %out
        : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
    return
  }
}
