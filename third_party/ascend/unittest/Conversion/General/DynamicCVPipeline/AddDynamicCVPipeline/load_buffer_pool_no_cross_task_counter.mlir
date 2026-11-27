// RUN: triton-opt %s '--add_dynamic_cv_pipeline=compile-on-910-95=True' -verify-each | FileCheck %s --implicit-check-not=hivm.load_buffer_pool

// The runtime bias puts the startup check in the Vector scope. Neither Cube
// matmul guard tests the other task's counter, so only dynamic_plan applies.
// CHECK: hivm.dynamic_plan = true
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
  func.func @load_buffer_pool_no_cross_task_counter(
      %q: tensor<16x16xf16>, %kv: memref<?x16xf16>,
      %bias: memref<?xf32>, %out: memref<16x16xf32>) {
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<16x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %result = scf.for %i = %c0 to %c64 step %c16 iter_args(%acc = %init) -> tensor<16x16xf32> : i32 {
      %offset = arith.index_cast %i : i32 to index
      %src = memref.subview %kv[%offset, 0] [16, 16] [1, 1]
          : memref<?x16xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
      %buf = memref.alloc() : memref<16x16xf16>
      memref.copy %src, %buf
          : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
      %tile = bufferization.to_tensor %buf restrict writable
          : memref<16x16xf16> to tensor<16x16xf16>
      %bias_src = memref.subview %bias[%offset] [16] [1]
          : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %bias_buf = memref.alloc() : memref<16xf32>
      memref.copy %bias_src, %bias_buf
          : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32>
      %bias_tile = bufferization.to_tensor %bias_buf restrict writable
          : memref<16xf32> to tensor<16xf32>
      %bias_matrix = linalg.broadcast ins(%bias_tile : tensor<16xf32>)
          outs(%empty : tensor<16x16xf32>) dimensions = [0]
      %qk = linalg.matmul ins(%q, %tile : tensor<16x16xf16>, tensor<16x16xf16>)
          outs(%bias_matrix : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p32 = math.exp %qk : tensor<16x16xf32>
      %p = arith.truncf %p32 : tensor<16x16xf32> to tensor<16x16xf16>
      %pv = linalg.matmul ins(%p, %tile : tensor<16x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %next = arith.addf %acc, %pv : tensor<16x16xf32>
      scf.yield %next : tensor<16x16xf32>
    }
    bufferization.materialize_in_destination %result in restrict writable %out
        : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
    return
  }
}
