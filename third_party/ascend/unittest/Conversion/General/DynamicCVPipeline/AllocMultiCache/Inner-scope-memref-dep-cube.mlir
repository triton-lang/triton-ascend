// RUN: triton-opt --add_multi_buffer_inner_scope %s 2>&1 | FileCheck %s
// Pass signals fallback via triton_ascend.dynamic_cv_pipeline.rc = 1
// (ERRCODE_FAILED); the IR is otherwise unchanged.

// CHECK-LABEL: module attributes
// CHECK-SAME: triton_ascend.dynamic_cv_pipeline.rc = 1

// Same shape as Inner-scope-memref-dep.mlir, in a CUBE scope. The scope is not
// multi-buffered, but its main loop still goes through CloneOps and the stage
// split, so a cross-block memref dependency must fall back here as well.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_memref_dep_fallback_cube() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32} : memref<128xf32>
        %tensor_from_alloc = bufferization.to_tensor %alloc {ssbuffer.block_id = 10 : i32} : memref<128xf32> to tensor<128xf32>
        %consumed = arith.addf %tensor_from_alloc, %tensor_from_alloc {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
        %new_prod = arith.addf %consumed, %arg {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
