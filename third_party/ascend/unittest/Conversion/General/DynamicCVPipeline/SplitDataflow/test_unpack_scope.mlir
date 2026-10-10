// RUN: triton-opt --ssbuf-unpack-scopeop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

  // ============================================================================
  // 1. test_unpack_custom_op_scope
  // Unpack scope containing custom op and buffer operations (alloc, to_tensor).
  // Verifies that:
  // - scope.scope is removed
  // - body ops are unpacked to parent block preserving order
  // - attributes of scope.scope (block_id=12, core_type="VECTOR") are propagated
  //   to all unpacked operations (overwriting previous internal block_id=15)
  // - downstream consumer directly consumes the result of hivm.hir.custom
  // ============================================================================
  // CHECK-LABEL: func.func @test_unpack_custom_op_scope
  // CHECK-NOT: scope.scope
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : memref<2048xf32>
  // CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : memref<2048xf32> to tensor<2048xf32>
  // CHECK: %[[CUSTOM:.*]] = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR", symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%[[TENSOR]] : tensor<2048xf32>) -> tensor<2048xf32>
  // CHECK: %[[EXP:.*]] = tensor.expand_shape %[[CUSTOM]] {{\[\[}}0, 1{{\]\]}} output_shape [64, 32] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2048xf32> into tensor<64x32xf32>
  // CHECK: return %[[EXP]] : tensor<64x32xf32>
  func.func @test_unpack_custom_op_scope(%arg0: tensor<2048xf32>) -> tensor<64x32xf32> {
    %scope_res = scope.scope : () -> tensor<2048xf32> {
      %alloc = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : memref<2048xf32>
      %tensor = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : memref<2048xf32> to tensor<2048xf32>
      %custom = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR", symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%tensor : tensor<2048xf32>) -> tensor<2048xf32>
      scope.return %custom : tensor<2048xf32>
    } {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %expanded = tensor.expand_shape %scope_res [[0, 1]] output_shape [64, 32] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2048xf32> into tensor<64x32xf32>
    return %expanded : tensor<64x32xf32>
  }

  // ============================================================================
  // 2. test_unpack_scope_multiple_results
  // Unpack scope returning multiple values. Verifies all return values are re-routed.
  // ============================================================================
  // CHECK-LABEL: func.func @test_unpack_scope_multiple_results
  // CHECK-NOT: scope.scope
  // CHECK: %[[A:.*]] = arith.addf %arg0, %arg1 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
  // CHECK: %[[B:.*]] = arith.mulf %arg0, %arg1 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
  // CHECK: return %[[A]], %[[B]] : tensor<32xf32>, tensor<32xf32>
  func.func @test_unpack_scope_multiple_results(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> (tensor<32xf32>, tensor<32xf32>) {
    %res:2 = scope.scope : () -> (tensor<32xf32>, tensor<32xf32>) {
      %a = arith.addf %arg0, %arg1 : tensor<32xf32>
      %b = arith.mulf %arg0, %arg1 : tensor<32xf32>
      scope.return %a, %b : tensor<32xf32>, tensor<32xf32>
    } {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"}
    return %res#0, %res#1 : tensor<32xf32>, tensor<32xf32>
  }

  // ============================================================================
  // 3. test_unpack_scope_inside_loop
  // Unpack scope inside scf.for loop body (matching flash_fwd_kernel in passes.txt).
  // ============================================================================
  // CHECK-LABEL: func.func @test_unpack_scope_inside_loop
  // CHECK: scf.for
  // CHECK-NOT: scope.scope
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<64xf32>
  // CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<64xf32> to tensor<64xf32>
  // CHECK: %[[CUSTOM:.*]] = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", symbol = "tanh_fp32"} "tanh_fp32" ins(%{{.*}} : tensor<64xf32>) outs(%[[TENSOR]] : tensor<64xf32>) -> tensor<64xf32>
  // CHECK: scf.yield %[[CUSTOM]] : tensor<64xf32>
  func.func @test_unpack_scope_inside_loop(%arg0: tensor<64xf32>, %lb: index, %ub: index, %step: index) -> tensor<64xf32> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%iter = %arg0) -> (tensor<64xf32>) {
      %scope_res = scope.scope : () -> tensor<64xf32> {
        %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32} : memref<64xf32>
        %tensor = bufferization.to_tensor %alloc restrict writable : memref<64xf32> to tensor<64xf32>
        %custom = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "tanh_fp32"} "tanh_fp32" ins(%iter : tensor<64xf32>) outs(%tensor : tensor<64xf32>) -> tensor<64xf32>
        scope.return %custom : tensor<64xf32>
      } {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}
      scf.yield %scope_res : tensor<64xf32>
    }
    return %res : tensor<64xf32>
  }

  // ============================================================================
  // 4. test_unpack_scope_single_block_id_attr
  // Op inside scope has only 1 attribute (kBlockId). Tests canReuseAttrs branch.
  // ============================================================================
  // CHECK-LABEL: func.func @test_unpack_scope_single_block_id_attr
  // CHECK-NOT: scope.scope
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<128xf32>
  // CHECK: return %[[ALLOC]] : memref<128xf32>
  func.func @test_unpack_scope_single_block_id_attr() -> memref<128xf32> {
    %res = scope.scope : () -> memref<128xf32> {
      %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32} : memref<128xf32>
      scope.return %alloc : memref<128xf32>
    } {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"}
    return %res : memref<128xf32>
  }
}
