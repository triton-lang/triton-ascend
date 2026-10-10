// RUN: triton-opt --op-classifier %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

  // ============================================================================
  // 1. test_classify_and_pack_vector_custom_op
  // Pattern: alloc -> to_tensor -> hivm.hir.custom (VECTOR) -> external consumer.
  // OpClassifierPass groups these related ops into a scope.scope with core_type="VECTOR".
  // ============================================================================
  // CHECK-LABEL: func.func @test_classify_and_pack_vector_custom_op
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<2048xf32> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<2048xf32>
  // CHECK:   %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.core_type = "VECTOR"} : memref<2048xf32> to tensor<2048xf32>
  // CHECK:   %[[CUSTOM:.*]] = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, ssbuffer.core_type = "VECTOR", symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%[[TENSOR]] : tensor<2048xf32>) -> tensor<2048xf32>
  // CHECK:   scope.return %[[CUSTOM]] : tensor<2048xf32>
  // CHECK: } {ssbuffer.core_type = "VECTOR"}
  // CHECK: %[[EXP:.*]] = tensor.expand_shape %[[SCOPE]] {{\[\[}}0, 1{{\]\]}} output_shape [64, 32] {ssbuffer.core_type = "VECTOR"} : tensor<2048xf32> into tensor<64x32xf32>
  // CHECK: return %[[EXP]] : tensor<64x32xf32>
  func.func @test_classify_and_pack_vector_custom_op(%arg0: tensor<2048xf32>) -> tensor<64x32xf32> {
    %alloc = memref.alloc() : memref<2048xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
    %1 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [64, 32] : tensor<2048xf32> into tensor<64x32xf32>
    return %expanded : tensor<64x32xf32>
  }

  // ============================================================================
  // 2. test_classify_and_pack_cube_custom_op
  // Pattern: alloc -> to_tensor -> hivm.hir.custom (CUBE) -> external consumer.
  // OpClassifierPass groups these related ops into a scope.scope with core_type="CUBE".
  // ============================================================================
  // CHECK-LABEL: func.func @test_classify_and_pack_cube_custom_op
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<2048xf32> {
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<2048xf32>
  // CHECK:   %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.core_type = "CUBE"} : memref<2048xf32> to tensor<2048xf32>
  // CHECK:   %[[CUSTOM:.*]] = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_MTE1>, hivm.tcore_type = #hivm.tcore_type<CUBE>, ssbuffer.core_type = "CUBE", symbol = "cube_custom"} "cube_custom" ins(%arg0 : tensor<2048xf32>) outs(%[[TENSOR]] : tensor<2048xf32>) -> tensor<2048xf32>
  // CHECK:   scope.return %[[CUSTOM]] : tensor<2048xf32>
  // CHECK: } {ssbuffer.core_type = "CUBE"}
  // CHECK: %[[EXP:.*]] = tensor.expand_shape %[[SCOPE]] {{\[\[}}0, 1{{\]\]}} output_shape [64, 32] {ssbuffer.core_type = "VECTOR"} : tensor<2048xf32> into tensor<64x32xf32>
  // CHECK: return %[[EXP]] : tensor<64x32xf32>
  func.func @test_classify_and_pack_cube_custom_op(%arg0: tensor<2048xf32>) -> tensor<64x32xf32> {
    %alloc = memref.alloc() : memref<2048xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
    %1 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_MTE1>, hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "cube_custom"} "cube_custom" ins(%arg0 : tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [64, 32] : tensor<2048xf32> into tensor<64x32xf32>
    return %expanded : tensor<64x32xf32>
  }

  // ============================================================================
  // 3. test_classify_and_pack_custom_op_in_loop
  // Pattern inside scf.for loop (matching flash_fwd_kernel in passes.txt).
  // Verifies that scope.scope correctly isolates customOp within loop iterations.
  // ============================================================================
  // CHECK-LABEL: func.func @test_classify_and_pack_custom_op_in_loop
  // CHECK: scf.for
  // CHECK:   %[[SCOPE_LOOP:.*]] = scope.scope : () -> tensor<2048xf32> {
  // CHECK:     %[[ALLOC_LOOP:.*]] = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<2048xf32>
  // CHECK:     %[[TENSOR_LOOP:.*]] = bufferization.to_tensor %[[ALLOC_LOOP]] restrict writable {ssbuffer.core_type = "VECTOR"} : memref<2048xf32> to tensor<2048xf32>
  // CHECK:     %[[CUSTOM_LOOP:.*]] = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, ssbuffer.core_type = "VECTOR", symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%[[TENSOR_LOOP]] : tensor<2048xf32>) -> tensor<2048xf32>
  // CHECK:     scope.return %[[CUSTOM_LOOP]] : tensor<2048xf32>
  // CHECK:   } {ssbuffer.core_type = "VECTOR"}
  // CHECK:   %[[EXP_LOOP:.*]] = tensor.expand_shape %[[SCOPE_LOOP]] {{\[\[}}0, 1{{\]\]}} output_shape [64, 32] {ssbuffer.core_type = "VECTOR"} : tensor<2048xf32> into tensor<64x32xf32>
  // CHECK:   scf.yield {{.*}}%[[EXP_LOOP]] : tensor<64x32xf32>
  func.func @test_classify_and_pack_custom_op_in_loop(%arg0: tensor<2048xf32>, %lb: index, %ub: index, %step: index) -> tensor<64x32xf32> {
    %empty = tensor.empty() : tensor<64x32xf32>
    %res = scf.for %iv = %lb to %ub step %step iter_args(%iter = %empty) -> (tensor<64x32xf32>) {
      %alloc = memref.alloc() : memref<2048xf32>
      %0 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32> to tensor<2048xf32>
      %1 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "tanh_fp32"} "tanh_fp32" ins(%arg0 : tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [64, 32] : tensor<2048xf32> into tensor<64x32xf32>
      scf.yield %expanded : tensor<64x32xf32>
    }
    return %res : tensor<64x32xf32>
  }
}
