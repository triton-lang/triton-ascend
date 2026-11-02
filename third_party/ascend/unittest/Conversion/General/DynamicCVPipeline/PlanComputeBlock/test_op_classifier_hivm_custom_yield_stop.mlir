// RUN: triton-opt --op-classifier %s | FileCheck %s

// Tests for the fix `[opclassifier](fix) stop BFS at hivm.hir.custom in yield
// propagation` (commit bc216bba4).
//
// Step 8's `propagateCoreTypeUpwardForYield` walks upstream from a yield
// operand (e.g. the matmul feeding `scf.yield`) to mark every reachable op
// with the yield's core_type.  The bug: a `hivm.hir.custom` (e.g. tanh) sits
// in the middle of the chain.  Without the fix, the BFS continues THROUGH
// the custom's operands and would (via mem-deps) re-mark a memref.alloc
// already classified as CUBE (because it backs a matmul input via
// bufferization.to_tensor) with CUBE_AND_VECTOR.  The fix adds `hivm.hir.custom`
// to `shouldStopPropagationForYield` so the BFS halts at the custom and the
// upstream alloc keeps its VECTOR classification.
//
// This test reproduces the cwy.mlir flash-attention pattern: a for body
// that contains `truncf -> to_tensor -> custom -> expand_shape -> matmul` and
// the matmul result is the yield operand.

module {
  // ============================================================================
  // 1. custom_stops_yield_bfs
  // A scf.for body yields a matmul (CUBE).  The matmul's input chain
  // includes a `hivm.hir.custom` (tanh, VECTOR).  The BFS walks from the
  // matmul upward: matmul -> expand_shape -> CUSTOM (stop).  The
  // bufferization.to_tensor and its backing memref.alloc (which would
  // otherwise be promoted to CUBE_AND_VECTOR via the cross-mem-dep edge)
  // stay VECTOR.
  // ============================================================================
  // CHECK-LABEL: func.func @custom_stops_yield_bfs(
  // CHECK:       memref.alloc() {{.*}}{ssbuffer.core_type = "VECTOR"
  // CHECK:       bufferization.to_tensor {{.*}}{ssbuffer.core_type = "VECTOR"
  // CHECK:       hivm.hir.custom {{.*}}ssbuffer.core_type = "VECTOR"
  // CHECK:       tensor.expand_shape {{.*}}{ssbuffer.core_type = "VECTOR"
  // CHECK:       linalg.matmul {{.*}}{ssbuffer.core_type = "CUBE"
  // CHECK:       scf.yield {{.*}}{ssbuffer.core_type = "CUBE"
  // CHECK:       } {ssbuffer.core_type = "CUBE"}
  // CHECK-NOT:   memref.alloc() {{.*}}{ssbuffer.core_type = "CUBE_AND_VECTOR"
  // CHECK-NOT:   hivm.hir.custom {{.*}}{ssbuffer.core_type = "CUBE_AND_VECTOR"
  func.func @custom_stops_yield_bfs(%a: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.0 : f32
    %empty_f32 = tensor.empty() : tensor<16x16xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty_f32 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %trunc = arith.truncf %a : tensor<16x16xf32> to tensor<16x16xf16>
    %alloc = memref.alloc() : memref<16x16xf16>
    %to_tensor = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16> to tensor<16x16xf16>
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) {
      %add = arith.addf %trunc, %to_tensor : tensor<16x16xf16>
      %extf = arith.extf %add : tensor<16x16xf16> to tensor<16x16xf32>
      %collapsed = tensor.collapse_shape %extf [[0, 1]] : tensor<16x16xf32> into tensor<256xf32>
      %alloc2 = memref.alloc() : memref<256xf32>
      %to_tensor2 = bufferization.to_tensor %alloc2 restrict writable : memref<256xf32> to tensor<256xf32>
      %tanh = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "tanh_fp32"} "tanh_fp32" ins(%collapsed : tensor<256xf32>) outs(%to_tensor2 : tensor<256xf32>) -> tensor<256xf32>
      %expanded = tensor.expand_shape %tanh [[0, 1]] output_shape [16, 16] : tensor<256xf32> into tensor<16x16xf32>
      %mm = linalg.matmul ins(%expanded, %expanded : tensor<16x16xf32>, tensor<16x16xf32>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %mm : tensor<16x16xf32>
    }
    return %result : tensor<16x16xf32>
  }

  // ============================================================================
  // 2. custom_yield_target_still_walks_through_unrelated_branches
  // A scf.if with two branches: then has a matmul (CUBE), else has a custom
  // (VECTOR) but no matmul.  The yield target is CUBE (from then).  With
  // the fix, the BFS stops at the else-branch custom, so the else branch's
  // alloc + to_tensor stay VECTOR.  The then branch's matmul + the
  // matmul's chain propagate CUBE upstream.
  // ============================================================================
  // CHECK-LABEL: func.func @custom_yield_target_still_walks_through_unrelated_branches(
  // CHECK:       memref.alloc() {{.*}}{ssbuffer.core_type = "VECTOR"
  // CHECK:       bufferization.to_tensor {{.*}}{ssbuffer.core_type = "VECTOR"
  // CHECK:       scf.if
  // CHECK:       linalg.matmul {{.*}}{ssbuffer.core_type = "CUBE"
  // CHECK:       scf.yield {{.*}}{ssbuffer.core_type = "CUBE"
  // CHECK:       hivm.hir.custom {{.*}}ssbuffer.core_type = "VECTOR"
  // CHECK:       scf.yield {{.*}}{ssbuffer.core_type = "CUBE"
  // CHECK:       } {ssbuffer.core_type = "CUBE"}
  func.func @custom_yield_target_still_walks_through_unrelated_branches(%a: tensor<16x16xf32>, %cond: i1) -> tensor<16x16xf32> {
    %cst = arith.constant 0.0 : f32
    %empty_f32 = tensor.empty() : tensor<16x16xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty_f32 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %trunc = arith.truncf %a : tensor<16x16xf32> to tensor<16x16xf16>
    %alloc = memref.alloc() : memref<16x16xf16>
    %to_tensor = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16> to tensor<16x16xf16>
    %result = scf.if %cond -> (tensor<16x16xf32>) {
      %add = arith.addf %trunc, %to_tensor : tensor<16x16xf16>
      %extf = arith.extf %add : tensor<16x16xf16> to tensor<16x16xf32>
      %mm = linalg.matmul ins(%extf, %extf : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %mm : tensor<16x16xf32>
    } else {
      %collapsed = tensor.collapse_shape %to_tensor [[0, 1]] : tensor<16x16xf16> into tensor<256xf16>
      %alloc2 = memref.alloc() : memref<256xf16>
      %to_tensor2 = bufferization.to_tensor %alloc2 restrict writable : memref<256xf16> to tensor<256xf16>
      %custom = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "some_op"} "some_op" ins(%collapsed : tensor<256xf16>) outs(%to_tensor2 : tensor<256xf16>) -> tensor<256xf16>
      %expanded = tensor.expand_shape %custom [[0, 1]] output_shape [16, 16] : tensor<256xf16> into tensor<16x16xf16>
      %extf2 = arith.extf %expanded : tensor<16x16xf16> to tensor<16x16xf32>
      scf.yield %extf2 : tensor<16x16xf32>
    }
    return %result : tensor<16x16xf32>
  }
}
