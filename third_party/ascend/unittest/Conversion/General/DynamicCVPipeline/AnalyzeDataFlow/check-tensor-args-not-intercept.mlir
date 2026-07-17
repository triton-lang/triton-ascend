// RUN: triton-opt --analyze-args %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chunk_gated_delta_rule_bwd_kernel_dhu_k128_blockdim128(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<32x64xf32>
    %1 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst_0 : f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    scf.for %arg14 = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%arg15 = %1) -> (tensor<32x64xf32>) : i32 {
      %5 = arith.addf %arg15, %arg15 {DataUse, ssbuffer.block_id = 10 : i32} : tensor<32x64xf32>
      %new = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst_0 : f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
      scf.yield %new : tensor<32x64xf32>
    } {ssbuffer.main_loop = 0 : i32}
    return
  }
}

// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK: return
