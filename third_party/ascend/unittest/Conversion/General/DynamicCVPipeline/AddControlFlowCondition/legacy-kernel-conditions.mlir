// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// The kernels listed in legacyKernel of AddControlFlowCondition.cpp keep the
// control flow conditions of the upstream implementation. The other kernels
// get the normal preload scheme: a while loop for the main loop, a flag chain
// computed with arith.select and no PIPE_S handshake.
//
// The upstream path keeps the for loop of the main loop.
// The func name is one of the legacyKernel names of AddControlFlowCondition.cpp,
// it is the way to reach the legacy path.
// CHECK-LABEL: func.func @chunk_gated_delta_rule_fwd_kernel_h_blockdim64_npu_kernel
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: scf.for
// The upstream path synchronizes the cores with PIPE_S, the normal preload
// scheme counts the rounds with the block counters instead.
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// The upstream path yields the counters from the blocks, the normal preload
// scheme computes the flags with a select after the blocks.
// CHECK-NOT: arith.select
// CHECK-NOT: scf.while

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chunk_gated_delta_rule_fwd_kernel_h_blockdim64_npu_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32

    scope.scope : () -> () {
      %alloc0 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem0 = memref.memory_space_cast %alloc0 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      %alloc1 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem1 = memref.memory_space_cast %alloc1 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>

      scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
        // Producers of attribute group 1. Its consumer is in the next block.
        %t1_p0 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t1_p0 : tensor<128xf32>) outs(%mem0 : memref<128xf32>) {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]}
        %t1_p1 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t1_p1 : tensor<128xf32>) outs(%mem0 : memref<128xf32>) {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]}
        %t1_p2 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t1_p2 : tensor<128xf32>) outs(%mem0 : memref<128xf32>) {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]}

        // First intra consumer (attribute group 1 -> DEP0). Also produces group 0.
        %t2_in = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t2_in : tensor<128xf32>) outs(%mem0 : memref<128xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.intraDeps = [1 : i32, 0 : i32]}
        %t2_out = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t2_out : tensor<128xf32>) outs(%mem1 : memref<128xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]}

        // Second intra consumer (attribute group 0 -> DEP1).
        %t3_in = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t3_in : tensor<128xf32>) outs(%mem1 : memref<128xf32>) {ssbuffer.block_id = 3 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]}
        scf.yield
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem = memref.memory_space_cast %alloc : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
        %tp = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%tp : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 3 : i32, ssbuffer.intraDeps = [20 : i32, 1 : i32]}
        %tc = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%tc : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 4 : i32, ssbuffer.intraDeps = [20 : i32, 0 : i32]}
        scf.yield
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
