// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// A VECTOR -> CUBE publication whose producer (sync_block_set) and matching
// read wait are both outside the main loops, while the matmul consumer is
// inside the cube loop. ProcessLoopInvariantRead runs as Step8 of
// AddControlFlowCondition and sinks the wait in front of the matmul, guarded
// by the first execution of the ssbuffer block, with a zero-trip fallback.

// CHECK-LABEL: func.func @process_loop_invariant_read
// The zero-trip fallback: negated loop entry condition + wait before the loop.
// CHECK: arith.xori
// CHECK-NEXT: scf.if
// CHECK-NEXT: hivm.hir.sync_block_wait{{.*}}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
// The first-execution mask in front of the consumer.
// CHECK: arith.cmpi eq,
// CHECK-NEXT: scf.if
// CHECK-NEXT: hivm.hir.sync_block_wait{{.*}}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
// CHECK: linalg.matmul

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32, ssbuffer.load_store_buf_conut = 1 : i32} {
  func.func @process_loop_invariant_read(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32

    scope.scope : () -> () {
      // The pipeline of the kernel: the vector core writes a tile into a cross
      // core buffer that the cube core reads, the cube core writes a tile of
      // its own that the vector core reads back. The three transfers make the
      // cross core dependency of the module a VECTOR -> CUBE -> VECTOR -> CUBE
      // chain, the shape the normal preload scheme is enabled on. The scheme
      // only counts the cross core blocks that carry a condition, so the
      // blocks of the chain are the blocks of the main loop. The transfers
      // themselves sit outside the loop: they give the module the pipeline
      // shape without changing the conditions of the loop the test checks.
      %cbuf_t2 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_t2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %dup_tcb_5 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %dup_tcb_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_t3 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_t3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_t4 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_t4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %dup_tcb_7 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %dup_tcb_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %tile_v2 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<8x4x16x16xf16>
      hivm.hir.copy ins(%tile_v2 : tensor<8x4x16x16xf16>) outs(%cbuf_t2 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [30 : i32, 1 : i32]}
      hivm.hir.copy ins(%tile_v2 : tensor<8x4x16x16xf16>) outs(%dup_tcb_5 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [30 : i32, 1 : i32]}
      %tile_v3 = bufferization.to_tensor %cbuf_t3 {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [31 : i32, 0 : i32]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>> to tensor<8x4x16x16xf16>
      %tile_v4 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<8x4x16x16xf16>
      hivm.hir.copy ins(%tile_v4 : tensor<8x4x16x16xf16>) outs(%cbuf_t4 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [32 : i32, 1 : i32]}
      hivm.hir.copy ins(%tile_v4 : tensor<8x4x16x16xf16>) outs(%dup_tcb_7 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [32 : i32, 1 : i32]}
      %alloc0 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem0 = memref.memory_space_cast %alloc0 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      %alloc1 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem1 = memref.memory_space_cast %alloc1 : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      // q is published once, outside the vector main loop.
      hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0

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

        // Pure scalar computation: it has no dependency at all.
        %t3 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t3 : tensor<128xf32>) outs(%mem1 : memref<128xf32>) {ssbuffer.block_id = 3 : i32}

        // Second intra consumer (attribute group 0 -> DEP1).
        %t4_in = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t4_in : tensor<128xf32>) outs(%mem1 : memref<128xf32>) {ssbuffer.block_id = 4 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]}
        scf.yield
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %cbuf_c2 = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_c2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_c3 = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_c3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %dup_tcb_6 = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %dup_tcb_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_c4 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_c4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %tile_c2 = bufferization.to_tensor %cbuf_c2 {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [30 : i32, 0 : i32]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>> to tensor<8x4x16x16xf16>
      %tile_c3 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<8x4x16x16xf16>
      hivm.hir.copy ins(%tile_c3 : tensor<8x4x16x16xf16>) outs(%cbuf_c3 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [31 : i32, 1 : i32]}
      hivm.hir.copy ins(%tile_c3 : tensor<8x4x16x16xf16>) outs(%dup_tcb_6 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [31 : i32, 1 : i32]}
      %tile_c4 = bufferization.to_tensor %cbuf_c4 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [32 : i32, 0 : i32]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>> to tensor<8x4x16x16xf16>
      %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem = memref.memory_space_cast %alloc : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      // Loop-invariant read wait at the cube block head, followed by the q
      // receive chain that the in-loop matmul consumes.
      hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
      %q_cl = hivm.hir.convert_layout %cbuf_c2 output_shape [128, 64] {srcLayout = #hivm.data_layout<nZ>, dstLayout = #hivm.data_layout<ND>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32} : (memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x64xf16, #hivm.address_space<cbuf>>
      %q_ms = memref.memory_space_cast %q_cl {ssbuffer.block_id = 4 : i32} : memref<128x64xf16, #hivm.address_space<cbuf>> to memref<128x64xf16>
      %q = bufferization.to_tensor %q_ms restrict writable {ssbuffer.block_id = 4 : i32} : memref<128x64xf16> to tensor<128x64xf16>
      scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
        %tp = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%tp : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 4 : i32, ssbuffer.intraDeps = [20 : i32, 1 : i32]}
        %qk = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<64x64xf16>
        %qacc = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128x64xf32>
        %qmm = linalg.matmul {ssbuffer.block_id = 4 : i32} ins(%q, %qk : tensor<128x64xf16>, tensor<64x64xf16>) outs(%qacc : tensor<128x64xf32>) -> tensor<128x64xf32>
        %tc = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%tc : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 5 : i32, ssbuffer.intraDeps = [20 : i32, 0 : i32]}
        scf.yield
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
