// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s --implicit-check-not=scf.while

// The normal preload scheme is only enabled for the kernels whose pipeline
// dependency is a single chain of four computation blocks that alternates the
// cores. A kernel that holds two such chains keeps the conditions of the
// upstream implementation: the scheme is only validated for a single chain.
//
// This kernel has the dependency chain of the test
// pattern-kernel-conditions.mlir:
//   cube block 0 -> vector block 1 -> cube block 2 -> vector block 3
// and a second chain that shares no block with it:
//   cube block 4 -> vector block 5 -> cube block 6 -> vector block 7
// Both chains are double buffered, so both of them match the pattern, but
// they are two independent pipelines: the kernel keeps the upstream scheme,
// the main loop stays an scf.for and the blocks handshake through PIPE_S. The
// implicit check forbids the scf.while of the normal preload scheme.

// CHECK-LABEL: func.func @pattern_two_chains_kernel_conditions

// The upstream scheme: the main loop stays an scf.for, the flags of the
// blocks are published through the PIPE_S handshake.
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pattern_two_chains_kernel_conditions(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c8192_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i32

    scope.scope : () -> () {
      %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc0b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc0b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_1 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>

      // The second chain, outside of the main loop: the vector block 5
      // consumes the group 3 produced by the cube block 4.
      %alloc_v3 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_v3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<8>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_v3b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_v3b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<9>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %memspacecast_v3 = memref.memory_space_cast %alloc_v3 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [3 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
      %memspacecast_v3b = memref.memory_space_cast %alloc_v3b {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [3 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
      // The vector block 5 produces the group 4 consumed by the cube block 6.
      %cbuf_v5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_v5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<10>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_v5b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_v5b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<11>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %reshape_5 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
      hivm.hir.copy ins(%reshape_5 : tensor<8x8x16x16xf16>) outs(%cbuf_v5 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [4 : i32, 1 : i32]}
      hivm.hir.copy ins(%reshape_5 : tensor<8x8x16x16xf16>) outs(%cbuf_v5b : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [4 : i32, 1 : i32]}
      // The vector block 7 consumes the group 5 produced by the cube block 6.
      %alloc_v5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_v5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_v5b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_v5b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<13>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %memspacecast_v5 = memref.memory_space_cast %alloc_v5 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 5 : i32, ssbuffer.crossCoreDeps = [5 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
      %memspacecast_v5b = memref.memory_space_cast %alloc_v5b {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 5 : i32, ssbuffer.crossCoreDeps = [5 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>

      %30:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        // Vector block 1 consumes the group 0 produced by the cube block 0.
        hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_1 = memref.memory_space_cast %alloc_1 {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32], ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        // Vector block 1 produces the group 1 consumed by the cube block 2.
        %reshape_1 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%reshape_1 : tensor<8x8x16x16xf16>) outs(%alloc0 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]}
        hivm.hir.copy ins(%reshape_1 : tensor<8x8x16x16xf16>) outs(%alloc0b : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]}
        // Vector block 3 consumes the group 2 produced by the cube block 2.
        hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_3 = memref.memory_space_cast %alloc_3 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc_c = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_c {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_c1 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_c1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_c1b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_c1b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_c2 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_c2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_c2b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_c2b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>

      // The second chain, outside of the main loop: the cube block 4
      // produces the group 3 consumed by the vector block 5.
      %alloc_c3 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_c3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<8>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_c3b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_c3b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<9>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %50 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128x128xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [3 : i32, 1 : i32]} ins(%50 : tensor<128x128xf32>) outs(%alloc_c3 : memref<128x128xf32, #hivm.address_space<ub>>)
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [3 : i32, 1 : i32]} ins(%50 : tensor<128x128xf32>) outs(%alloc_c3b : memref<128x128xf32, #hivm.address_space<ub>>)
      // The cube block 6 consumes the group 4 produced by the vector block 5.
      %cbuf_c6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_c6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<10>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_c6b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_c6b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<11>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %tile_c6 = bufferization.to_tensor %cbuf_c6 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [4 : i32, 0 : i32]} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to tensor<8x8x16x16xf16>
      %tile_c6b = bufferization.to_tensor %cbuf_c6b {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [4 : i32, 0 : i32]} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to tensor<8x8x16x16xf16>
      // The cube block 6 produces the group 5 consumed by the vector block 7.
      %alloc_c5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_c5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_c5b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_c5b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<13>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %60 = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<128x128xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 5 : i32, ssbuffer.crossCoreDeps = [5 : i32, 1 : i32]} ins(%60 : tensor<128x128xf32>) outs(%alloc_c5 : memref<128x128xf32, #hivm.address_space<ub>>)
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 5 : i32, ssbuffer.crossCoreDeps = [5 : i32, 1 : i32]} ins(%60 : tensor<128x128xf32>) outs(%alloc_c5b : memref<128x128xf32, #hivm.address_space<ub>>)

      %20:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        // Cube block 0 produces the group 0 consumed by the vector block 1.
        hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %30 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%30 : tensor<128x128xf32>) outs(%alloc_c1 : memref<128x128xf32, #hivm.address_space<ub>>)
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%30 : tensor<128x128xf32>) outs(%alloc_c1b : memref<128x128xf32, #hivm.address_space<ub>>)
        // Cube block 2 consumes the group 1 produced by the vector block 1.
        hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %conv = hivm.hir.convert_layout %alloc_c output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        // Cube block 2 produces the group 2 consumed by the vector block 3.
        %40 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]} ins(%40 : tensor<128x128xf32>) outs(%alloc_c2 : memref<128x128xf32, #hivm.address_space<ub>>)
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]} ins(%40 : tensor<128x128xf32>) outs(%alloc_c2b : memref<128x128xf32, #hivm.address_space<ub>>)
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
