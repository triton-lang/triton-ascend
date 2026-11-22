// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s --implicit-check-not=scf.while

// The two chains of the kernel of
// pattern-two-pipelines-kernel-conditions.mlir share a tightly coupled
// buffer: the two pipelines are not independent, so the kernel keeps the
// conditions of the upstream implementation and its main loops stay scf.for
// loops.

// CHECK-LABEL: func.func @pattern_two_pipelines_shared_buffer_kernel_conditions

// The upstream scheme: the main loops stay scf.for loops and the flags
// of the blocks are published through the PIPE_S handshake.
// CHECK: scf.for
// CHECK-DAG: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-DAG: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-DAG: hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-DAG: hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pattern_two_pipelines_shared_buffer_kernel_conditions(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c8192_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i32

    scope.scope : () -> () {
      %v_a0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %v_a0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %v_a2 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %v_a2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %v_a1 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %v_a1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %v_a1b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %v_a1b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %v_b0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %v_b0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %v_b2 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %v_b2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<10>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %v_b1 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %v_b1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<8>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %v_b1b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %v_b1b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<9>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>

      %10:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_1 = memref.memory_space_cast %v_a0 {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %reshape_1 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%reshape_1 : tensor<8x8x16x16xf16>) outs(%v_a1 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]}
        hivm.hir.copy ins(%reshape_1 : tensor<8x8x16x16xf16>) outs(%v_a1b : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]}
        hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_3 = memref.memory_space_cast %v_a2 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}

      %30:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_5 = memref.memory_space_cast %v_b0 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [3 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %reshape_5 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%reshape_5 : tensor<8x8x16x16xf16>) outs(%v_b1 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [4 : i32, 1 : i32]}
        hivm.hir.copy ins(%reshape_5 : tensor<8x8x16x16xf16>) outs(%v_b1b : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [4 : i32, 1 : i32]}
        hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_7 = memref.memory_space_cast %v_b2 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [5 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 1 : i64}

      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %c_a0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %c_a0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %c_a0b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %c_a0b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %c_a1 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %c_a1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %c_a2 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %c_a2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %c_a2b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %c_a2b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %c_b0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %c_b0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %c_b0b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %c_b0b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %c_b1 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %c_b1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<8>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %c_b2 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %c_b2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<10>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %c_b2b = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %c_b2b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<11>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>

      %10:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %tensor_0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%tensor_0 : tensor<128x128xf32>) outs(%c_a0 : memref<128x128xf32, #hivm.address_space<ub>>)
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%tensor_0 : tensor<128x128xf32>) outs(%c_a0b : memref<128x128xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %conv_2 = hivm.hir.convert_layout %c_a1 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %tensor_2 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]} ins(%tensor_2 : tensor<128x128xf32>) outs(%c_a2 : memref<128x128xf32, #hivm.address_space<ub>>)
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]} ins(%tensor_2 : tensor<128x128xf32>) outs(%c_a2b : memref<128x128xf32, #hivm.address_space<ub>>)
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}

      %30:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %tensor_4 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [3 : i32, 1 : i32]} ins(%tensor_4 : tensor<128x128xf32>) outs(%c_b0 : memref<128x128xf32, #hivm.address_space<ub>>)
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [3 : i32, 1 : i32]} ins(%tensor_4 : tensor<128x128xf32>) outs(%c_b0b : memref<128x128xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_wait {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %conv_6 = hivm.hir.convert_layout %c_b1 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [4 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %tensor_6 = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [5 : i32, 1 : i32]} ins(%tensor_6 : tensor<128x128xf32>) outs(%c_b2 : memref<128x128xf32, #hivm.address_space<ub>>)
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [5 : i32, 1 : i32]} ins(%tensor_6 : tensor<128x128xf32>) outs(%c_b2b : memref<128x128xf32, #hivm.address_space<ub>>)
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 1 : i64}

      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
