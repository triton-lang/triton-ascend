// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s --implicit-check-not=scf.for

// A kernel whose pipeline is a forest of four chains of three computation
// blocks:
//   cube 5 -> vector 24 -> cube 7
//   cube 9 -> vector 25 -> cube 11
//   cube 13 -> vector 27 -> cube 15
//   cube 17 -> vector 28 -> cube 19
// The blocks that carry a condition are the twelve blocks of the forest. The
// transfers of the chains are single buffered, so every direction carries
// exactly one tightly coupled buffer and the forest holds eight of them. The
// pattern recognizes the shape, so the kernel uses the normal preload
// scheme: every main loop is an scf.while.

// CHECK-LABEL: func.func @pattern_forest_kernel_conditions
// CHECK: scf.while
// CHECK: scf.while
// CHECK: scf.while
// CHECK: scf.while

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pattern_forest_kernel_conditions(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 1 : i32, ssbuffer.intra_buf_count = 2 : i32} {
    %c0_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 0 : i32
    %c8192_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 128 : i32

    scope.scope : () -> () {
      // The vector scope: the groups 0, 2, 4 and 6 are consumed here.
      %v_c0 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %v_c0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      // The groups 1, 3, 5 and 7 are produced here.
      %v_p1 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %v_p1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      %v_c2 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 2 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %v_c2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 2 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      %v_p3 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 3 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %v_p3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 3 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      %v_c4 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 4 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %v_c4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 4 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      %v_p5 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 5 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %v_p5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 5 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      %v_c6 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 6 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %v_c6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 6 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      %v_p7 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 7 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %v_p7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 7 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>

      %10:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        %memspacecast_0 = memref.memory_space_cast %v_c0 {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : memref<16x32xf32, #hivm.address_space<ub>> to memref<16x32xf32>
        %reshape_1_0 = tensor.empty() {ssbuffer.block_id = 24 : i32} : tensor<2x1x16x16xbf16>
        hivm.hir.copy ins(%reshape_1_0 : tensor<2x1x16x16xbf16>) outs(%v_p1 : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]}
        %memspacecast_2 = memref.memory_space_cast %v_c2 {ssbuffer.block_id = 25 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]} : memref<16x32xf32, #hivm.address_space<ub>> to memref<16x32xf32>
        %reshape_3_0 = tensor.empty() {ssbuffer.block_id = 25 : i32} : tensor<2x1x16x16xbf16>
        hivm.hir.copy ins(%reshape_3_0 : tensor<2x1x16x16xbf16>) outs(%v_p3 : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 25 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [3 : i32, 1 : i32]}
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 12 : i32, ssbuffer.main_loop = 0 : i64}

      %11:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        %memspacecast_4 = memref.memory_space_cast %v_c4 {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [4 : i32, 0 : i32]} : memref<16x32xf32, #hivm.address_space<ub>> to memref<16x32xf32>
        %reshape_5_0 = tensor.empty() {ssbuffer.block_id = 27 : i32} : tensor<2x1x16x16xbf16>
        hivm.hir.copy ins(%reshape_5_0 : tensor<2x1x16x16xbf16>) outs(%v_p5 : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 5 : i32, ssbuffer.crossCoreDeps = [5 : i32, 1 : i32]}
        %memspacecast_6 = memref.memory_space_cast %v_c6 {ssbuffer.block_id = 28 : i32, ssbuffer.transfer_id = 6 : i32, ssbuffer.crossCoreDeps = [6 : i32, 0 : i32]} : memref<16x32xf32, #hivm.address_space<ub>> to memref<16x32xf32>
        %reshape_7_0 = tensor.empty() {ssbuffer.block_id = 28 : i32} : tensor<2x1x16x16xbf16>
        hivm.hir.copy ins(%reshape_7_0 : tensor<2x1x16x16xbf16>) outs(%v_p7 : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 28 : i32, ssbuffer.transfer_id = 7 : i32, ssbuffer.crossCoreDeps = [7 : i32, 1 : i32]}
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 12 : i32, ssbuffer.main_loop = 1 : i64}

      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      // The cube scope: the groups 0, 2, 4 and 6 are produced here.
      %c_p0 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %c_p0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      // The groups 1, 3, 5 and 7 are consumed here.
      %c_c1 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %c_c1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      %c_p2 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 2 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %c_p2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 2 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      %c_c3 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 3 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %c_c3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 3 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      %c_p4 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 4 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %c_p4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 4 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      %c_c5 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 5 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %c_c5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 5 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      %c_p6 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 6 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %c_p6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 6 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      %c_c7 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 7 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %c_c7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 7 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>

      %10:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        %tensor_0_0 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<16x32xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%tensor_0_0 : tensor<16x32xf32>) outs(%c_p0 : memref<16x32xf32, #hivm.address_space<ub>>)
        %conv_1 = hivm.hir.convert_layout %c_c1 output_shape [16, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32]} : (memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<16x32xbf16, #hivm.address_space<cbuf>>
        %tensor_2_0 = tensor.empty() {ssbuffer.block_id = 9 : i32} : tensor<16x32xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]} ins(%tensor_2_0 : tensor<16x32xf32>) outs(%c_p2 : memref<16x32xf32, #hivm.address_space<ub>>)
        %conv_3 = hivm.hir.convert_layout %c_c3 output_shape [16, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [3 : i32, 0 : i32]} : (memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<16x32xbf16, #hivm.address_space<cbuf>>
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 12 : i32, ssbuffer.main_loop = 0 : i64}

      %11:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        %tensor_4_0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16x32xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [4 : i32, 1 : i32]} ins(%tensor_4_0 : tensor<16x32xf32>) outs(%c_p4 : memref<16x32xf32, #hivm.address_space<ub>>)
        %conv_5 = hivm.hir.convert_layout %c_c5 output_shape [16, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 5 : i32, ssbuffer.crossCoreDeps = [5 : i32, 0 : i32]} : (memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<16x32xbf16, #hivm.address_space<cbuf>>
        %tensor_6_0 = tensor.empty() {ssbuffer.block_id = 17 : i32} : tensor<16x32xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 6 : i32, ssbuffer.crossCoreDeps = [6 : i32, 1 : i32]} ins(%tensor_6_0 : tensor<16x32xf32>) outs(%c_p6 : memref<16x32xf32, #hivm.address_space<ub>>)
        %conv_7 = hivm.hir.convert_layout %c_c7 output_shape [16, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 7 : i32, ssbuffer.crossCoreDeps = [7 : i32, 0 : i32]} : (memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<16x32xbf16, #hivm.address_space<cbuf>>
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 12 : i32, ssbuffer.main_loop = 1 : i64}

      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}

    return {ssbuffer.core_type = "VECTOR"}
  }
}
