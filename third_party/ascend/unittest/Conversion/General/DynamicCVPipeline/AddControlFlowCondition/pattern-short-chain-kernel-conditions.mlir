// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s --implicit-check-not=scf.for

// A kernel whose pipeline is a chain of three computation blocks that
// alternate the cores: cube 4 -> vector 11 -> cube 6. The chain holds two
// transfers, so it carries four tightly coupled buffers. The blocks that
// carry a condition are the three blocks of the chain and the vector block
// 6, that takes part in no cross core transfer. The pattern recognizes the
// shape, so the kernel uses the normal preload scheme: every main loop is
// an scf.while.

// CHECK-LABEL: func.func @pattern_short_chain_kernel_conditions
// CHECK: scf.while
// CHECK: scf.while

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pattern_short_chain_kernel_conditions(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c0_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 0 : i32
    %c8192_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 128 : i32

    scope.scope : () -> () {
      // The vector scope: the group 0 is produced here, the group 1 consumed.
      %v_p0 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %v_p0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      %v_p1 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %v_p1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      %v_c0 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %v_c0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      %v_c1 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %v_c1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>

      %10:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        // The vector block 11 consumes the group 1 and produces the group 0.
        %memspacecast_0 = memref.memory_space_cast %v_c0 {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32]} : memref<16x32xf32, #hivm.address_space<ub>> to memref<16x32xf32>
        %reshape_0 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<2x1x16x16xbf16>
        hivm.hir.copy ins(%reshape_0 : tensor<2x1x16x16xbf16>) outs(%v_p0 : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]}
        hivm.hir.copy ins(%reshape_0 : tensor<2x1x16x16xbf16>) outs(%v_p1 : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]}
        // The vector block 6 carries a condition and takes no transfer.
        %addi_0 = arith.addi %arg17, %c128_i32 {ssbuffer.block_id = 6 : i32} : i32
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 12 : i32, ssbuffer.main_loop = 0 : i64}

      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      // The cube scope: the group 1 is produced here, the group 0 consumed.
      %c_p0 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %c_p0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      %c_p1 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %c_p1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      %c_c0 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %c_c0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      %c_c1 = memref.alloc() {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %c_c1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>

      %10:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        // The cube block 4 produces the group 1.
        %tensor_0 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<16x32xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%tensor_0 : tensor<16x32xf32>) outs(%c_p0 : memref<16x32xf32, #hivm.address_space<ub>>)
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%tensor_0 : tensor<16x32xf32>) outs(%c_p1 : memref<16x32xf32, #hivm.address_space<ub>>)
        // The cube block 6 consumes the group 0.
        %conv_0 = hivm.hir.convert_layout %c_c0 output_shape [16, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : (memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<16x32xbf16, #hivm.address_space<cbuf>>
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 12 : i32, ssbuffer.main_loop = 0 : i64}

      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
