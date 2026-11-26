// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// The cross core conditions count the rounds the two cores run ahead of each
// other, and the numbers they use are the numbers of the cross core buffers.
//
// The pipeline of this test is a chain of three computation blocks that
// alternate the cores: cube 4 -> vector 11 -> cube 6. The chain holds two
// transfers, so it carries four tightly coupled buffers.
//
// The number of the cross core buffers is the number of buffers the cross core
// data passes through, the depth of the pipeline between the two cores. It is
// read from the ssbuffer.inter_core_buf_count attribute. It is the number the
// three kinds of conditions use:
//
// The throttle of the producer: the producer of a group may not run more than
// that many rounds ahead of its consumer, the producer counter minus the
// consumer counter has to stay below it.
//
// The preload bound of the cube core: the cube block 4 is on the first level
// of the core and the cube block 6 on the second one, so the cube block 4 may
// not run more than 2 - 1 + (3 - 1) = 3 rounds ahead of it.
//
// The flowOpt condition of the cube block 6: the condition holds the cube
// block 6 back until the cube block 4 has produced enough rounds:
//
//   c_cube4 >= lower_bound + step * opt_num  or  c_cube4 >= upper_bound
//
// opt_num is the number of rounds the source has to be ahead of the block. It
// is min(cross core buffer count - 1, copy count). The first term is the depth
// of the pipeline minus one: a threshold larger than the preload bound would
// have the two conditions wait for each other, so the two read the same
// number. The second term is the number of copies the producer makes for the
// consumer: the producer may not run more than that many rounds ahead of the
// consumer, so the consumer cannot wait for more rounds than that.
//
// The depth of this test is 3, the function declares
// ssbuffer.inter_core_buf_count = 3.
//
// The group of the cube block 4 is written by four fixpipe ops, so the copy
// count is 4. The copy count plays no part in the throttle: the throttle is
// the depth, 3, and not the copy count, 4.
//
// The number of the intra core buffers plays no part in it at all. The module
// declares ssbuffer.intra_buf_count = 4: the threshold would be 4 - 1 = 3 if
// the intra core buffers were counted, and the flowOpt threshold would be
// min(4 - 1, 4) = 3 instead of min(3 - 1, 4) = 2.

// The producer side of the cube block 4: the throttle of the cross core
// buffers, the depth, 3.
// CHECK: %[[THROTTLE1:[a-zA-Z0-9_]+]] = arith.constant 3 : i32
// CHECK: arith.cmpi slt, %{{.*}}, %[[THROTTLE1]] : i32
// The producer side of the vector block 11, the same depth.
// CHECK: %[[THROTTLE2:[a-zA-Z0-9_]+]] = arith.constant 3 : i32
// CHECK: arith.cmpi slt, %{{.*}}, %[[THROTTLE2]] : i32
// CHECK: %[[THROTTLE3:[a-zA-Z0-9_]+]] = arith.constant 3 : i32
// CHECK: arith.cmpi slt, %{{.*}}, %[[THROTTLE3]] : i32
// The preload bound of the cube core, the depth.
// CHECK: %[[BOUND:[a-zA-Z0-9_]+]] = arith.constant 3 : i32
// CHECK: arith.cmpi sle, %{{.*}}, %[[BOUND]] : i32
// The flowOpt condition of the cube block 6, the threshold is the depth of
// the pipeline minus one, the copy count does not lower it.
// CHECK: %[[OPT:[a-zA-Z0-9_]+]] = arith.constant 2 : i32
// CHECK: arith.muli %{{.*}}, %[[OPT]] : i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 4 : i32} {
  func.func @flow_opt_threshold_cross_core_buffers(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 3 : i32} {
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
        // The cube block 4 produces the group 1. The group is written by
        // four ops, the copy count is 4.
        %tensor_0 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<16x32xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%tensor_0 : tensor<16x32xf32>) outs(%c_p0 : memref<16x32xf32, #hivm.address_space<ub>>)
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%tensor_0 : tensor<16x32xf32>) outs(%c_p1 : memref<16x32xf32, #hivm.address_space<ub>>)
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%tensor_0 : tensor<16x32xf32>) outs(%c_p1 : memref<16x32xf32, #hivm.address_space<ub>>)
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
