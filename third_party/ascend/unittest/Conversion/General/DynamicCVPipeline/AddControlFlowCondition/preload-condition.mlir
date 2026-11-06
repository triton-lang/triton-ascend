// RUN: triton-opt --add-control-flow-condition --split-input-file %s | FileCheck %s

// Preload condition of the block dependency DAG.
//
// A block at depth 3 or 4 of the block DAG is a preload target. It waits until
// the block two positions earlier in its chain has produced
// inter_core_buf_count + 1 more rounds than it has consumed:
//
//     c_reference - c_target >= buffer_num + 1
//
// Near the end of the main loop the backlog can no longer grow, the condition
// falls back to "the loop is about to end":
//
//     total_rounds - c_target <= buffer_num
//
// total_rounds is (upper_bound - lower_bound) / step, the number of rounds of
// the main loop. The counter of the block is used on both sides of the
// comparison, so the upper bound has to be counted in rounds as well.
//
// The throttle of the producer is not relaxed for the preload targets: with
// two inter-core buffers the producer may not run two rounds ahead of its
// consumer, the physical buffer index is round % 2 and a producer that leads
// by two rounds overwrites the buffer the consumer is still reading.

// ============================================
// Test Case 1: @preload_short_chain
// ============================================
// The chain only holds two blocks, the depth of the consumer is 2 and no block
// reaches the depth of a preload target. No preload condition is generated and
// the producer throttle keeps the depth of the edge.

// CHECK-LABEL: func.func @preload_short_chain
// The vector block 5 waits for the group of the cube block 0.
// CHECK-NOT: arith.cmpi sge
// CHECK: %[[C0:.+]] = memref.load
// CHECK: %[[V5:.+]] = memref.load
// CHECK: %[[D0:.+]] = arith.subi %[[C0]], %[[V5]]
// CHECK: arith.cmpi sgt, %[[D0]]
// CHECK: scf.if
// CHECK-NOT: arith.cmpi sge
// The cube block 0 waits until it is less than one round ahead of the
// consumer, the throttle is not relaxed.
// CHECK: %[[CB0:.+]] = memref.load
// CHECK: %[[V5A:.+]] = memref.load
// CHECK: %[[PD0:.+]] = arith.subi %[[CB0]], %[[V5A]]
// CHECK: arith.constant 1 : i32
// CHECK: arith.cmpi slt, %[[PD0]]
// CHECK: %[[V5B:.+]] = memref.load
// CHECK: %[[PD1:.+]] = arith.subi %[[CB0]], %[[V5B]]
// CHECK: arith.constant 1 : i32
// CHECK: arith.cmpi slt, %[[PD1]]
// CHECK: arith.andi
// CHECK: scf.if
// No block of the chain reaches the depth of a preload target, so the preload
// condition is not generated at all.
// CHECK-NOT: arith.cmpi sge

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @preload_short_chain(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c8192_i32 = arith.constant 8192 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    scope.scope : () -> () {
      %alloc_ub0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %30:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast = memref.memory_space_cast %alloc_ub0 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc_ub2 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %20:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %t0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%t0 : tensor<128x128xf32>) outs(%alloc_ub2 : memref<128x128xf32, #hivm.address_space<ub>>)
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}

// -----

// ============================================
// Test Case 2: @preload_for_loop
// ============================================
// The chain is
//
//   C0 -> V5 -> C1 -> V6
//
// Every edge crosses the cores, so the depth of the blocks is 1, 2, 3 and 4.
// The cube block 1 (depth 3) and the vector block 6 (depth 4) are preload
// targets. The reference of the cube block 1 is the cube block 0, the
// reference of the vector block 6 is the vector block 5.

// CHECK-LABEL: func.func @preload_for_loop

// The vector block 5 consumes the group of the cube block 0 and produces the
// group of the cube block 1. The throttle of the producer is not relaxed.
// CHECK: %[[C0:.+]] = memref.load
// CHECK: %[[V5:.+]] = memref.load
// CHECK: %[[D0:.+]] = arith.subi %[[C0]], %[[V5]]
// CHECK: arith.cmpi sgt, %[[D0]]
// CHECK: %[[C1:.+]] = memref.load
// CHECK: %[[D1:.+]] = arith.subi %[[V5]], %[[C1]]
// CHECK: arith.constant 1 : i32
// CHECK: arith.cmpi slt, %[[D1]]
// CHECK: scf.if

// The vector block 6 consumes the group of the cube block 1. It is at depth 4,
// its reference is the vector block 5, two positions earlier in the chain.
// CHECK: %[[C1V:.+]] = memref.load
// CHECK: %[[V6:.+]] = memref.load
// CHECK: %[[DV:.+]] = arith.subi %[[C1V]], %[[V6]]
// CHECK: arith.cmpi sgt, %[[DV]]
// The backlog of the reference has to reach buffer_num + 1 rounds.
// CHECK: %[[V5REF:.+]] = memref.load
// CHECK: %[[PDV:.+]] = arith.subi %[[V5REF]], %[[V6]]
// CHECK: arith.cmpi sge, %[[PDV]], %c3_i32{{.*}} : i32
// The main loop is within buffer_num rounds of its end.
// CHECK: arith.subi %{{[^,]+}}, %{{[^,]+}} : i32
// CHECK: arith.divsi
// CHECK: arith.subi %{{[^,]+}}, %c2_i32{{.*}} : i32
// CHECK: arith.cmpi sge, %[[V6]], %{{[^,]+}} : i32
// CHECK: arith.ori
// CHECK: scf.if

// The cube block 0 produces the group of the vector block 5. The vector block
// 5 is not a preload target, the throttle keeps the depth of the edge.
// CHECK: %[[CB0:.+]] = memref.load
// CHECK: %[[V5A:.+]] = memref.load
// CHECK: %[[PD0:.+]] = arith.subi %[[CB0]], %[[V5A]]
// CHECK: arith.constant 1 : i32
// CHECK: arith.cmpi slt, %[[PD0]]
// CHECK: %[[V5B:.+]] = memref.load
// CHECK: %[[PD1:.+]] = arith.subi %[[CB0]], %[[V5B]]
// CHECK: arith.constant 1 : i32
// CHECK: arith.cmpi slt, %[[PD1]]
// CHECK: arith.andi
// CHECK: scf.if

// The cube block 1 produces the group of the vector block 6 and consumes the
// group of the vector block 5. The throttle of the producer is not relaxed.
// CHECK: %[[CB1:.+]] = memref.load
// CHECK: %[[V6A:.+]] = memref.load
// CHECK: %[[PD2:.+]] = arith.subi %[[CB1]], %[[V6A]]
// CHECK: arith.constant 1 : i32
// CHECK: arith.cmpi slt, %[[PD2]]
// CHECK: %[[V6B:.+]] = memref.load
// CHECK: %[[PD3:.+]] = arith.subi %[[CB1]], %[[V6B]]
// CHECK: arith.constant 1 : i32
// CHECK: arith.cmpi slt, %[[PD3]]
// CHECK: arith.andi
// CHECK: %[[V5C:.+]] = memref.load
// CHECK: %[[CD1:.+]] = arith.subi %[[V5C]], %[[CB1]]
// CHECK: arith.cmpi sgt, %[[CD1]]
// CHECK: %[[V5D:.+]] = memref.load
// CHECK: %[[CD2:.+]] = arith.subi %[[V5D]], %[[CB1]]
// CHECK: arith.cmpi sgt, %[[CD2]]
// CHECK: arith.cmpi eq, %[[V5C]], %[[V5D]]
// CHECK: arith.andi
// The cube block 1 is at depth 3, its reference is the cube block 0, two
// positions earlier in the chain.
// CHECK: %[[C0REF:.+]] = memref.load
// CHECK: %[[PDC:.+]] = arith.subi %[[C0REF]], %[[CB1]]
// CHECK: arith.cmpi sge, %[[PDC]], %c3_i32{{.*}} : i32
// The main loop is within buffer_num rounds of its end.
// CHECK: arith.subi %{{[^,]+}}, %{{[^,]+}} : i32
// CHECK: arith.divsi
// CHECK: arith.subi %{{[^,]+}}, %c2_i32{{.*}} : i32
// CHECK: arith.cmpi sge, %[[CB1]], %{{[^,]+}} : i32
// CHECK: arith.ori
// CHECK: scf.if

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @preload_for_loop(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c8192_i32 = arith.constant 8192 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    scope.scope : () -> () {
      %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_ub0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_ub1 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %30:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast = memref.memory_space_cast %alloc_ub0 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %empty = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%empty : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]}

        hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_1 = memref.memory_space_cast %alloc_ub1 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1, 0]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc_c = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_c {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_ub2 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_ub3 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %20:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %t0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%t0 : tensor<128x128xf32>) outs(%alloc_ub2 : memref<128x128xf32, #hivm.address_space<ub>>)

        hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %conv = hivm.hir.convert_layout %alloc_c output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %t1 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%t1 : tensor<128x128xf32>) outs(%alloc_ub3 : memref<128x128xf32, #hivm.address_space<ub>>)
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
