// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// Unit Tests for AddControlFlowCondition Pass - the preload conditions with
// three cross-core buffers.
//
// The computation blocks of a core that take part in a cross-core dependency
// are put on levels: the level of a block comes from the dependency graph.
// The blocks that open a dependency chain are on the first level of their
// core, a block is one level below its producer when the edge that links them
// crosses the cores. Every block of a level is compared with the blocks of
// the higher levels of the same core:
//
//   n == 1 : c_Bn - c_Bm <= m - n + (buffer_num - 1)
//   n >  1 : c_Bn - c_Bm <= m - n
//
// The first level of the core is the one that fills the buffers, it may run
// buffer_num - 1 rounds further ahead than the other levels.
//
// A block that takes part in no cross-core dependency has no counter, it is
// not numbered at all: it takes no place in the order.
//
// The dependency graph of this test is a single chain:
//   cube block 0 -> vector block 4 -> cube block 1 -> vector block 6
// The vector core of this test has three blocks:
//   block 4: consumes a group produced by the cube block 0, produces a group
//            consumed by the cube block 1
//   block 5: takes part in no cross-core dependency, it has no counter
//   block 6: consumes a group produced by the cube block 1
// Only the blocks that take part in a cross-core dependency are numbered, the
// block 5 owns no counter, it is not numbered at all. The preload conditions
// of the vector core are then:
//   block 4 (n=1): c_4 - c_6 <= 3 (2 - 1 + (3 - 1)) against the block 6 (m=2)
//   block 6 (m=2): nothing, it is the last block of the core
// The cube core has two blocks:
//   block 0: c_cube0 - c_cube1 <= 3 (2 - 1 + (3 - 1))

// CHECK-LABEL: func.func @preload_condition_three_buffers

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @preload_condition_three_buffers(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 3 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c8192_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i32
    scope.scope : () -> () {
      // Vector scope only process each vector core's ssbuffer ptr
      %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_5a = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5a {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %30:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {

        // The counter loads of the loop are gathered at the beginning of the
        // body of the main loop, in front of the blocks that read them. A
        // counter is read once per iteration, the blocks share the value.
        // The vector block 5 has no cross-core dependency, it has no counter,
        // its counter is never loaded.
        // CHECK:       %[[CB0V:.+]] = memref.load
        // CHECK:       %[[V4:.+]] = memref.load
        // The counter of the cube block 1 is read by the producer side
        // condition of the vector block 4, the counter of the vector block 6
        // by its preload condition.
        // CHECK:       %[[CB1:.+]] = memref.load
        // CHECK:       %[[V6:.+]] = memref.load
        // The preload condition of the vector block 4: it is on the first
        // level of the core, it may not run more than 3 rounds ahead of the
        // vector block 6 (2 - 1 + (3 - 1)). The block 5 owns no counter, it is
        // not numbered, the block 6 is on the second level of the core.
        // CHECK:       %[[DP4:.+]] = arith.subi %[[V4]], %[[V6]]
        // CHECK:       arith.constant 3 : i32
        // CHECK:       arith.cmpi sle, %[[DP4]]
        hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_11 = memref.memory_space_cast %alloc_5 {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32], ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %reshape_13a = tensor.empty() {ssbuffer.block_id = 4 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]} : tensor<8x8x16x16xf16>
        %reshape_14a = tensor.empty() {ssbuffer.block_id = 4 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]} : tensor<8x8x16x16xf16>
        // The vector block 4 produces the group 2 consumed by the cube block 1.
        %reshape_13b = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%reshape_13b : tensor<8x8x16x16xf16>) outs(%alloc0 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]}
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 4 : i32}
    // Vector block 5 takes part in no cross-core dependency, it has no
    // counter and no preload condition.

        %reshape_13 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        %reshape_14 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        hivm.hir.copy ins(%reshape_13 : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]}
        hivm.hir.copy ins(%reshape_14 : tensor<8x8x16x16xf16>) outs(%alloc0 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.intraDeps = [1 : i32, 0 : i32]}
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 5 : i32}
    // Vector block 6 consumes a group produced by cube block 1.
    // The consumer side condition is (c_cube1 - c_v6) > 0.
    // CHECK:       %[[CD:.+]] = arith.subi %[[CB1]], %[[V6]]
    // CHECK:       arith.cmpi sgt, %[[CD]], %{{.+}} : i32

        hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_16 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1, 0]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 6 : i32}
    // Cube block 0 produces a group of one buffer consumed by vector block 4.
    // The producer side condition is (c_cube0 - c_v4) < buffer_number, one
    // condition per vector core of the consumer block.
    // CHECK:       %[[CB0:.+]] = memref.load
    // CHECK:       %[[V4C0:.+]] = memref.load
    // CHECK:       %[[V4C1:.+]] = memref.load
    // CHECK:       %[[CB1C:.+]] = memref.load
    // CHECK:       %[[V6C0:.+]] = memref.load
    // CHECK:       %[[V6C1:.+]] = memref.load
    // CHECK:       %[[PD0:.+]] = arith.subi %[[CB0]], %[[V4C0]]
    // CHECK:       arith.cmpi slt, %[[PD0]]
    // CHECK:       %[[PD1:.+]] = arith.subi %[[CB0]], %[[V4C1]]
    // CHECK:       arith.cmpi slt, %[[PD1]]
    // CHECK:       arith.andi
    // The preload condition against the cube block 1: 2 - 1 + (3 - 1) = 3
    // rounds.
    // CHECK:       %[[PQ:.+]] = arith.subi %[[CB0]], %[[CB1C]]
    // CHECK:       arith.constant 3 : i32
    // CHECK:       arith.cmpi sle, %[[PQ]]
    // CHECK:       arith.cmpi slt
    // CHECK:       arith.andi
    // CHECK:       scf.if
    // The block counter has a single writer: the value that the conditions
    // of the block already loaded is reused, add one and store it.
    // CHECK:       %[[CB0N1:.+]] = arith.addi %[[CB0]]
    // CHECK:       memref.store %[[CB0N1]]
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %20:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {

        hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %30 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%30 : tensor<128x128xf32>) outs(%alloc_6 : memref<128x128xf32, #hivm.address_space<ub>>)
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 0 : i32}
    // Cube block 1 produces a group consumed by vector block 6.
    // CHECK:       %[[PD2:.+]] = arith.subi %[[CB1C]], %[[V6C0]]
    // CHECK:       arith.cmpi slt, %[[PD2]]
    // CHECK:       %[[PD3:.+]] = arith.subi %[[CB1C]], %[[V6C1]]
    // CHECK:       arith.cmpi slt, %[[PD3]]
    // CHECK:       arith.andi
    // CHECK:       arith.cmpi slt
    // CHECK:       arith.andi
    // CHECK:       scf.if
        hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %32 = hivm.hir.convert_layout %alloc_4 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %321 = hivm.hir.convert_layout %alloc_5 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %40 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%40 : tensor<128x128xf32>) outs(%alloc_7 : memref<128x128xf32, #hivm.address_space<ub>>)
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 1 : i32}
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
