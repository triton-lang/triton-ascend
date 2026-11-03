// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// Unit Tests for AddControlFlowCondition Pass - Cross Core Dependencies Test
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // ============================================
  // Test Case 1: @test_multiple_CV_data_flow
  // ============================================
  // C1 C2 -> V1
  // V1 -> C3
  // C3-> V2
  // flowopt condition will be added on C3 block
  func.func @test_multiple_CV_data_flow(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
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
        // Vector block 5 consumes the groups produced by cube block 0 and cube
        // block 2, and produces the group consumed by cube block 1.
        // CHECK:       %[[C0:.+]] = memref.load
        // CHECK:       %[[V5:.+]] = memref.load
        // CHECK:       %[[D0:.+]] = arith.subi %[[C0]], %[[V5]]
        // CHECK:       arith.cmpi sgt, %[[D0]]
        // CHECK:       %[[C2:.+]] = memref.load
        // CHECK:       %[[D1:.+]] = arith.subi %[[C2]], %[[V5]]
        // CHECK:       arith.cmpi sgt, %[[D1]]
        // CHECK:       %[[C1:.+]] = memref.load
        // CHECK:       %[[D2:.+]] = arith.subi %[[V5]], %[[C1]]
        // CHECK:       arith.constant 2 : i32
        // CHECK:       arith.cmpi slt, %[[D2]]
        // CHECK:       scf.if
        hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_11 = memref.memory_space_cast %alloc_5 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %memspacecast_13 = memref.memory_space_cast %alloc_5a {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [3 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %reshape_13 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%reshape_13 : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]}
        %reshape_14 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%reshape_14 : tensor<8x8x16x16xf16>) outs(%alloc0 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]}
        // The block counter of the vector block 5 has a single writer: the
        // producer block itself.
        // CHECK:       %[[V5N:.+]] = memref.load
        // CHECK:       %[[V5N1:.+]] = arith.addi %[[V5N]]
        // CHECK:       memref.store %[[V5N1]]
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 5 : i32}

        // This block only contain an input
        // Vector block 6 consumes the group produced by cube block 1.
        // CHECK:       %[[C1V:.+]] = memref.load
        // CHECK:       %[[V6:.+]] = memref.load
        // CHECK:       %[[DV:.+]] = arith.subi %[[C1V]], %[[V6]]
        // CHECK:       arith.cmpi sgt, %[[DV]]
        // CHECK:       scf.if
        hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_16 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1, 0]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        // The block counter of the vector block 6 has a single writer: the
        // consumer block itself.
        // CHECK:       %[[V6N:.+]] = memref.load
        // CHECK:       %[[V6N1:.+]] = arith.addi %[[V6N]]
        // CHECK:       memref.store %[[V6N1]]
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 6 : i32}
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
      %alloc_6a = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6a {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %20:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        // This block contain a output
        // in CUBE scope, it need to process two ssbuffer ptr
        // Cube block 0 produces the group consumed by vector block 5. The
        // producer side condition is (c_cube0 - c_v5) < buffer_number, one
        // condition per vector core of the consumer block.
        // CHECK:       %[[CB0:.+]] = memref.load
        // CHECK:       %[[V5C0:.+]] = memref.load
        // CHECK:       %[[P0:.+]] = arith.subi %[[CB0]], %[[V5C0]]
        // CHECK:       arith.cmpi slt, %[[P0]]
        // CHECK:       %[[V5C1:.+]] = memref.load
        // CHECK:       %[[P1:.+]] = arith.subi %[[CB0]], %[[V5C1]]
        // CHECK:       arith.cmpi slt, %[[P1]]
        // CHECK:       scf.if
        hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %30 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%30 : tensor<128x128xf32>) outs(%alloc_6 : memref<128x128xf32, #hivm.address_space<ub>>)
        // The block counter of the cube block 0 has a single writer.
        // CHECK:       %[[CB0N:.+]] = memref.load
        // CHECK:       %[[CB0N1:.+]] = arith.addi %[[CB0N]]
        // CHECK:       memref.store %[[CB0N1]]
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 0 : i32}

        // Cube block 2 produces the group consumed by vector block 5.
        // CHECK:       %[[CB2:.+]] = memref.load
        // CHECK:       %[[V5D0:.+]] = memref.load
        // CHECK:       %[[P2:.+]] = arith.subi %[[CB2]], %[[V5D0]]
        // CHECK:       arith.cmpi slt, %[[P2]]
        // CHECK:       %[[V5D1:.+]] = memref.load
        // CHECK:       %[[P3:.+]] = arith.subi %[[CB2]], %[[V5D1]]
        // CHECK:       arith.cmpi slt, %[[P3]]
        // CHECK:       scf.if
        hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %31 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [3 : i32, 1 : i32]} ins(%31 : tensor<128x128xf32>) outs(%alloc_6a : memref<128x128xf32, #hivm.address_space<ub>>)
        // The block counter of the cube block 2 has a single writer.
        // CHECK:       %[[CB2N:.+]] = memref.load
        // CHECK:       %[[CB2N1:.+]] = arith.addi %[[CB2N]]
        // CHECK:       memref.store %[[CB2N1]]
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 2 : i32}

        // This block contain an input and a output
        // Cube block 1 produces the group consumed by vector block 6 and
        // consumes the group produced by vector block 5.
        // CHECK:       %[[CB1:.+]] = memref.load
        // CHECK:       %[[V6C0:.+]] = memref.load
        // CHECK:       %[[P4:.+]] = arith.subi %[[CB1]], %[[V6C0]]
        // CHECK:       arith.cmpi slt, %[[P4]]
        // CHECK:       %[[V6C1:.+]] = memref.load
        // CHECK:       %[[P5:.+]] = arith.subi %[[CB1]], %[[V6C1]]
        // CHECK:       arith.cmpi slt, %[[P5]]
        // CHECK:       %[[V5E0:.+]] = memref.load
        // CHECK:       %[[C4:.+]] = arith.subi %[[V5E0]], %[[CB1]]
        // CHECK:       arith.cmpi sgt, %[[C4]]
        // CHECK:       %[[V5E1:.+]] = memref.load
        // CHECK:       %[[C5:.+]] = arith.subi %[[V5E1]], %[[CB1]]
        // CHECK:       arith.cmpi sgt, %[[C5]]
        // The two vector cores of the producer block must be aligned.
        // CHECK:       arith.cmpi eq, %[[V5E0]], %[[V5E1]]
        // CHECK:       arith.andi
        // CHECK:       arith.cmpi sge
        // CHECK:       arith.cmpi sge
        // CHECK:       arith.ori
        // CHECK:       scf.if
        hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %32 = hivm.hir.convert_layout %alloc_4 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %321 = hivm.hir.convert_layout %alloc_5 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %40 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%40 : tensor<128x128xf32>) outs(%alloc_7 : memref<128x128xf32, #hivm.address_space<ub>>)
        // The block counter of the cube block 1 has a single writer.
        // CHECK:       %[[CB1N:.+]] = memref.load
        // CHECK:       %[[CB1N1:.+]] = arith.addi %[[CB1N]]
        // CHECK:       memref.store %[[CB1N1]]
        // CHECK:       } {hivm.matmul_limited_in_cube, ssbuffer.if = 1 : i32}
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
