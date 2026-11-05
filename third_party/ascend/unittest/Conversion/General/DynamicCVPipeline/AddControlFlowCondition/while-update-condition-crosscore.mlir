// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// UpdateConditionInfo on scf.while: cross-core (核间) buffer conditions.
// Same dual-scope crossCoreDeps skeleton as while-update-condition-counter;
// every cross-core edge is described by the difference of the two block
// counters: the consumer waits for c_producer - c_consumer > 0, the producer
// waits for c_producer - c_consumer < N. Each block owns exactly one counter
// and only bumps that one. The counter has a single writer, so the update
// reuses the value that the conditions of the block already loaded
// (addi/store, no second load).

// CHECK-LABEL: func.func @while_update_condition_crosscore

// SSBuffer slots initialized to 0 via pointer_cast + memref.store.
// CHECK: hivm.hir.pointer_cast(%{{.*}}) : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: memref.store %{{.*}}, %{{.*}}[] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: memref.store %{{.*}}, %{{.*}}[] : memref<i32, #hivm.address_space<ssbuf>>

// CHECK: scf.while

// Vector if=5: consumes a cube block (c_cube - c_vec > 0) and produces to
// another cube block (c_vec - c_cube < 1); both and-ed with the while counter.
// CHECK: %[[LD_C5:.*]] = memref.load %{{.*}}[] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: annotation.mark %[[LD_C5]] {memref_ext.volatile}
// CHECK: %[[LD_V5:.*]] = memref.load %[[MEM_V5:.*]][] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: annotation.mark %[[LD_V5]] {memref_ext.volatile}
// CHECK: %[[DIFF_IN5:.*]] = arith.subi %[[LD_C5]], %[[LD_V5]] : i32
// CHECK: %[[SGT5:.*]] = arith.cmpi sgt, %[[DIFF_IN5]], %{{.*}} : i32
// CHECK: %[[LD_C5B:.*]] = memref.load %{{.*}}[] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: annotation.mark %[[LD_C5B]] {memref_ext.volatile}
// CHECK: %[[DIFF_OUT5:.*]] = arith.subi %[[LD_V5]], %[[LD_C5B]] : i32
// CHECK: %[[SLT5:.*]] = arith.cmpi slt, %[[DIFF_OUT5]], %{{.*}} : i32
// CHECK: %[[CROSS5:.*]] = arith.andi %[[SGT5]], %[[SLT5]]
// CHECK: %[[AND5:.*]] = arith.andi %[[CROSS5]], %{{.*}}
// CHECK: scf.if %[[AND5]]
// The block bumps only its own counter: the value it already read for the
// conditions is reused, the counter is not loaded again.
// CHECK: %[[ADD5:.*]] = arith.addi %[[LD_V5]], %{{.*}} : i32
// CHECK: memref.store %[[ADD5]], %[[MEM_V5]][] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: } {{.*}}ssbuffer.if = 5

// Vector if=6: consumer-only (c_cube - c_vec > 0), and-ed with while counter.
// CHECK: %[[LD_C6:.*]] = memref.load %{{.*}}[] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: annotation.mark %[[LD_C6]] {memref_ext.volatile}
// CHECK: %[[LD_V6:.*]] = memref.load %[[MEM_V6:.*]][] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: annotation.mark %[[LD_V6]] {memref_ext.volatile}
// CHECK: %[[DIFF_IN6:.*]] = arith.subi %[[LD_C6]], %[[LD_V6]] : i32
// CHECK: %[[SGT6:.*]] = arith.cmpi sgt, %[[DIFF_IN6]], %{{.*}} : i32
// CHECK: %[[AND6:.*]] = arith.andi %[[SGT6]], %{{.*}}
// CHECK: scf.if %[[AND6]]
// CHECK: %[[ADD6:.*]] = arith.addi %[[LD_V6]], %{{.*}} : i32
// CHECK: memref.store %[[ADD6]], %[[MEM_V6]][] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: } {{.*}}ssbuffer.if = 6

// Cube if=0: producer of the vector block, so it waits until it is less than
// N rounds ahead of the consumer. The two vector cores own separate counter
// copies, so the cube compares against both of them.
// CHECK: %[[LD_OWN0:.*]] = memref.load %[[MEM_OWN0:.*]][] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: annotation.mark %[[LD_OWN0]] {memref_ext.volatile}
// CHECK: %[[LD_V0:.*]] = memref.load %{{.*}}[] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: annotation.mark %[[LD_V0]] {memref_ext.volatile}
// CHECK: %[[DIFF0:.*]] = arith.subi %[[LD_OWN0]], %[[LD_V0]] : i32
// CHECK: %[[SLT0:.*]] = arith.cmpi slt, %[[DIFF0]], %{{.*}} : i32
// CHECK: %[[LD_V1:.*]] = memref.load %{{.*}}[] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: annotation.mark %[[LD_V1]] {memref_ext.volatile}
// CHECK: %[[DIFF1:.*]] = arith.subi %[[LD_OWN0]], %[[LD_V1]] : i32
// CHECK: %[[SLT1:.*]] = arith.cmpi slt, %[[DIFF1]], %{{.*}} : i32
// CHECK: %[[CROSS0:.*]] = arith.andi %[[SLT0]], %[[SLT1]]
// CHECK: %[[AND0:.*]] = arith.andi %[[CROSS0]], %{{.*}}
// CHECK: scf.if %[[AND0]]
// The cube reuses the value of its own counter that it already read.
// CHECK: %[[ADD0:.*]] = arith.addi %[[LD_OWN0]], %{{.*}} : i32
// CHECK: memref.store %[[ADD0]], %[[MEM_OWN0]][] : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: } {{.*}}ssbuffer.if = 0

// CHECK: scf.yield

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @while_update_condition_crosscore(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %bound = arith.extsi %arg7 : i32 to i64

    scope.scope : () -> () {
      %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_ub0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_ub1 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>

      %0 = scf.while (%arg14 = %c0_i32) : (i32) -> i32 {
        %1 = arith.extsi %arg14 {Undefined, ssbuffer.block_id = 4 : i32} : i32 to i64
        %2 = arith.cmpi slt, %1, %bound {Undefined, ssbuffer.block_id = 4 : i32} : i64
        scf.condition(%2) %arg14 : i32
      } do {
      ^bb0(%arg14: i32):
        hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast = memref.memory_space_cast %alloc_ub0 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %empty = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%empty : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]}
        %next5 = arith.addi %arg14, %c16_i32 {ssbuffer.block_id = 5 : i32} : i32

        hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_1 = memref.memory_space_cast %alloc_ub1 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %next6 = arith.addi %arg14, %c16_i32 {ssbuffer.block_id = 6 : i32} : i32
        scf.yield %next6 : i32
      } attributes {Undefined, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc_c = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_c {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_ub2 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_ub3 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>

      %0 = scf.while (%arg14 = %c0_i32) : (i32) -> i32 {
        %1 = arith.extsi %arg14 {Undefined, ssbuffer.block_id = 4 : i32} : i32 to i64
        %2 = arith.cmpi slt, %1, %bound {Undefined, ssbuffer.block_id = 4 : i32} : i64
        scf.condition(%2) %arg14 : i32
      } do {
      ^bb0(%arg14: i32):
        hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %t0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%t0 : tensor<128x128xf32>) outs(%alloc_ub2 : memref<128x128xf32, #hivm.address_space<ub>>)
        %next0 = arith.addi %arg14, %c16_i32 {ssbuffer.block_id = 0 : i32} : i32

        hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %conv = hivm.hir.convert_layout %alloc_c output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %t1 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%t1 : tensor<128x128xf32>) outs(%alloc_ub3 : memref<128x128xf32, #hivm.address_space<ub>>)
        %next1 = arith.addi %arg14, %c16_i32 {ssbuffer.block_id = 1 : i32} : i32
        scf.yield %next1 : i32
      } attributes {Undefined, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
