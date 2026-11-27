// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// Both starts reach block 4. Starts are walked earliest-first, and the later
// write wins, so the sge uses block 1's counter (second iter arg). The slt
// stays on block 4's own counter.
// CHECK-LABEL: func.func @flow_opt_earliest_start
// CHECK: scf.while (%{{[^ ]+}} = %{{[^ ,]+}}, %[[CNT_A:[^ ,]+]] = %{{[^ ,]+}}, %[[CNT_B:[^ ,]+]] = %{{[^ ,]+}}, %[[CNT_MIDA:[^ ,]+]] = %{{[^ ,]+}}, %[[CNT_MIDB:[^ ,]+]] = %{{[^ ,]+}}, %[[CNT_TGT:[^ ,]+]] = %{{[^ ,]+}}, %[[DEP:[^ ,]+]] = %{{[^ ,]+}})
// CHECK: arith.cmpi sge, %[[CNT_B]], %{{.*}}
// CHECK: arith.cmpi slt, %[[CNT_TGT]], %{{.*}}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32, ssbuffer.load_store_buf_conut = 1 : i32} {
  func.func @flow_opt_earliest_start(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
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
      %cbuf_t2 = memref.alloc() {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_t2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %dup_tcb_5 = memref.alloc() {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %dup_tcb_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_t3 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_t3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_t4 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_t4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %dup_tcb_7 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %dup_tcb_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %tile_v2 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<8x4x16x16xf16>
      hivm.hir.copy ins(%tile_v2 : tensor<8x4x16x16xf16>) outs(%cbuf_t2 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [30 : i32, 1 : i32]}
      hivm.hir.copy ins(%tile_v2 : tensor<8x4x16x16xf16>) outs(%dup_tcb_5 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [30 : i32, 1 : i32]}
      %tile_v3 = bufferization.to_tensor %cbuf_t3 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [31 : i32, 0 : i32]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>> to tensor<8x4x16x16xf16>
      %tile_v4 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<8x4x16x16xf16>
      hivm.hir.copy ins(%tile_v4 : tensor<8x4x16x16xf16>) outs(%cbuf_t4 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [32 : i32, 1 : i32]}
      hivm.hir.copy ins(%tile_v4 : tensor<8x4x16x16xf16>) outs(%dup_tcb_7 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [32 : i32, 1 : i32]}
      %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      // Present so parsing loads the annotation dialect. Cross-core conditions
      // insert annotation.mark on ssbuf loads; without it the context has not
      // loaded the dialect and triton-opt aborts.
      annotation.mark %alloc {effects = ["write", "read"]} : memref<128xf32, #hivm.address_space<ub>>
      %mem = memref.memory_space_cast %alloc : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>

      scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
        // ifA: earliest start. Two cross producers (group 0) and three intra
        // producers (group 10).
        %t0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t0 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]}
        hivm.hir.copy ins(%t0 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]}
        hivm.hir.copy ins(%t0 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.intraDeps = [10 : i32, 1 : i32]}
        hivm.hir.copy ins(%t0 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.intraDeps = [10 : i32, 1 : i32]}
        hivm.hir.copy ins(%t0 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 0 : i32, ssbuffer.intraDeps = [10 : i32, 1 : i32]}

        // ifB: later start. Cross producer of group 2.
        %t1 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t1 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 1 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]}

        // ifMidA: consume group 0, produce group 1, consume intra group 10.
        %t2 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t2 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]}
        hivm.hir.copy ins(%t2 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]}
        hivm.hir.copy ins(%t2 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 2 : i32, ssbuffer.intraDeps = [10 : i32, 0 : i32]}

        // ifMidB: consume group 2, produce group 3.
        %t3 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t3 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 3 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]}
        hivm.hir.copy ins(%t3 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 3 : i32, ssbuffer.crossCoreDeps = [3 : i32, 1 : i32]}

        // ifTarget: depth 3 from both starts. Consumes groups 1 and 3.
        %t4 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%t4 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 4 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32]}
        hivm.hir.copy ins(%t4 : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 4 : i32, ssbuffer.crossCoreDeps = [3 : i32, 0 : i32]}
        scf.yield
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %cbuf_c2 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_c2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_c3 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_c3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %dup_tcb_6 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %dup_tcb_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %cbuf_c4 = memref.alloc() {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cbuf_c4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 4 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %tile_c2 = bufferization.to_tensor %cbuf_c2 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [30 : i32, 0 : i32]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>> to tensor<8x4x16x16xf16>
      %tile_c3 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x4x16x16xf16>
      hivm.hir.copy ins(%tile_c3 : tensor<8x4x16x16xf16>) outs(%cbuf_c3 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [31 : i32, 1 : i32]}
      hivm.hir.copy ins(%tile_c3 : tensor<8x4x16x16xf16>) outs(%dup_tcb_6 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32, ssbuffer.crossCoreDeps = [31 : i32, 1 : i32]}
      %tile_c4 = bufferization.to_tensor %cbuf_c4 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 4 : i32, ssbuffer.crossCoreDeps = [32 : i32, 0 : i32]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>> to tensor<8x4x16x16xf16>
      %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %mem = memref.memory_space_cast %alloc : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      scf.for %iv = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
        %tp = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%tp : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 5 : i32, ssbuffer.intraDeps = [20 : i32, 1 : i32]}
        %tc = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        hivm.hir.copy ins(%tc : tensor<128xf32>) outs(%mem : memref<128xf32>) {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [20 : i32, 0 : i32]}
        scf.yield
      } {ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
