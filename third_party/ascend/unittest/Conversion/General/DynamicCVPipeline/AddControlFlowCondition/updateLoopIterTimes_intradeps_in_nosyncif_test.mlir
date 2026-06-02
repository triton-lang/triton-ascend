// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// ============================================================================
// TC-06: insufficient intra multicache and intra dependency in no-cv-sync ifblocks
// ============================================================================
// If Vm has intra-core dependency on Vn (m>n), to achieve optimal pipeline, at most m-n+1 data flows need to exist simultaneously, which is requireNum
// If multicache count is insufficient (bufferNum < requireNum), iteration count needs to be multiplied by requireNum / bufferNum

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c8192_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i32
    scope.scope : () -> () {
      %alloc_7 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc_7 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      %alloc_8 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %memspacecast_9 = memref.memory_space_cast %alloc_8 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      %alloc_11 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_11 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_1 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %memspacecast_1 = memref.memory_space_cast %alloc_1 {ssbuffer.intraDeps = [1 : i32, 1 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      %alloc_2 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %memspacecast_2 = memref.memory_space_cast %alloc_2 {ssbuffer.intraDeps = [1 : i32, 1 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>

      // CHECK:       %[[IFCOUNT:.*]] = arith.constant 4
      // CHECK:       %[[REQUIRENUM:.*]] = arith.constant 3
      // CHECK:       %[[BUFFERNUM:.*]] = arith.constant 2
      // CHECK:       arith.subi
      // CHECK:       arith.ceildivui
      // CHECK:       arith.muli %{{.*}}, %[[REQUIRENUM]]
      // CHECK:       arith.ceildivui %{{.*}}, %[[BUFFERNUM]]
      // CHECK:       arith.addi %{{.*}}, %[[IFCOUNT]]
      // CHECK:       arith.muli
      // CHECK:       arith.addi
      // CHECK:       scf.for
      // CHECK:       iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32, %[[CNT0:.*]] = %{{.+}}, %[[CNT1:.*]] = %{{.+}}, %[[CNT2:.*]] = %{{.+}}, %[[CNT3:.*]] = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}})
      %30:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        %tensor = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128xf32>
        %c0_i32_55 = arith.constant {ssbuffer.block_id = 1 : i32} 0 : i32
        %cond = arith.cmpi eq, %arg16, %c0_i32_55 {ssbuffer.block_id = 1 : i32} : i32
        scf.if %cond {
          hivm.hir.copy ins(%tensor : tensor<128xf32>) outs(%memspacecast_1 : memref<128xf32>) {ssbuffer.block_id = 5 : i32}
        } else {
          hivm.hir.copy ins(%tensor : tensor<128xf32>) outs(%memspacecast_2 : memref<128xf32>) {ssbuffer.block_id = 5 : i32}
        } {ssbuffer.block_id = 1 : i32}

        %42 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        // CHECK:       arith.divui %[[CNT1]], %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
        %43 = arith.divui %arg16, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
        %c0_i32_15 = arith.constant {ssbuffer.block_id = 5 : i32} 0 : i32
        %c2_i32 = arith.constant {ssbuffer.block_id = 5 : i32} 2 : i32
        %44 = arith.remsi %43, %c2_i32 {ssbuffer.block_id = 5 : i32} : i32
        %45 = arith.cmpi eq, %44, %c0_i32_15 {ssbuffer.block_id = 5 : i32} : i32
        scf.if %45 {
          hivm.hir.copy ins(%42 : tensor<128xf32>) outs(%memspacecast : memref<128xf32>) {ssbuffer.block_id = 5 : i32}
        } else {
          hivm.hir.copy ins(%42 : tensor<128xf32>) outs(%memspacecast_9 : memref<128xf32>) {ssbuffer.block_id = 5 : i32}
        } {ssbuffer.block_id = 5 : i32}
        %comsumer = scf.if %45 -> (tensor<128xf32>) {
          %55 = bufferization.to_tensor %memspacecast_1 restrict writable : memref<128xf32>
          scf.yield %55 : tensor<128xf32>
        } else {
          %55 = bufferization.to_tensor %memspacecast_2 restrict writable : memref<128xf32>
          scf.yield %55 : tensor<128xf32>
        } {ssbuffer.block_id = 5 : i32, ssbuffer.intraDeps = [1 : i32, 0 : i32]}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        %reshape_40 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x4x16x16xf16>
        hivm.hir.copy ins(%reshape_40 : tensor<8x4x16x16xf16>) outs(%alloc_11 : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}

        // CHECK:       arith.divui %[[CNT2]], %c128_i32 {ssbuffer.block_id = 6 : i32} : i32
        %46 = arith.divui %arg16, %c128_i32 {ssbuffer.block_id = 6 : i32} : i32
        %c0_i32_16 = arith.constant {ssbuffer.block_id = 6 : i32} 0 : i32
        %c2_i32_6 = arith.constant {ssbuffer.block_id = 6 : i32} 2 : i32
        hivm.hir.sync_block_wait {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2

        // CHECK:       arith.divui %[[CNT3]], %c128_i32 {ssbuffer.block_id = 7 : i32} : i32
        %49 = arith.divui %arg16, %c128_i32 {ssbuffer.block_id = 7 : i32} : i32
        %c2_i32_17 = arith.constant {ssbuffer.block_id = 7 : i32} 2 : i32
        %50 = arith.remsi %49, %c2_i32_17 {ssbuffer.block_id = 7 : i32} : i32
        %c0_i32_18 = arith.constant {ssbuffer.block_id = 7 : i32} 0 : i32
        %51 = arith.cmpi eq, %50, %c0_i32_18 {ssbuffer.block_id = 7 : i32} : i32
        %52 = scf.if %51 -> (tensor<128xf32>) {
          %55 = bufferization.to_tensor %memspacecast restrict writable : memref<128xf32>
          scf.yield %55 : tensor<128xf32>
        } else {
          %55 = bufferization.to_tensor %memspacecast_9 restrict writable : memref<128xf32>
          scf.yield %55 : tensor<128xf32>
        } {ssbuffer.block_id = 7 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc_11 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.crossDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_11 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %30:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %48 = hivm.hir.convert_layout %alloc_11 output_shape [64, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.crossDeps = [1 : i32, 0 : i32], ssbuffer.transfer_id = 1 : i32} : (memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) -> memref<64x128xf16, #hivm.address_space<cbuf>>
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}