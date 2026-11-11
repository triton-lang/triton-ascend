// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// The group2 consumer's matmul sits in an scf.if inside the ssbuffer.if.
// That if may not run, so the consumer decrement stays after the inner if,
// before the ssbuffer.if yield, and is not placed after the matmul.

// CHECK-LABEL: func.func @for_crosscore_matmul_nested_if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 1 : i32{{.*}}tensor<128x128xf16>, tensor<128x128xf16>
// CHECK-NOT: arith.subi
// CHECK: hivm.hir.fixpipe {{.*}}ssbuffer.block_id = 1 : i32
// CHECK: arith.subi
// CHECK: } {{.*}}ssbuffer.if = 1 : i32}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @for_crosscore_matmul_nested_if(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8192_i32 = arith.constant 8192 : i32

    scope.scope : () -> () {
      %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_ub0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_ub1 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>

      scf.for %arg14 = %c0_i32 to %c8192_i32 step %c128_i32 : i32 {
        hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast = memref.memory_space_cast %alloc_ub0 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %empty = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%empty : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]}

        hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_1 = memref.memory_space_cast %alloc_ub1 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
      } {ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    scope.scope : () -> () {
      %alloc_c = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_c {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_ub2 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_ub3 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_ub3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>

      scf.for %arg14 = %c0_i32 to %c8192_i32 step %c128_i32 : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %t0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%t0 : tensor<128x128xf32>) outs(%alloc_ub2 : memref<128x128xf32, #hivm.address_space<ub>>)

        hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %conv = hivm.hir.convert_layout %alloc_c output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %cast = memref.memory_space_cast %conv {ssbuffer.block_id = 1 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
        %tt = bufferization.to_tensor %cast restrict writable {ssbuffer.block_id = 1 : i32} : memref<128x128xf16> to tensor<128x128xf16>
        %b1 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf16>
        %c1 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf32>
        %cond = arith.cmpi sgt, %arg14, %c0_i32 {ssbuffer.block_id = 1 : i32} : i32
        %mm_if = scf.if %cond -> (tensor<128x128xf32>) {
          %mm1 = linalg.matmul {ssbuffer.block_id = 1 : i32} ins(%tt, %b1 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%c1 : tensor<128x128xf32>) -> tensor<128x128xf32>
          scf.yield %mm1 : tensor<128x128xf32>
        } else {
          scf.yield %c1 : tensor<128x128xf32>
        } {ssbuffer.block_id = 1 : i32}
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%mm_if : tensor<128x128xf32>) outs(%alloc_ub3 : memref<128x128xf32, #hivm.address_space<ub>>)
      } {ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
