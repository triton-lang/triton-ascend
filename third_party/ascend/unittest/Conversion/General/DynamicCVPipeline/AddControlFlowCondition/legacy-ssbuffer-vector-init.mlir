// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// The kernel name selects the legacy pipeline. Cube produces one group for
// Vector and appears first, so fixed pointers are available to both scopes.
// Starting a new tile must only clear the current vector sub-block's counter.
// CHECK-LABEL: func.func @_sdpa_infer_kernel(
// CHECK-NOT: memref.store
// CHECK: scope.scope
// CHECK-NOT: memref.store
// CHECK: scf.for
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
// CHECK: scope.scope
// CHECK-NOT: memref.store
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[STRIDE:.*]] = arith.constant 1024 : i64
// CHECK: %[[ID:.*]] = hivm.hir.get_sub_block_idx
// CHECK: %[[BANK:.*]] = arith.muli %[[ID]], %[[STRIDE]]
// CHECK: %[[OFFSET:.*]] = arith.constant 0 : i64
// CHECK: %[[ADDR:.*]] = arith.addi %[[OFFSET]], %[[BANK]]
// CHECK: %[[PTR:.*]] = hivm.hir.pointer_cast(%[[ADDR]]) : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: memref.store %[[ZERO]], %[[PTR]][]
// CHECK-NOT: memref.store
// CHECK: scf.for
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_sdpa_infer_kernel(%lock: memref<?xi8>, %workspace: memref<?xi8>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8192_i32 = arith.constant 8192 : i32
    scope.scope : () -> () {
      %tile = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %tile {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      scf.for %iv = %c0_i32 to %c8192_i32 step %c128_i32 : i32 {
        %a0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf16>
        %b0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf16>
        %c0 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        %mm0 = linalg.matmul {ssbuffer.block_id = 0 : i32} ins(%a0, %b0 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%c0 : tensor<128x128xf32>) -> tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%mm0 : tensor<128x128xf32>) outs(%tile : memref<128x128xf32, #hivm.address_space<ub>>)
      } {ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      %tile = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %tile {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      scf.for %iv = %c0_i32 to %c8192_i32 step %c128_i32 : i32 {
        %memspacecast = memref.memory_space_cast %tile {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
      } {ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
