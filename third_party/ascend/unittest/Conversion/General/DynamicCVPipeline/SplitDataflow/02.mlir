// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @tc02_basic_c2v_transfer(%arg0: memref<128x128xf16>) {
    %c0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c128 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 128 : index
    scf.for %i = %c0 to %c128 step %c128 {
      %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
      %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16> to tensor<128x128xf16>
      %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
      %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf16>
      %init = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%empty : tensor<128x128xf16>) -> tensor<128x128xf16>
      %mat = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t0, %init : tensor<128x128xf16>, tensor<128x128xf16>) outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
      %v = arith.extf %mat {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16> to tensor<128x128xf32>
    }

    return
  }
}

// CHECK-LABEL: func.func @tc02_basic_c2v_transfer
// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16, #hivm.address_space<ub>>
// CHECK: %[[ALLOC_0:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_0]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16, #hivm.address_space<ub>>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
// CHECK: scf.for
// CHECK: memref.alloc()
// CHECK: bufferization.to_tensor
// CHECK: arith.constant
// CHECK: tensor.empty()
// CHECK: linalg.fill
// CHECK: %[[MATMUL_3:[a-z0-9_]+]] = linalg.matmul
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} ins(%[[MATMUL_3]] : tensor<128x128xf16>) outs(%[[ALLOC]] : memref<128x128xf16, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[ALLOC_0]] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16, #hivm.address_space<ub>> to memref<128x128xf16>
// CHECK: %[[TENSOR_4:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16> to tensor<128x128xf16>
// CHECK: arith.extf %[[TENSOR_4]]
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
// CHECK: } {ssbuffer.block_id = 3 : i32, ssbuffer.cube_first, ssbuffer.main_loop = 0 : i32}
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
// CHECK: return
