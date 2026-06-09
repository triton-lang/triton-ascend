// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @test_chained_matmul(%arg0: memref<128x64xf16>, %arg1: memref<64x128xf16>, %arg2: memref<128x64xf16>, %arg3: memref<64x128xf16>) {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
    %alloc_a = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16>
    %t_a = bufferization.to_tensor %alloc_a {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16>
    %alloc_b = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16>
    %t_b = bufferization.to_tensor %alloc_b {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16>
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mm1 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t_a, %t_b : tensor<128x64xf16>, tensor<64x128xf16>) outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
    %alloc_c = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16>
    %t_c = bufferization.to_tensor %alloc_c {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16>
    %alloc_d = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16>
    %t_d = bufferization.to_tensor %alloc_d {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16>
    %mm2 = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%t_c, %t_d : tensor<128x64xf16>, tensor<64x128xf16>) outs(%mm1 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %add = arith.addf %mm1, %mm2 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    return
  }
}

// CHECK-LABEL: func.func @test_chained_matmul
// CHECK: arith.constant
// CHECK: memref.alloc()
// CHECK: bufferization.to_tensor
// CHECK: memref.alloc()
// CHECK: bufferization.to_tensor
// CHECK: tensor.empty()
// CHECK: linalg.fill
// CHECK: %[[MATMUL_4:[a-z0-9_]+]] = linalg.matmul
// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_1]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} ins(%[[MATMUL_4]] : tensor<128x128xf32>) outs(%[[ALLOC_1]] : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
// CHECK: memref.alloc()
// CHECK: bufferization.to_tensor
// CHECK: memref.alloc()
// CHECK: bufferization.to_tensor
// CHECK: %[[MATMUL_7:[a-z0-9_]+]] = linalg.matmul {{.*}} outs(%[[MATMUL_4]] : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK: %[[ALLOC_4:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_4]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} ins(%[[MATMUL_7]] : tensor<128x128xf32>) outs(%[[ALLOC_4]] : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
// CHECK: %[[ALLOC_5:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_5]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[ALLOC_5]] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
// CHECK: %[[TENSOR_8:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32>
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: %[[ALLOC_6:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_6]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: %[[MEMSPACECAST_7:[a-z0-9_]+]] = memref.memory_space_cast %[[ALLOC_6]] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
// CHECK: %[[TENSOR_9:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST_7]] restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32>
// CHECK: arith.addf %[[TENSOR_8]], %[[TENSOR_9]]
// CHECK: return

