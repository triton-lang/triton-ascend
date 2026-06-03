// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @test_bidirectional_transfer(%arg0: memref<256x256xf16>) {
    %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16>
    %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16>
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<256x256xf32>
    %init = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>
    %mat1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t0, %t0 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
    %exp = math.exp %mat1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x256xf32>
    %t2 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<256x256xf32>
    %transposed = linalg.transpose ins(%exp : tensor<256x256xf32>) outs(%t2 : tensor<256x256xf32>) permutation = [1, 0]  {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"}
    %alloc2 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf32>
    %t3 = bufferization.to_tensor %alloc2 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf32>
    %mat2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%t3, %transposed : tensor<256x256xf32>, tensor<256x256xf32>) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>
    return
  }
}

// CHECK-LABEL: func.func @test_bidirectional_transfer
// CHECK: %[[MATMUL_3:[a-z0-9_]+]] = linalg.matmul
// CHECK: %[[ALLOC_0:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<256x256xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_0]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<256x256xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} ins(%[[MATMUL_3]] : tensor<256x256xf32>) outs(%[[ALLOC_0]] : memref<256x256xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<256x256xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_1]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<256x256xf32, #hivm.address_space<ub>>
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[ALLOC_1]] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<256x256xf32, #hivm.address_space<ub>> to memref<256x256xf32>
// CHECK: %[[TENSOR_4:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<256x256xf32>
// CHECK: %[[EXP_5:[a-z0-9_]+]] = math.exp %[[TENSOR_4]] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x256xf32>
// CHECK: %[[cst_2:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} dense<[256, 32, 8]> : tensor<3xi64>
// CHECK: %[[RESHAPE:[a-z0-9_]+]] = tensor.reshape %[[EXP_5]](%[[cst_2]]) {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<256x256xf32>, tensor<3xi64>) -> tensor<256x32x8xf32>
// CHECK: %[[EMPTY_6:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x256x8xf32>
// CHECK: %[[TRANSPOSED:[a-z0-9_]+]] = linalg.transpose ins(%[[RESHAPE]] : tensor<256x32x8xf32>) outs(%[[EMPTY_6]] : tensor<32x256x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: %[[CST_3:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} dense<[32, 16, 16, 8]> : tensor<4xi64>
// CHECK: %[[RESHAPE_4:[a-z0-9_]+]] = tensor.reshape %[[TRANSPOSED]](%[[CST_3]]) {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<32x256x8xf32>, tensor<4xi64>) -> tensor<32x16x16x8xf32>
// CHECK: %[[ALLOC_5:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<32x16x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_5]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<32x16x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.copy ins(%[[RESHAPE_4]] : tensor<32x16x16x8xf32>) outs(%[[ALLOC_5]] : memref<32x16x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
// CHECK: %[[EMPTY_9:[a-z0-9_]+]] = tensor.empty()
// CHECK: %[[ALLOC_6:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<32x16x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_6]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<32x16x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
// CHECK: %[[MEM_7:[a-z0-9_]+]] = hivm.hir.convert_layout %[[ALLOC_6]] output_shape [256, 256] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : (memref<32x16x16x8xf32, #hivm.address_space<cbuf>>) -> memref<256x256xf32, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACECAST_7:[a-z0-9_]+]] = memref.memory_space_cast %[[MEM_7]] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<256x256xf32, #hivm.address_space<cbuf>> to memref<256x256xf32>
// CHECK: %[[TENSOR_8:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST_7]] restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<256x256xf32>
// CHECK: %[[EMPTY_9:[a-z0-9_]+]] = tensor.empty()
// CHECK: linalg.transpose ins(%[[TENSOR_8]] : tensor<256x256xf32>) outs(%[[EMPTY_9]] : tensor<256x256xf32>) permutation = [1, 0]  {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"}
// CHECK: memref.alloc()
// CHECK: bufferization.to_tensor
// CHECK: linalg.matmul
// CHECK: return
