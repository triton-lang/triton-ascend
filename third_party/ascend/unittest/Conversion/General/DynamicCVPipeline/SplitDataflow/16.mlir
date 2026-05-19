// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @tc16_chained_matmul(%arg0: memref<128x64xf16>, %arg1: memref<64x128xf16>, %arg2: memref<128x64xf16>, %arg3: memref<64x128xf16>) {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
    %alloc_a = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16>
    %t_a = bufferization.to_tensor %alloc_a {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16> to tensor<128x64xf16>
    %alloc_b = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16>
    %t_b = bufferization.to_tensor %alloc_b {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16> to tensor<64x128xf16>
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mm1 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t_a, %t_b : tensor<128x64xf16>, tensor<64x128xf16>) outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
    %alloc_c = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16>
    %t_c = bufferization.to_tensor %alloc_c {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16> to tensor<128x64xf16>
    %alloc_d = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16>
    %t_d = bufferization.to_tensor %alloc_d {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16> to tensor<64x128xf16>
    %mm2 = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%t_c, %t_d : tensor<128x64xf16>, tensor<64x128xf16>) outs(%mm1 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %add = arith.addf %mm1, %mm2 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    return
  }
}

// CHECK-LABEL: func.func @tc16_chained_matmul
// CHECK: %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f16
// CHECK: %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16>
// CHECK: %0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16> to tensor<128x64xf16>
// CHECK: %alloc_0 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16>
// CHECK: %1 = bufferization.to_tensor %alloc_0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16> to tensor<64x128xf16>
// CHECK: %2 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
// CHECK: %3 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK: %4 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%0, %1 : tensor<128x64xf16>, tensor<64x128xf16>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK: %alloc_1 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} ins(%4 : tensor<128x128xf32>) outs(%alloc_1 : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
// CHECK: %alloc_2 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16>
// CHECK: %5 = bufferization.to_tensor %alloc_2 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16> to tensor<128x64xf16>
// CHECK: %alloc_3 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16>
// CHECK: %6 = bufferization.to_tensor %alloc_3 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16> to tensor<64x128xf16>
// CHECK: %7 = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%5, %6 : tensor<128x64xf16>, tensor<64x128xf16>) outs(%4 : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK: %alloc_4 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} ins(%7 : tensor<128x128xf32>) outs(%alloc_4 : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
// CHECK: %alloc_5 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: %memspacecast = memref.memory_space_cast %alloc_5 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
// CHECK: %8 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32> to tensor<128x128xf32>
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: %alloc_6 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: %memspacecast_7 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
// CHECK: %9 = bufferization.to_tensor %memspacecast_7 restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32> to tensor<128x128xf32>
// CHECK: %10 = arith.addf %8, %9 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK: return