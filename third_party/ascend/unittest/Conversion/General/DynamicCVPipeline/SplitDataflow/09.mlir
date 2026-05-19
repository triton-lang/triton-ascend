// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @tc09_multi_iter_args(%arg0: memref<128x128xf16>, %n: index, %init1: tensor<128x128xf32>, %init2: tensor<128x128xf32>) {
    %c0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16

    %result:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc1 = %init1, %acc2 = %init2) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
      %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
      %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16> to tensor<128x128xf16>
      %mm1 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t0, %t0 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc1 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %mm2 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t0, %t0 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc2 : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield {ssbuffer.core_type = "CUBE, CUBE"} %mm1, %mm2 : tensor<128x128xf32>, tensor<128x128xf32>
    } {ssbuffer.core_type = "CUBE, CUBE"}
    %add = arith.addf %result#0, %result#1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>

    return
  }
}

// CHECK-LABEL: func.func @tc09_multi_iter_args
// CHECK: %c0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
// CHECK: %c1 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
// CHECK: %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f16
// CHECK: %0:2 = scf.for %arg4 = %c0 to %arg1 step %c1 iter_args(%arg5 = %arg2, %arg6 = %arg3) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
// CHECK: %alloc_4 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
// CHECK: %4 = bufferization.to_tensor %alloc_4 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16> to tensor<128x128xf16>
// CHECK: %5 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%4, %4 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%arg5 : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK: %6 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%4, %4 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%arg6 : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK: scf.yield {ssbuffer.core_type = "CUBE, CUBE"} %5, %6 : tensor<128x128xf32>, tensor<128x128xf32>
// CHECK: } {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE, CUBE"}
// CHECK: %alloc = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} ins(%0#0 : tensor<128x128xf32>) outs(%alloc : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: %alloc_0 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %alloc_0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} ins(%0#1 : tensor<128x128xf32>) outs(%alloc_0 : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
// CHECK: %alloc_1 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: %memspacecast = memref.memory_space_cast %alloc_1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
// CHECK: %1 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32> to tensor<128x128xf32>
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: %alloc_2 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: %memspacecast_3 = memref.memory_space_cast %alloc_2 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
// CHECK: %2 = bufferization.to_tensor %memspacecast_3 restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32> to tensor<128x128xf32>
// CHECK: %3 = arith.addf %2, %1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK: return