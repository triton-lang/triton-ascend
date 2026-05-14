// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @tc07_nested_forop(%arg0: memref<128x128xf16>, %m: index, %n: index, %init: tensor<128x128xf32>) {
    %c0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0.0 : f16
    %result = scf.for %i = %c0 to %m step %c1 iter_args(%acc1 = %init) -> (tensor<128x128xf32>) {
      %inner = scf.for %j = %c0 to %n step %c1 iter_args(%acc2 = %acc1) -> (tensor<128x128xf32>) {
        %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf16>
        %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf16> to tensor<128x128xf16>
        %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%t0 : tensor<128x128xf16>) -> tensor<128x128xf16>
        %exp = math.exp %fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
        %alloc2 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
        %t1 = bufferization.to_tensor %alloc2 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16> to tensor<128x128xf16>
        %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        %mm = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%exp, %t1 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
        scf.yield {ssbuffer.core_type = "CUBE"} %mm : tensor<128x128xf32>
      } {ssbuffer.core_type = "CUBE"}
      %exp2 = math.exp %inner {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
      scf.yield {ssbuffer.core_type = "VECTOR"} %exp2 : tensor<128x128xf32>
    } {ssbuffer.core_type = "VECTOR"}
    return
  }
}

// CHECK-LABEL: func.func @tc07_nested_forop
// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: %[[ALLOC_0:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_0]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
// CHECK: scf.for
// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_1]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %[[ALLOC_2:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_2]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: %[[FOR_1:[a-z0-9_]+]] = scf.for
// CHECK: memref.alloc()
// CHECK: bufferization.to_tensor
// CHECK: %[[FILL_5:[a-z0-9_]+]] = linalg.fill
// CHECK: %[[EXP_6:[a-z0-9_]+]] = math.exp %[[FILL_5]] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
// CHECK: %[[CST_4:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[128, 8, 16]> : tensor<3xi64>
// CHECK: %[[RESHAPE:[a-z0-9_]+]] = tensor.reshape %[[EXP_6]](%[[CST_4]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
// CHECK: %[[EMPTY_7:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x128x16xf16>
// CHECK: %[[TRANSPOSED:[a-z0-9_]+]] = linalg.transpose ins(%[[RESHAPE]] : tensor<128x8x16xf16>) outs(%[[EMPTY_7]] : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: %[[CST_5:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[8, 8, 16, 16]> : tensor<4xi64>
// CHECK: %[[RESHAPE_6:[a-z0-9_]+]] = tensor.reshape %[[TRANSPOSED]](%[[CST_5]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: hivm.hir.copy ins(%[[RESHAPE_6]] : tensor<8x8x16x16xf16>) outs(%[[ALLOC_1]] : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
// CHECK: %[[MEM_8:[a-z0-9_]+]] = hivm.hir.convert_layout %[[ALLOC_2]] output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACECAST_7:[a-z0-9_]+]] = memref.memory_space_cast %[[MEM_8]] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
// CHECK: %[[TENSOR_9:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST_7]] restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16> to tensor<128x128xf16>
// CHECK: memref.alloc()
// CHECK: %[[TENSOR_10:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: %[[EMPTY_11:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
// CHECK: linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%[[TENSOR_9]], %[[TENSOR_10]] : tensor<128x128xf16>, tensor<128x128xf16>) outs(%[[EMPTY_11]] : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: scf.yield
// CHECK: } {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.main_loop = 0 : i32, ssbuffer.vector_first}
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} ins(%[[FOR_1]] : tensor<128x128xf32>) outs(%[[ALLOC]] : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[ALLOC_0]] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
// CHECK: %[[TENSOR_13:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32> to tensor<128x128xf32>
// CHECK: math.exp %[[TENSOR_13]] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
// CHECK: scf.yield
// CHECK: } {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
// CHECK: return