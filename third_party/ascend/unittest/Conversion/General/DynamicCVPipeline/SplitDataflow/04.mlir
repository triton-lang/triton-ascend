// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @tc04_single_v_multi_c(%arg0: memref<128x128xf16>) {
    %c0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c128 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 128 : index
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f16
    scf.for %i = %c0 to %c128 step %c128 {
      %t0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
      %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%t0 : tensor<128x128xf16>) -> tensor<128x128xf16>
      %exp = math.exp %fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
      %alloc1 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
      %t1 = bufferization.to_tensor %alloc1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16> to tensor<128x128xf16>
      %empty1 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf16>
      %mat1 = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%exp, %t1 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%empty1 : tensor<128x128xf16>) -> tensor<128x128xf16>
      %alloc2 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
      %t2 = bufferization.to_tensor %alloc2 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16> to tensor<128x128xf16>
      %empty2 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf16>
      %mat2 = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%exp, %t2 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%empty2 : tensor<128x128xf16>) -> tensor<128x128xf16>
    }

    return
  }
}

// CHECK-LABEL: func.func @tc04_single_v_multi_c
// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %[[ALLOC_0:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_0]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_1]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %[[ALLOC_2:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_2]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
// CHECK: scf.for
// CHECK: %[[EXP_2:[a-z0-9_]+]] = math.exp
// CHECK: %[[CST_3:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[128, 8, 16]> : tensor<3xi64>
// CHECK: %[[RESHAPE:[a-z0-9_]+]] = tensor.reshape %[[EXP_2]](%[[CST_3]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
// CHECK: %[[EMPTY_3:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x128x16xf16>
// CHECK: %[[TRANSPOSED:[a-z0-9_]+]] = linalg.transpose ins(%[[RESHAPE]] : tensor<128x8x16xf16>) outs(%[[EMPTY_3]] : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: %[[CST_4:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[8, 8, 16, 16]> : tensor<4xi64>
// CHECK: %[[RESHAPE_5:[a-z0-9_]+]] = tensor.reshape %[[TRANSPOSED]](%[[CST_4]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: hivm.hir.copy ins(%[[RESHAPE_5]] : tensor<8x8x16x16xf16>) outs(%[[ALLOC]] : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
// CHECK: hivm.hir.copy ins(%[[RESHAPE_5]] : tensor<8x8x16x16xf16>) outs(%[[ALLOC_1]] : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
// CHECK: %[[MEM_4:[a-z0-9_]+]] = hivm.hir.convert_layout %[[ALLOC_0]] output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[MEM_4]] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
// CHECK: %[[TENSOR_5:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16> to tensor<128x128xf16>
// CHECK: memref.alloc()
// CHECK: %[[TENSOR_6:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: %[[TENSOR_7:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf16>
// CHECK: linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%[[TENSOR_5]], %[[TENSOR_6]] : tensor<128x128xf16>, tensor<128x128xf16>) outs(%[[TENSOR_7]] : tensor<128x128xf16>) -> tensor<128x128xf16>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
// CHECK: %[[MEM_9:[a-z0-9_]+]] = hivm.hir.convert_layout %[[ALLOC_2]] output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACECAST_7:[a-z0-9_]+]] = memref.memory_space_cast %[[MEM_9]] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
// CHECK: %[[TENSOR_10:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST_7]] restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<128x128xf16> to tensor<128x128xf16>
// CHECK: memref.alloc()
// CHECK: %[[TENSOR_11:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: %[[EMPTY_12:[a-z0-9_]+]] = tensor.empty()
// CHECK: linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%[[TENSOR_10]], %[[TENSOR_11]] : tensor<128x128xf16>, tensor<128x128xf16>) outs(%[[EMPTY_12]] : tensor<128x128xf16>) -> tensor<128x128xf16>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
// CHECK: } {ssbuffer.block_id = 4 : i32, ssbuffer.main_loop = 0 : i32, ssbuffer.vector_first}
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: return