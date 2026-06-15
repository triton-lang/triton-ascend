// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @test_forop_outer_transfer(%arg0: memref<128x128xf16>, %n: index, %init: tensor<128x128xf32>) {
    %c0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f16
    %t0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%t0 : tensor<128x128xf16>) -> tensor<128x128xf16>
    %exp = math.exp %fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (tensor<128x128xf32>) {
      %alloc = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
      %t1 = bufferization.to_tensor %alloc {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
      %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
      %mm = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%exp, %t1 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield {ssbuffer.core_type = "CUBE"} %mm : tensor<128x128xf32>
    } {ssbuffer.core_type = "CUBE"}

    return
  }
}

// CHECK-LABEL: func.func @test_forop_outer_transfer
// CHECK: %[[EXP_2:[a-z0-9_]+]] = math.exp
// CHECK: %[[CST_0:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[128, 8, 16]> : tensor<3xi64>
// CHECK: %[[RESHAPE:[a-z0-9_]+]] = tensor.reshape %[[EXP_2]](%[[CST_0]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
// CHECK: %[[EMPTY_3:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x128x16xf16>
// CHECK: %[[TRANSPOSED:[a-z0-9_]+]] = linalg.transpose ins(%[[RESHAPE]] : tensor<128x8x16xf16>) outs(%[[EMPTY_3]] : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: %[[CST_1:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[8, 8, 16, 16]> : tensor<4xi64>
// CHECK: %[[RESHAPE_2:[a-z0-9_]+]] = tensor.reshape %[[TRANSPOSED]](%[[CST_1]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.copy ins(%[[RESHAPE_2]] : tensor<8x8x16x16xf16>) outs(%[[ALLOC]] : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
// CHECK: hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[FLAG_1:[0-9]+]]
// CHECK: hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[FLAG_1]]
// CHECK: %[[ALLOC_3:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_3]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %[[MEM_4:[a-z0-9_]+]] = hivm.hir.convert_layout %[[ALLOC_3]] output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[MEM_4]] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
// CHECK: %[[TENSOR_5:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16>
// CHECK: scf.for
// CHECK: memref.alloc()
// CHECK: %[[TENSOR_7:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: %[[TENSOR_8:[a-z0-9_]+]] = tensor.empty()
// CHECK: linalg.matmul {ssbuffer.adep, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%[[TENSOR_5]], %[[TENSOR_7]] : tensor<128x128xf16>, tensor<128x128xf16>) outs(%[[TENSOR_8]] : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK: scf.yield
// CHECK: } {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"}
// CHECK: return