// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">}  {
  func.func @test_multi_unaligned_shape() {
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f32
    %t0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<63x7xf32>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%t0 : tensor<63x7xf32>) -> tensor<63x7xf32>
    %exp = math.exp %fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<63x7xf32>

    %t2 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<7x7xf32>
    %fill_2 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%t2 : tensor<7x7xf32>) -> tensor<7x7xf32>
    %exp_2 = math.exp %fill_2 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<7x7xf32>

    %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<63x7xf32>
    %cst_cube = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f32
    %init = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_cube : f32) outs(%empty : tensor<63x7xf32>) -> tensor<63x7xf32>
    %mat = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%exp, %exp_2 : tensor<63x7xf32>, tensor<7x7xf32>) outs(%init : tensor<63x7xf32>) -> tensor<63x7xf32>

    %exp_3 = math.exp %mat {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<63x7xf32>
    return
  }
}

// CHECK-LABEL: func.func @test_multi_unaligned_shape

// CHECK: %[[EXP_2:[a-z0-9_]+]] = math.exp
// CHECK: %[[CST_0:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
// CHECK: %[[EMPTY_3:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x16xf32>
// CHECK: %[[FILL_4:[a-z0-9_]+]] = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%[[CST_0]] : f32) outs(%[[EMPTY_3]] : tensor<64x16xf32>) -> tensor<64x16xf32>
// CHECK: %[[INSERTED_SLICE:[a-z0-9_]+]] = tensor.insert_slice %[[EXP_2]] into %[[FILL_4]][0, 0] [63, 7] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<63x7xf32> into tensor<64x16xf32>

// CHECK: %[[EXP_7:[a-z0-9_]+]] = math.exp
// CHECK: %[[CST_1:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
// CHECK: %[[EMPTY_8:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x8xf32>
// CHECK: %[[FILL_9:[a-z0-9_]+]] = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%[[CST_1]] : f32) outs(%[[EMPTY_8]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK: %[[INSERTED_SLICE_2:[a-z0-9_]+]] = tensor.insert_slice %[[EXP_7]] into %[[FILL_9]][0, 0] [7, 7] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<7x7xf32> into tensor<16x8xf32>

// CHECK: %[[CST_3:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[64, 2, 8]> : tensor<3xi64>
// CHECK: %[[RESHAPE:[a-z0-9_]+]] = tensor.reshape %[[INSERTED_SLICE]](%[[CST_3]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x16xf32>, tensor<3xi64>) -> tensor<64x2x8xf32>
// CHECK: %[[EMPTY_10:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x64x8xf32>
// CHECK: %[[TRANSPOSED:[a-z0-9_]+]] = linalg.transpose ins(%[[RESHAPE]] : tensor<64x2x8xf32>) outs(%[[EMPTY_10]] : tensor<2x64x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: %[[CST_4:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[2, 4, 16, 8]> : tensor<4xi64>
// CHECK: %[[RESHAPE_5:[a-z0-9_]+]] = tensor.reshape %[[TRANSPOSED]](%[[CST_4]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<2x64x8xf32>, tensor<4xi64>) -> tensor<2x4x16x8xf32>

// CHECK: %[[CST_6:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[16, 1, 8]> : tensor<3xi64>
// CHECK: %[[RESHAPE_7:[a-z0-9_]+]] = tensor.reshape %[[INSERTED_SLICE_2]](%[[CST_6]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<16x8xf32>, tensor<3xi64>) -> tensor<16x1x8xf32>
// CHECK: %[[EMPTY_11:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x16x8xf32>
// CHECK: %[[TRANSPOSED_8:[a-z0-9_]+]] = linalg.transpose ins(%[[RESHAPE_7]] : tensor<16x1x8xf32>) outs(%[[EMPTY_11]] : tensor<1x16x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: %[[CST_9:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[1, 1, 16, 8]> : tensor<4xi64>
// CHECK: %[[RESHAPE_10:[a-z0-9_]+]] = tensor.reshape %[[TRANSPOSED_8]](%[[CST_9]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<1x16x8xf32>, tensor<4xi64>) -> tensor<1x1x16x8xf32>

// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<2x4x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<2x4x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.copy ins(%[[RESHAPE_5]] : tensor<2x4x16x8xf32>) outs(%[[ALLOC]] : memref<2x4x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
// CHECK: hivm.hir.sync_block_set {{.*}} flag = [[FLAG_1:[0-9]+]]

// CHECK: %[[ALLOC_11:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<1x1x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_11]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<1x1x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.copy ins(%[[RESHAPE_10]] : tensor<1x1x16x8xf32>) outs(%[[ALLOC_11]] : memref<1x1x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}
// CHECK: hivm.hir.sync_block_set {{.*}} flag = [[FLAG_2:[0-9]+]]
// CHECK: arith.constant
// CHECK: tensor.empty()
// CHECK: linalg.fill
// CHECK: %[[CST_14:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
// CHECK: %[[EMPTY_16:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x8xf32>
// CHECK: %[[FILL_17:[a-z0-9_]+]] = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%[[CST_14]] : f32) outs(%[[EMPTY_16]] : tensor<64x8xf32>) -> tensor<64x8xf32>

// CHECK: %[[ALLOC_15:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<2x4x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_15]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<2x4x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = [[FLAG_1]]
// CHECK: %[[MEM_18:[a-z0-9_]+]] = hivm.hir.convert_layout %[[ALLOC_15]] output_shape [64, 16] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : (memref<2x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<64x16xf32, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[MEM_18]] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<64x16xf32, #hivm.address_space<cbuf>> to memref<64x16xf32>
// CHECK: %[[TENSOR_19:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<64x16xf32>

// CHECK: %[[ALLOC_16:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<1x1x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_16]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<1x1x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = [[FLAG_2]]
// CHECK: %[[MEM_20:[a-z0-9_]+]] = hivm.hir.convert_layout %[[ALLOC_16]] output_shape [16, 8] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : (memref<1x1x16x8xf32, #hivm.address_space<cbuf>>) -> memref<16x8xf32, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACECAST_17:[a-z0-9_]+]] = memref.memory_space_cast %[[MEM_20]] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<16x8xf32, #hivm.address_space<cbuf>> to memref<16x8xf32>
// CHECK: %[[TENSOR_21:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST_17]] restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<16x8xf32>

// CHECK: %[[MATMUL_22:[a-z0-9_]+]] = linalg.matmul {ssbuffer.adep, ssbuffer.bdep, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%[[TENSOR_19]], %[[TENSOR_21]] : tensor<64x16xf32>, tensor<16x8xf32>) outs(%[[FILL_17]] : tensor<64x8xf32>) -> tensor<64x8xf32>
// CHECK: %[[EXTRACTED_SLICE:[a-z0-9_]+]] = tensor.extract_slice %[[MATMUL_22]][0, 0] [63, 7] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.matmul_extract} : tensor<64x8xf32> to tensor<63x7xf32>

// CHECK: %[[ALLOC_18:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 2 : i32} : memref<63x7xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_18]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 2 : i32} : memref<63x7xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 2 : i32} ins(%[[EXTRACTED_SLICE]] : tensor<63x7xf32>) outs(%[[ALLOC_18]] : memref<63x7xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 3

// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 3
// CHECK: %[[ALLOC_19:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 2 : i32} : memref<63x7xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_19]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 2 : i32} : memref<63x7xf32, #hivm.address_space<ub>>
// CHECK: %[[MEMSPACECAST_20:[a-z0-9_]+]] = memref.memory_space_cast %[[ALLOC_19]] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 2 : i32} : memref<63x7xf32, #hivm.address_space<ub>> to memref<63x7xf32>
// CHECK: %[[TENSOR_23:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST_20]] restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 2 : i32} : memref<63x7xf32>
// CHECK: math.exp %[[TENSOR_23]] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<63x7xf32>

// CHECK: return


