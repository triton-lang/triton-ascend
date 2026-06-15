// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">}  {
  func.func @test_inner_unaligned_shape() {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f8E4M3FN
    %t0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<31x1xf8E4M3FN>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f8E4M3FN) outs(%t0 : tensor<31x1xf8E4M3FN>) -> tensor<31x1xf8E4M3FN>
    %exp = math.exp %fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<31x1xf8E4M3FN>
    %alloc = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<1x63xf8E4M3FN>
    %t1 = bufferization.to_tensor %alloc {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<1x63xf8E4M3FN>
    %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<31x63xf8E4M3FN>
    %cst_cube = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f8E4M3FN
    %init = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_cube : f8E4M3FN) outs(%empty : tensor<31x63xf8E4M3FN>) -> tensor<31x63xf8E4M3FN>
    %mat = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%exp, %t1 : tensor<31x1xf8E4M3FN>, tensor<1x63xf8E4M3FN>) outs(%init : tensor<31x63xf8E4M3FN>) -> tensor<31x63xf8E4M3FN>
    return
  }
}


// CHECK-LABEL: func.func @test_inner_unaligned_shape
// CHECK: %[[EXP_2:[a-z0-9_]+]] = math.exp

// CHECK: %[[CST_0:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f8E4M3FN
// CHECK: %[[EMPTY_3:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf8E4M3FN>
// CHECK: %[[FILL_4:[a-z0-9_]+]] = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%[[CST_0]] : f8E4M3FN) outs(%[[EMPTY_3]] : tensor<32x32xf8E4M3FN>) -> tensor<32x32xf8E4M3FN>
// CHECK: %[[INSERTED_SLICE:[a-z0-9_]+]] = tensor.insert_slice %[[EXP_2]] into %[[FILL_4]][0, 0] [31, 1] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<31x1xf8E4M3FN> into tensor<32x32xf8E4M3FN>

// CHECK: %[[CST_1:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[32, 1, 32]> : tensor<3xi64>
// CHECK: %[[RESHAPE:[a-z0-9_]+]] = tensor.reshape %[[INSERTED_SLICE]](%[[CST_1]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<32x32xf8E4M3FN>, tensor<3xi64>) -> tensor<32x1x32xf8E4M3FN>
// CHECK: %[[EMPTY_5:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<1x32x32xf8E4M3FN>
// CHECK: %[[TRANSPOSED:[a-z0-9_]+]] = linalg.transpose ins(%[[RESHAPE]] : tensor<32x1x32xf8E4M3FN>) outs(%[[EMPTY_5]] : tensor<1x32x32xf8E4M3FN>) permutation = [1, 0, 2]  {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: %[[CST_2:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[1, 2, 16, 32]> : tensor<4xi64>
// CHECK: %[[RESHAPE_3:[a-z0-9_]+]] = tensor.reshape %[[TRANSPOSED]](%[[CST_2]]) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<1x32x32xf8E4M3FN>, tensor<4xi64>) -> tensor<1x2x16x32xf8E4M3FN>

// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<1x2x16x32xf8E4M3FN, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<1x2x16x32xf8E4M3FN, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.copy ins(%[[RESHAPE_3]] : tensor<1x2x16x32xf8E4M3FN>) outs(%[[ALLOC]] : memref<1x2x16x32xf8E4M3FN, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
// CHECK: hivm.hir.sync_block_set {{.*}} flag = [[FLAG_1:[0-9]+]]

// CHECK: %[[ALLOC_4:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<1x63xf8E4M3FN>
// CHECK: %[[TENSOR_6:[a-z0-9_]+]] = bufferization.to_tensor %[[ALLOC_4]] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<1x63xf8E4M3FN>
// CHECK: tensor.empty()
// CHECK: arith.constant
// CHECK: linalg.fill
// CHECK: %[[CST_6:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f8E4M3FN
// CHECK: %[[EMPTY_9:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x63xf8E4M3FN>
// CHECK: %[[FILL_10:[a-z0-9_]+]] = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%[[CST_6]] : f8E4M3FN) outs(%[[EMPTY_9]] : tensor<32x63xf8E4M3FN>) -> tensor<32x63xf8E4M3FN>
// CHECK: %[[INSERTED_SLICE_7:[a-z0-9_]+]] = tensor.insert_slice %[[TENSOR_6]] into %[[FILL_10]][0, 0] [1, 63] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<1x63xf8E4M3FN> into tensor<32x63xf8E4M3FN>

// CHECK: %[[CST_8:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f8E4M3FN
// CHECK: %[[EMPTY_11:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x63xf8E4M3FN>
// CHECK: %[[FILL_12:[a-z0-9_]+]] = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%[[CST_8]] : f8E4M3FN) outs(%[[EMPTY_11]] : tensor<32x63xf8E4M3FN>) -> tensor<32x63xf8E4M3FN>

// CHECK: %[[ALLOC_9:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<1x2x16x32xf8E4M3FN, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_9]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<1x2x16x32xf8E4M3FN, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = [[FLAG_1]]
// CHECK: %[[MEM_13:[a-z0-9_]+]] = hivm.hir.convert_layout %[[ALLOC_9]] output_shape [32, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : (memref<1x2x16x32xf8E4M3FN, #hivm.address_space<cbuf>>) -> memref<32x32xf8E4M3FN, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[MEM_13]] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<32x32xf8E4M3FN, #hivm.address_space<cbuf>> to memref<32x32xf8E4M3FN>
// CHECK: %[[TENSOR_14:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<32x32xf8E4M3FN>
// CHECK: %[[MATMUL_15:[a-z0-9_]+]] = linalg.matmul {ssbuffer.adep, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%[[TENSOR_14]], %[[INSERTED_SLICE_7]] : tensor<32x32xf8E4M3FN>, tensor<32x63xf8E4M3FN>) outs(%[[FILL_12]] : tensor<32x63xf8E4M3FN>) -> tensor<32x63xf8E4M3FN>
// CHECK: tensor.extract_slice %[[MATMUL_15]][0, 0] [31, 63] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.matmul_extract} : tensor<32x63xf8E4M3FN> to tensor<31x63xf8E4M3FN>
// CHECK: return

