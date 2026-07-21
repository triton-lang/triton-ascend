// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">}  {
  func.func @test_inner_unaligned_shape() {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f8E4M3FN
    %t0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<31x1xf8E4M3FN>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f8E4M3FN) outs(%t0 : tensor<31x1xf8E4M3FN>) -> tensor<31x1xf8E4M3FN>
    %exp = math.exp %fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<31x1xf8E4M3FN>
    %alloc = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<1x63xf8E4M3FN>
    %t1 = bufferization.to_tensor %alloc {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<1x63xf8E4M3FN> to tensor<1x63xf8E4M3FN>
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
// CHECK: %[[INSERTED_SLICE:[a-z0-9_]+]] = tensor.insert_slice %[[EXP_2]] into %[[FILL_4]][0, 0] [31, 1] [1, 1] {{.*}} : tensor<31x1xf8E4M3FN> into tensor<32x32xf8E4M3FN>

// CHECK: arith.constant
// CHECK: tensor.reshape
// CHECK: tensor.empty()
// CHECK: linalg.transpose
// CHECK: arith.constant
// CHECK: %[[RESHAPE_3:[a-z0-9_]+]] = tensor.reshape
// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc()
// CHECK: annotation.mark %[[ALLOC]]
// CHECK: hivm.hir.copy ins(%[[RESHAPE_3]] : tensor<1x2x16x32xf8E4M3FN>) outs(%[[ALLOC]] : memref<1x2x16x32xf8E4M3FN, #hivm.address_space<cbuf>>)
// CHECK: hivm.hir.sync_block_set
// CHECK: memref.alloc()
// CHECK: %[[TENSOR_6:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: tensor.empty()
// CHECK: arith.constant
// CHECK: %[[FILL_8:[a-z0-9_]+]] = linalg.fill
// CHECK: %[[ALLOC_6:[a-z0-9_]+]] = memref.alloc()
// CHECK: annotation.mark %[[ALLOC_6]]
// CHECK: hivm.hir.sync_block_wait
// CHECK: %[[MEM_9:[a-z0-9_]+]] = hivm.hir.convert_layout %[[ALLOC_6]] output_shape [31, 1] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>{{.*}}} : (memref<1x2x16x32xf8E4M3FN, #hivm.address_space<cbuf>>) -> memref<31x1xf8E4M3FN, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[MEM_9]]
// CHECK: %[[TENSOR_10:[a-z0-9_]+]] = bufferization.to_tensor %memspacecast restrict writable {{.*}} : memref<31x1xf8E4M3FN>
// CHECK: linalg.matmul {{.*}} ins(%[[TENSOR_10]], %[[TENSOR_6]] : tensor<31x1xf8E4M3FN>, tensor<1x63xf8E4M3FN>) outs(%[[FILL_8]] : tensor<31x63xf8E4M3FN>) -> tensor<31x63xf8E4M3FN>
// CHECK: return
