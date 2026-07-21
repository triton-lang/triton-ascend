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
// CHECK: %[[INSERTED_SLICE:[a-z0-9_]+]] = tensor.insert_slice %[[EXP_2]] into {{.*}}[0, 0] [63, 7] [1, 1] {{.*}} : tensor<63x7xf32> into tensor<64x8xf32>

// CHECK: %[[EXP_7:[a-z0-9_]+]] = math.exp
// CHECK: %[[INSERTED_SLICE_2:[a-z0-9_]+]] = tensor.insert_slice %[[EXP_7]] into {{.*}}[0, 0] [7, 7] [1, 1] {{.*}} : tensor<7x7xf32> into tensor<16x8xf32>

// CHECK: arith.constant
// CHECK: tensor.reshape %[[INSERTED_SLICE]]
// CHECK: linalg.transpose
// CHECK: arith.constant
// CHECK: %[[RESHAPE_5:[a-z0-9_]+]] = tensor.reshape {{.*}} : (tensor<1x64x8xf32>, tensor<4xi64>) -> tensor<1x4x16x8xf32>

// CHECK: arith.constant
// CHECK: tensor.reshape %[[INSERTED_SLICE_2]]
// CHECK: tensor.empty()
// CHECK: linalg.transpose
// CHECK: arith.constant
// CHECK: %[[RESHAPE_10:[a-z0-9_]+]] = tensor.reshape {{.*}} : (tensor<1x16x8xf32>, tensor<4xi64>) -> tensor<1x1x16x8xf32>

// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc()
// CHECK: annotation.mark %[[ALLOC]]

// CHECK: hivm.hir.copy ins(%[[RESHAPE_5]] : tensor<1x4x16x8xf32>) outs(%[[ALLOC]] : memref<1x4x16x8xf32, #hivm.address_space<cbuf>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: %[[ALLOC_11:[a-z0-9_]+]] = memref.alloc()
// CHECK: annotation.mark %[[ALLOC_11]]

// CHECK: hivm.hir.copy ins(%[[RESHAPE_10]] : tensor<1x1x16x8xf32>) outs(%[[ALLOC_11]] : memref<1x1x16x8xf32, #hivm.address_space<cbuf>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: tensor.empty()
// CHECK: arith.constant
// CHECK: %[[FILL_13:[a-z0-9_]+]] = linalg.fill

// CHECK: %[[ALLOC_13:[a-z0-9_]+]] = memref.alloc()
// CHECK: annotation.mark %[[ALLOC_13]]
// CHECK: hivm.hir.sync_block_wait

// CHECK: hivm.hir.convert_layout %[[ALLOC_13]] output_shape [63, 7] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>{{.*}}} : (memref<1x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<63x7xf32, #hivm.address_space<cbuf>>
// CHECK: memref.memory_space_cast
// CHECK: %[[TENSOR_15:[a-z0-9_]+]] = bufferization.to_tensor

// CHECK: %[[ALLOC_14:[a-z0-9_]+]] = memref.alloc()
// CHECK: annotation.mark %[[ALLOC_14]]
// CHECK: hivm.hir.sync_block_wait

// CHECK: hivm.hir.convert_layout %[[ALLOC_14]] output_shape [7, 7] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>{{.*}}} : (memref<1x1x16x8xf32, #hivm.address_space<cbuf>>) -> memref<7x7xf32, #hivm.address_space<cbuf>>
// CHECK: memref.memory_space_cast
// CHECK: %[[TENSOR_17:[a-z0-9_]+]] = bufferization.to_tensor

// CHECK: %[[MATMUL_18:[a-z0-9_]+]] = linalg.matmul {{.*}} ins(%[[TENSOR_15]], %[[TENSOR_17]] : tensor<63x7xf32>, tensor<7x7xf32>) outs(%[[FILL_13]] : tensor<63x7xf32>) -> tensor<63x7xf32>

// CHECK: %[[ALLOC_16:[a-z0-9_]+]] = memref.alloc()
// CHECK: annotation.mark %[[ALLOC_16]]

// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[MATMUL_18]] : tensor<63x7xf32>) outs(%[[ALLOC_16]] : memref<63x7xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait
// CHECK: %[[ALLOC_17:[a-z0-9_]+]] = memref.alloc()
// CHECK: annotation.mark %[[ALLOC_17]]

// CHECK: memref.memory_space_cast %[[ALLOC_17]] {{.*}} : memref<63x7xf32, #hivm.address_space<ub>> to memref<63x7xf32>
// CHECK: %[[TENSOR_19:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: math.exp %[[TENSOR_19]]
// CHECK: return
