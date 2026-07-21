// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @test_unaligned_shape() {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f16
    %t0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<100x64xf16>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%t0 : tensor<100x64xf16>) -> tensor<100x64xf16>
    %exp = math.exp %fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<100x64xf16>
    %alloc = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xf16>
    %t1 = bufferization.to_tensor %alloc {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xf16> to tensor<64x64xf16>
    %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<100x64xf32>
    %cst_cube = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
    %init = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_cube : f16) outs(%empty : tensor<100x64xf32>) -> tensor<100x64xf32>
    %mat = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%exp, %t1 : tensor<100x64xf16>, tensor<64x64xf16>) outs(%init : tensor<100x64xf32>) -> tensor<100x64xf32>
    return
  }
}

// CHECK-LABEL: func.func @test_unaligned_shape
// CHECK: %[[EXP_2:[a-z0-9_]+]] = math.exp
// CHECK: %[[CST_0:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f16
// CHECK: %[[EMPTY_3:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<112x64xf16>
// CHECK: %[[FILL_4:[a-z0-9_]+]] = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%[[CST_0]] : f16) outs(%[[EMPTY_3]] : tensor<112x64xf16>) -> tensor<112x64xf16>
// CHECK: %[[INSERTED_SLICE:[a-z0-9_]+]] = tensor.insert_slice %[[EXP_2]] into %[[FILL_4]][0, 0] [100, 64] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<100x64xf16> into tensor<112x64xf16>

// CHECK: arith.constant {{.*}} dense<[112, 4, 16]> : tensor<3xi64>
// CHECK: %[[RESHAPE:[a-z0-9_]+]] = tensor.reshape %[[INSERTED_SLICE]]{{.*}} : (tensor<112x64xf16>, tensor<3xi64>) -> tensor<112x4x16xf16>
// CHECK: %[[EMPTY_5:[a-z0-9_]+]] = tensor.empty() {{.*}} : tensor<4x112x16xf16>
// CHECK: %[[TRANSPOSED:[a-z0-9_]+]] = linalg.transpose ins(%[[RESHAPE]] : tensor<112x4x16xf16>) outs(%[[EMPTY_5]] : tensor<4x112x16xf16>) permutation = [1, 0, 2]
// CHECK: arith.constant {{.*}} dense<[4, 7, 16, 16]> : tensor<4xi64>
// CHECK: %[[RESHAPE_3:[a-z0-9_]+]] = tensor.reshape %[[TRANSPOSED]]{{.*}} : (tensor<4x112x16xf16>, tensor<4xi64>) -> tensor<4x7x16x16xf16>

// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc()
// CHECK: annotation.mark %[[ALLOC]]

// CHECK: hivm.hir.copy ins(%[[RESHAPE_3]] : tensor<4x7x16x16xf16>) outs(%[[ALLOC]] : memref<4x7x16x16xf16, #hivm.address_space<cbuf>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: memref.alloc()
// CHECK: %[[TENSOR_6:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: tensor.empty()
// CHECK: arith.constant
// CHECK: %[[FILL_8:[a-z0-9_]+]] = linalg.fill

// CHECK: %[[ALLOC_6:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<4x7x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_6]] {{.*}} : memref<4x7x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.sync_block_wait

// CHECK: %[[MEM_9:[a-z0-9_]+]] = hivm.hir.convert_layout %[[ALLOC_6]] output_shape [100, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>{{.*}}} : (memref<4x7x16x16xf16, #hivm.address_space<cbuf>>) -> memref<100x64xf16, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[MEM_9]] {{.*}} : memref<100x64xf16, #hivm.address_space<cbuf>> to memref<100x64xf16>
// CHECK: %[[TENSOR_10:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable {{.*}} : memref<100x64xf16>
// CHECK: linalg.matmul {{.*}} ins(%[[TENSOR_10]], %[[TENSOR_6]] : tensor<100x64xf16>, tensor<64x64xf16>) outs(%[[FILL_8]] : tensor<100x64xf32>) -> tensor<100x64xf32>
// CHECK: return
