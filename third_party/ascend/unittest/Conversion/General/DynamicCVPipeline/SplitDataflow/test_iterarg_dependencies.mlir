// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
    func.func @iterarg_dependencies(%arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}){
    %c1_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %cst_0 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x32xf32>
    %3 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_0 : f32) outs(%2 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %4 = math.exp %3 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x32xf32>
    %0 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %1 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_0 : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %91 = scf.for %arg20 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg21 = %4) -> (tensor<64x32xf32>)  : i32 {
        %alloc_23 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<32x64xf32>
        %179 = bufferization.to_tensor %alloc_23 restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<32x64xf32>
        %182 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%179, %arg21 : tensor<32x64xf32>, tensor<64x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>

        %201 = math.exp %182 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %cst_100 = arith.constant {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f16
        %300 = tensor.empty() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x32xf32>
        %400 = linalg.fill {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_100 : f16) outs(%300 : tensor<64x32xf32>) -> tensor<64x32xf32>
        %inserted_slice = tensor.insert_slice %201 into %400[0, 0] [32, 32] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32> into tensor<64x32xf32>

        scf.yield {ssbuffer.core_type = "VECTOR"} %inserted_slice : tensor<64x32xf32>
    } {ssbuffer.core_type = "VECTOR"}
    return
}}


// CHECK-LABEL: func.func @iterarg_dependencies

// CHECK: %[[EXP_2:[a-z0-9_]+]] = math.exp
// CHECK: %[[FILL_4:[a-z0-9_]+]] = linalg.fill
// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: %[[ALLOC_0:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_0]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_1]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: %[[ALLOC_2:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_2]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2

// CHECK: scf.for {{.*}} iter_args(%[[ARG_2:[a-z0-9_]+]] = %[[EXP_2]])
// CHECK: arith.constant {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
// CHECK: %[[CST_4:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} dense<[64, 4, 8]> : tensor<3xi64>
// CHECK: %[[RESHAPE:[a-z0-9_]+]] = tensor.reshape %[[ARG_2]](%[[CST_4]]) {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x32xf32>, tensor<3xi64>) -> tensor<64x4x8xf32>
// CHECK: %[[EMPTY_6:[a-z0-9_]+]] = tensor.empty() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4x64x8xf32>
// CHECK: %[[TRANSPOSED:[a-z0-9_]+]] = linalg.transpose ins(%[[RESHAPE]] : tensor<64x4x8xf32>) outs(%[[EMPTY_6]] : tensor<4x64x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: %[[CST_5:[a-z0-9_]+]] = arith.constant {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} dense<[4, 4, 16, 8]> : tensor<4xi64>
// CHECK: %[[RESHAPE_6:[a-z0-9_]+]] = tensor.reshape %[[TRANSPOSED]](%[[CST_5]]) {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<4x64x8xf32>, tensor<4xi64>) -> tensor<4x4x16x8xf32>
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: hivm.hir.copy ins(%[[RESHAPE_6]] : tensor<4x4x16x8xf32>) outs(%[[ALLOC]] : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
// CHECK: %[[MEM_7:[a-z0-9_]+]] = hivm.hir.convert_layout %[[ALLOC_0]] output_shape [64, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : (memref<4x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<64x32xf32, #hivm.address_space<cbuf>>
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[MEM_7]] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<64x32xf32, #hivm.address_space<cbuf>> to memref<64x32xf32>
// CHECK: %[[TENSOR_8:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<64x32xf32>
// CHECK: memref.alloc()
// CHECK: %[[TENSOR_9:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: %[[MATMUL_10:[a-z0-9_]+]] = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%[[TENSOR_9]], %[[TENSOR_8]] : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[FILL_4]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} ins(%[[MATMUL_10]] : tensor<32x32xf32>) outs(%[[ALLOC_1]] : memref<32x32xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
// CHECK: %[[MEMSPACECAST_8:[a-z0-9_]+]] = memref.memory_space_cast %[[ALLOC_2]] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
// CHECK: %[[TENSOR_11:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST_8]] restrict writable {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32} : memref<32x32xf32>
// CHECK: %[[EXP_12:[a-z0-9_]+]] = math.exp %[[TENSOR_11]]
// CHECK: %[[INSERTED_SLICE:[a-z0-9_]+]] = tensor.insert_slice %[[EXP_12]]
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
// CHECK: scf.yield {ssbuffer.core_type = "VECTOR"} %[[INSERTED_SLICE]] : tensor<64x32xf32>
// CHECK: } {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 0 : i32}

// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
// CHECK: return
