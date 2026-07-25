// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
    func.func @test_iterarg_dependencies_multi_conflict(%arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}){
    %c1_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 1 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0 : i32
    %cst_0 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %3 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_0 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %1 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_0 : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%3, %3 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>

    %91 = scf.for %arg20 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg21 = %4) -> (tensor<32x32xf32>)  : i32 {
        %6 = math.exp %arg21 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %5 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%arg21, %arg21 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>

        scf.yield {ssbuffer.core_type = "VECTOR"} %6 : tensor<32x32xf32>
    } {ssbuffer.core_type = "VECTOR"}
    return
}}



// CHECK-LABEL: func.func @test_iterarg_dependencies_multi_conflict

// CHECK: %[[MATMUL_4:[a-z0-9_]+]] = linalg.matmul

// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC]]
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[MATMUL_4]] : tensor<32x32xf32>) outs(%[[ALLOC]] : memref<32x32xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait

// CHECK: %[[ALLOC_0:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_0]]
// CHECK: memref.memory_space_cast %[[ALLOC_0]]
// CHECK: %[[TENSOR_5:[a-z0-9_]+]] = bufferization.to_tensor

// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_1]]
// CHECK: %[[ALLOC_2:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_2]]

// CHECK: hivm.hir.sync_block_set

// CHECK: scf.for {{.*}} iter_args(%[[ARG_2:[a-z0-9_]+]] = %[[TENSOR_5]])
// CHECK: arith.constant

// CHECK: arith.constant
// CHECK: tensor.reshape %[[ARG_2]]
// CHECK: tensor.empty()
// CHECK: linalg.transpose
// CHECK: arith.constant
// CHECK: %[[RESHAPE_6:[a-z0-9_]+]] = tensor.reshape
// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.copy ins(%[[RESHAPE_6]] : tensor<4x2x16x8xf32>) outs(%[[ALLOC_1]] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.convert_layout %[[ALLOC_2]]
// CHECK: memref.memory_space_cast
// CHECK: %[[TENSOR_9:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: %[[EXP_10:[a-z0-9_]+]] = math.exp %[[TENSOR_9]]
// CHECK: hivm.hir.sync_block_set
// CHECK: linalg.matmul {{.*}} ins(%[[ARG_2]], %[[ARG_2]] : tensor<32x32xf32>, tensor<32x32xf32>)
// CHECK: scf.yield {ssbuffer.core_type = "VECTOR"} %[[EXP_10]] : tensor<32x32xf32>
// CHECK: } {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 0 : i32}
// CHECK: hivm.hir.sync_block_wait
// CHECK: return
