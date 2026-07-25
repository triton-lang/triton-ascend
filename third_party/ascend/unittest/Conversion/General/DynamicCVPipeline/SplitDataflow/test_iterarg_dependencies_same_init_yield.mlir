// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
    func.func @test_iterarg_dependencies_same_init_yield(%arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}){
    %c1_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %cst_0 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %3 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_0 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = math.exp %3 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    %cst_10 = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %20 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %30 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_10 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %100 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %10 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_10 : f32) outs(%100 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %40 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%30, %30 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%10 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %0 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %1 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_0 : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %91:2 = scf.for %arg20 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg21 = %4, %arg22 = %40) -> (tensor<32x32xf32>, tensor<32x32xf32>)  : i32 {

        %60 = math.exp %arg22 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

        %182 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%arg21, %arg21 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>

        scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %60, %182 : tensor<32x32xf32>, tensor<32x32xf32>
    } {ssbuffer.core_type = "VECTOR, CUBE"}
    return
}}


// CHECK-LABEL: func.func @test_iterarg_dependencies_same_init_yield

// CHECK: %[[EXP_2:[a-z0-9_]+]] = math.exp
// CHECK: %[[MATMUL_7:[a-z0-9_]+]] = linalg.matmul

// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC]]
// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_1]]
// CHECK: hivm.hir.sync_block_set

// CHECK: %[[ALLOC_2:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_2]]
// CHECK: %[[ALLOC_3:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_3]]
// CHECK: hivm.hir.sync_block_set

// CHECK: scf.for {{.*}} iter_args(%[[ARG_2:[a-z0-9_]+]] = %2, %[[ARG_3:[a-z0-9_]+]] = %7)

// CHECK: arith.constant
// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[ARG_3]] : tensor<32x32xf32>) outs(%[[ALLOC_2]] : memref<32x32xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: arith.constant

// CHECK: arith.constant
// CHECK: tensor.reshape %[[ARG_2]]
// CHECK: tensor.empty()
// CHECK: linalg.transpose
// CHECK: arith.constant
// CHECK: %[[RESHAPE_8:[a-z0-9_]+]] = tensor.reshape
// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.copy ins(%[[RESHAPE_8]] : tensor<4x2x16x8xf32>) outs(%[[ALLOC]] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait
// CHECK: memref.memory_space_cast %[[ALLOC_3]]
// CHECK: %[[TENSOR_12:[a-z0-9_]+]] = bufferization.to_tensor

// CHECK: %[[EXP_13:[a-z0-9_]+]] = math.exp %[[TENSOR_12]]
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.convert_layout %[[ALLOC_1]]
// CHECK: memref.memory_space_cast
// CHECK: %[[TENSOR_15:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: %[[MATMUL_16:[a-z0-9_]+]] = linalg.matmul {{.*}} ins(%[[TENSOR_15]], %[[TENSOR_15]] : tensor<32x32xf32>, tensor<32x32xf32>)

// CHECK: hivm.hir.sync_block_set
// CHECK: scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %[[EXP_13]], %[[MATMUL_16]] : tensor<32x32xf32>, tensor<32x32xf32>
// CHECK: } {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR, CUBE", ssbuffer.main_loop = 0 : i32}
// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.sync_block_wait
// CHECK: return
