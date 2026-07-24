// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
    func.func @test_broadcast_used_in_mmad_bias(%arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}){
    %c1_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %cst_0 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
    %3 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_0 : f32) outs(%2 : tensor<32xf32>) -> tensor<32xf32>
    %4 = math.exp %3 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>

    %0 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %1 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_0 : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>

    %91:2 = scf.for %arg20 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg21 = %1, %arg22 = %4) -> (tensor<32x32xf32>, tensor<32xf32>)  : i32 {
        %alloc_23 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf32>
        %179 = bufferization.to_tensor %alloc_23 restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf32> to tensor<32x32xf32>

        %broadcasted_26 = linalg.broadcast ins(%arg22 : tensor<32xf32>) outs(%0 : tensor<32x32xf32>) dimensions = [0]  {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.used_in_mmad_bias}
        %182 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%179, %179 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%broadcasted_26  : tensor<32x32xf32>) -> tensor<32x32xf32>

        %146 = tensor.empty() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32xf32>
        %147 = linalg.fill {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_0 : f32) outs(%146 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_45 = linalg.reduce ins(%182 : tensor<32x32xf32>) outs(%147 : tensor<32xf32>) dimensions = [0]  {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"}
            (%in: f32, %init: f32) {
                %174 = arith.addf %in, %init {ssbuffer.block_id = 4 : i32} : f32
                linalg.yield %174 {ssbuffer.block_id = 4 : i32} : f32
            }

        scf.yield {ssbuffer.core_type = "VECTOR, VECTOR"} %182, %reduced_45 : tensor<32x32xf32>, tensor<32xf32>
    } {ssbuffer.core_type = "VECTOR, VECTOR"}
    return
}}

// CHECK-LABEL: @test_broadcast_used_in_mmad_bias
// CHECK: tensor.empty()
// CHECK: linalg.fill
// CHECK: %[[EXP_2:[a-z0-9_]+]] = math.exp
// CHECK: tensor.empty()
// CHECK: %[[FILL_4:[a-z0-9_]+]] = linalg.fill

// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<32xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC]]
// CHECK: %[[ALLOC_0:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<32xf32, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_0]]
// CHECK: hivm.hir.sync_block_set

// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_1]]
// CHECK: %[[ALLOC_2:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_2]]
// CHECK: hivm.hir.sync_block_set

// CHECK: scf.for {{.*}} iter_args(%[[ARG2:[a-z0-9_]+]] = %[[FILL_4]], %[[ARG3:[a-z0-9_]+]] = %[[EXP_2]])
// CHECK: arith.constant
// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.copy ins(%[[ARG3]] : tensor<32xf32>) outs(%[[ALLOC]] : memref<32xf32, #hivm.address_space<cbuf>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: memref.alloc()
// CHECK: %[[TENSOR_6:[a-z0-9_]+]] = bufferization.to_tensor

// CHECK: hivm.hir.sync_block_wait
// CHECK: memref.memory_space_cast %[[ALLOC_0]] {{.*}} : memref<32xf32, #hivm.address_space<cbuf>> to memref<32xf32>
// CHECK: %[[TENSOR_7:[a-z0-9_]+]] = bufferization.to_tensor

// CHECK: %[[BROADCASTED:[a-z0-9_]+]] = linalg.broadcast ins(%[[TENSOR_7]] : tensor<32xf32>)
// CHECK: %[[MATMUL_8:[a-z0-9_]+]] = linalg.matmul {{.*}} ins(%[[TENSOR_6]], %[[TENSOR_6]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[BROADCASTED]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[MATMUL_8]] : tensor<32x32xf32>) outs(%[[ALLOC_1]] : memref<32x32xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait
// CHECK: memref.memory_space_cast
// CHECK: %[[TENSOR_9:[a-z0-9_]+]] = bufferization.to_tensor

// CHECK: linalg.reduce ins(%[[TENSOR_9]] : tensor<32x32xf32>)
// CHECK: hivm.hir.sync_block_set

// CHECK: scf.yield

// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.sync_block_wait
// CHECK: return
