// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @test_chained_matmul(%arg0: memref<128x64xf16>, %arg1: memref<64x128xf16>, %arg2: memref<128x64xf16>, %arg3: memref<64x128xf16>) {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
    %alloc_a = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16>
    %t_a = bufferization.to_tensor %alloc_a {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16> to tensor<128x64xf16>
    %alloc_b = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16>
    %t_b = bufferization.to_tensor %alloc_b {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16> to tensor<64x128xf16>
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mm1 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t_a, %t_b : tensor<128x64xf16>, tensor<64x128xf16>) outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
    %alloc_c = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16>
    %t_c = bufferization.to_tensor %alloc_c {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x64xf16> to tensor<128x64xf16>
    %alloc_d = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16>
    %t_d = bufferization.to_tensor %alloc_d {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf16> to tensor<64x128xf16>
    %mm2 = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%t_c, %t_d : tensor<128x64xf16>, tensor<64x128xf16>) outs(%mm1 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %add = arith.addf %mm1, %mm2 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    return
  }
}

// CHECK-LABEL: func.func @test_chained_matmul

// CHECK: %[[MATMUL_4:[a-z0-9_]+]] = linalg.matmul
// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_1]]

// CHECK: %[[ALLOC_2:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_2]]

// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[MATMUL_4]] : tensor<128x128xf32>) outs(%[[ALLOC_2]] : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: %[[MATMUL_7:[a-z0-9_]+]] = linalg.matmul

// CHECK: %[[ALLOC_5:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_5]]
// CHECK: %[[ALLOC_6:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_6]]
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[MATMUL_7]] : tensor<128x128xf32>) outs(%[[ALLOC_6]] : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set
// CHECK: hivm.hir.sync_block_wait
// CHECK: memref.memory_space_cast %[[ALLOC_5]]
// CHECK: %[[TENSOR_8:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: hivm.hir.sync_block_wait

// CHECK: memref.memory_space_cast %[[ALLOC_1]]
// CHECK: %[[TENSOR_9:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: arith.addf %[[TENSOR_9]], %[[TENSOR_8]]
// CHECK: return
