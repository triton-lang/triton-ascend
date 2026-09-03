// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">}  {
  func.func @test_inline_transpose(%arg0: memref<256x256xf16>) {
    %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16>
    %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16> to tensor<256x256xf16>
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<256x256xf32>
    %init = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>
    %mat = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t0, %t0 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
    %empty1 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x256xf32>
    %transposed = linalg.transpose ins(%mat : tensor<256x256xf32>) outs(%empty1 : tensor<256x256xf32>) permutation = [1, 0]  {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"}
    %exp = math.exp %transposed {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x256xf32>
    return
  }
}

// CHECK-LABEL: func.func @test_inline_transpose


// CHECK: %[[MATMUL_3:[a-z0-9_]+]] = linalg.matmul
// CHECK: %[[ALLOC_0:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<256x256xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_0]]
// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<256x256xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_1]]

// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>{{.*}} ins(%[[MATMUL_3]]
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait
// CHECK: memref.memory_space_cast %[[ALLOC_0]]
// CHECK: %[[TENSOR_5:[a-z0-9_]+]] = bufferization.to_tensor

// CHECK: math.exp %[[TENSOR_5]]
// CHECK: return
