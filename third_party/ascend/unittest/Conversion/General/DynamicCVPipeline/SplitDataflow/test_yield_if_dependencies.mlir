// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_yield_if_dependencies(%arg0: i32) {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %c128_i32 = arith.constant {MixUse, ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 128 : i32
    %c64_i32 = arith.constant {Undefined, ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 64 : i32
    %0 = arith.muli %arg0, %c128_i32 {MixUse, ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %1 = arith.addi %arg0, %c64_i32 {Undefined, ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.cmpi sge, %1, %0 {Undefined, ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %3 = tensor.empty() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %4 = linalg.fill {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %5 = math.exp2 %4 {DataUse, ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>

    %6 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
    %7 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32>
    %8 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32>
    %9 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%8, %8 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>

    %10:2 = scf.if %2 -> (tensor<128x128xf32>, tensor<128x128xf32>) {
      %12 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
      scf.yield {Undefined, ssbuffer.core_type = "VECTOR, VECTOR"} %12, %12 : tensor<128x128xf32>, tensor<128x128xf32>
    } else {
      scf.yield {Undefined, ssbuffer.core_type = "VECTOR, VECTOR"} %5, %9 : tensor<128x128xf32>, tensor<128x128xf32>
    } {DataUse, ssbuffer.core_type = "VECTOR, VECTOR"}
    %11 = arith.mulf %10#0, %10#1 {DataUse, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    return
  }
}

// CHECK-LABEL: @test_yield_if_dependencies

// CHECK: %[[EXP_5:[a-z0-9_]+]] = math.exp2
// CHECK: %[[MATMUL_9:[a-z0-9_]+]] = linalg.matmul
// CHECK: %[[ALLOC_0:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_0]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} ins(%[[MATMUL_9]] : tensor<128x128xf32>) outs(%[[ALLOC_0]] : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 1
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 1
// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_1]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: %[[MEMSPACECAST:[a-z0-9_]+]] = memref.memory_space_cast %[[ALLOC_1]] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
// CHECK: %[[TENSOR_10:[a-z0-9_]+]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf32>
// CHECK: %[[IF_11:[a-z0-9_]+]]:2 = scf.if
// CHECK: %[[EMPTY_13:[a-z0-9_]+]] = tensor.empty()
// CHECK: scf.yield {Undefined, ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR, VECTOR"} %[[EMPTY_13]], %[[EMPTY_13]] : tensor<128x128xf32>, tensor<128x128xf32>
// CHECK: } else {
// CHECK: scf.yield {Undefined, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR, VECTOR"} %[[EXP_5]], %[[TENSOR_10]] : tensor<128x128xf32>, tensor<128x128xf32>
// CHECK: } {DataUse, ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR, VECTOR"}
// CHECK: arith.mulf %[[IF_11]]#0, %[[IF_11]]#1
// CHECK: return
