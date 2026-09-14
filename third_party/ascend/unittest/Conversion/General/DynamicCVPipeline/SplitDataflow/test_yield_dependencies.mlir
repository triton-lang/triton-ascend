// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @test_yield_dependencies(%arg0: memref<128x128xf16>, %n: index, %init1: tensor<128x128xf32>, %init2: tensor<128x128xf32>) {
    %c0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1 : index

    %result:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc1 = %init1, %acc2 = %init2) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
      %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
      %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16> to tensor<128x128xf16>
      %mm1 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t0, %t0 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc1 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %mm2 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t0, %t0 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%acc2 : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield {ssbuffer.core_type = "CUBE, CUBE"} %mm1, %mm2 : tensor<128x128xf32>, tensor<128x128xf32>
    } {ssbuffer.core_type = "CUBE, CUBE"}
    %add = arith.addf %result#0, %result#1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>

    return
  }
}

// CHECK-LABEL: func.func @test_yield_dependencies

// CHECK: %[[FOR_0:[a-z0-9_]+]]:2 = scf.for
// CHECK: scf.yield

// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC]]
// CHECK: %[[ALLOC_0:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_0]]
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[FOR_0]]#0 : tensor<128x128xf32>) outs(%[[ALLOC_0]] : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: %[[ALLOC_1:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_1]]
// CHECK: %[[ALLOC_2:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<128x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC_2]]
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[FOR_0]]#1 : tensor<128x128xf32>) outs(%[[ALLOC_2]] : memref<128x128xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.sync_block_set
// CHECK: hivm.hir.sync_block_wait
// CHECK: memref.memory_space_cast %[[ALLOC]]
// CHECK: %[[TENSOR_1:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: hivm.hir.sync_block_wait
// CHECK: memref.memory_space_cast %[[ALLOC_1]]
// CHECK: %[[TENSOR_2:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: arith.addf %[[TENSOR_1]], %[[TENSOR_2]]
