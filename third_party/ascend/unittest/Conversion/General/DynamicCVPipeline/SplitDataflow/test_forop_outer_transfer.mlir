// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @test_forop_outer_transfer(%arg0: memref<128x128xf16>, %n: index, %init: tensor<128x128xf32>) {
    %c0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f16
    %t0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%t0 : tensor<128x128xf16>) -> tensor<128x128xf16>
    %exp = math.exp %fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (tensor<128x128xf32>) {
      %alloc = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
      %t1 = bufferization.to_tensor %alloc {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16> to tensor<128x128xf16>
      %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
      %mm = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%exp, %t1 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield {ssbuffer.core_type = "CUBE"} %mm : tensor<128x128xf32>
    } {ssbuffer.core_type = "CUBE"}

    return
  }
}

// CHECK-LABEL: func.func @test_forop_outer_transfer

// CHECK: %[[EXP_2:[a-z0-9_]+]] = math.exp
// CHECK: arith.constant
// CHECK: tensor.reshape %[[EXP_2]]
// CHECK: tensor.empty()
// CHECK: linalg.transpose
// CHECK: arith.constant
// CHECK: %[[RESHAPE_2:[a-z0-9_]+]] = tensor.reshape


// CHECK: %[[ALLOC:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC]]

// CHECK: %[[ALLOC_3:[a-z0-9_]+]] = memref.alloc() {{.*}} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %[[ALLOC_3]]

// CHECK: hivm.hir.copy ins(%[[RESHAPE_2]] : tensor<8x8x16x16xf16>) outs(%[[ALLOC_3]] : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.convert_layout %[[ALLOC]]
// CHECK: memref.memory_space_cast
// CHECK: %[[TENSOR_5:[a-z0-9_]+]] = bufferization.to_tensor

// CHECK: scf.for

// CHECK: linalg.matmul {{.*}} ins(%[[TENSOR_5]],{{.*}})
