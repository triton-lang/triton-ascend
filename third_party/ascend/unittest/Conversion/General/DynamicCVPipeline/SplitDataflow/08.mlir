// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @tc08_ifop_transfer(%cond: i1, %arg0: memref<128x128xf16>, %init: tensor<128x128xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
    %result = scf.if %cond -> (tensor<128x128xf32>) {
      %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf16>
      %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf16> to tensor<128x128xf16>
      %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%t0 : tensor<128x128xf16>) -> tensor<128x128xf16>
      %exp = math.exp %fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
      %alloc2 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
      %t1 = bufferization.to_tensor %alloc2 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16> to tensor<128x128xf16>
      %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
      %mm = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%exp, %t1 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield {ssbuffer.core_type = "CUBE"} %mm : tensor<128x128xf32>
    } else {
      scf.yield {ssbuffer.core_type = "CUBE"} %init : tensor<128x128xf32>
    } {ssbuffer.core_type = "CUBE"}

    return
  }
}

// CHECK-LABEL: func.func @tc08_ifop_transfer
// CHECK: %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f16
// CHECK: %0 = scf.if %arg0 -> (tensor<128x128xf32>) {
// CHECK: %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf16>
// CHECK: %1 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf16> to tensor<128x128xf16>
// CHECK: %2 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%1 : tensor<128x128xf16>) -> tensor<128x128xf16>
// CHECK: %3 = math.exp %2 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf16>
// CHECK: %cst_0 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[128, 8, 16]> : tensor<3xi64>
// CHECK: %reshape = tensor.reshape %3(%cst_0) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
// CHECK: %4 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x128x16xf16>
// CHECK: %transposed = linalg.transpose ins(%reshape : tensor<128x8x16xf16>) outs(%4 : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: %cst_1 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[8, 8, 16, 16]> : tensor<4xi64>
// CHECK: %reshape_2 = tensor.reshape %transposed(%cst_1) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
// CHECK: %alloc_3 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.copy ins(%reshape_2 : tensor<8x8x16x16xf16>) outs(%alloc_3 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
// CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
// CHECK: %alloc_4 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %5 = hivm.hir.convert_layout %alloc_4 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
// CHECK: %memspacecast = memref.memory_space_cast %5 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
// CHECK: %6 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16> to tensor<128x128xf16>
// CHECK: %alloc_5 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
// CHECK: %7 = bufferization.to_tensor %alloc_5 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16> to tensor<128x128xf16>
// CHECK: %8 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
// CHECK: %9 = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%6, %7 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%8 : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK: scf.yield {ssbuffer.core_type = "CUBE"} %9 : tensor<128x128xf32>
// CHECK: } else {
// CHECK: scf.yield {ssbuffer.core_type = "CUBE"} %arg2 : tensor<128x128xf32>
// CHECK: } {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"}
// CHECK: return