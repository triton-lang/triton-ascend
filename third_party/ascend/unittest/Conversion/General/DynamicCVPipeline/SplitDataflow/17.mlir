// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @tc17_unaligned_shape() {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f16
    %cst_zero = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0.0 : f16
    %t0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<100x64xf16>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%t0 : tensor<100x64xf16>) -> tensor<100x64xf16>
    %exp = math.exp %fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<100x64xf16>
    %alloc = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xf16>
    %t1 = bufferization.to_tensor %alloc {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xf16> to tensor<64x64xf16>
    %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<100x64xf32>
    %cst_cube = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
    %init = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_cube : f16) outs(%empty : tensor<100x64xf32>) -> tensor<100x64xf32>
    %mat = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%exp, %t1 : tensor<100x64xf16>, tensor<64x64xf16>) outs(%init : tensor<100x64xf32>) -> tensor<100x64xf32>
    return
  }
}

// CHECK-LABEL: func.func @tc17_unaligned_shape
// CHECK: %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f16
    // CHECK: %cst_0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f16
    // CHECK: %0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<100x64xf16>
    // CHECK: %1 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%0 : tensor<100x64xf16>) -> tensor<100x64xf16>
    // CHECK: %2 = math.exp %1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<100x64xf16>
    // CHECK: %cst_1 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f16
    // CHECK: %3 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<112x64xf16>
    // CHECK: %4 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_1 : f16) outs(%3 : tensor<112x64xf16>) -> tensor<112x64xf16>
    // CHECK: %inserted_slice = tensor.insert_slice %2 into %4[0, 0] [100, 64] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<100x64xf16> into tensor<112x64xf16>
    // CHECK: %cst_2 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[112, 4, 16]> : tensor<3xi64>
    // CHECK: %reshape = tensor.reshape %inserted_slice(%cst_2) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<112x64xf16>, tensor<3xi64>) -> tensor<112x4x16xf16>
    // CHECK: %5 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<4x112x16xf16>
    // CHECK: %transposed = linalg.transpose ins(%reshape : tensor<112x4x16xf16>) outs(%5 : tensor<4x112x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
    // CHECK: %cst_3 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} dense<[4, 7, 16, 16]> : tensor<4xi64>
    // CHECK: %reshape_4 = tensor.reshape %transposed(%cst_3) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<4x112x16xf16>, tensor<4xi64>) -> tensor<4x7x16x16xf16>
    // CHECK: %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<4x7x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32} : memref<4x7x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.copy ins(%reshape_4 : tensor<4x7x16x16xf16>) outs(%alloc : memref<4x7x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
    // CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    // CHECK: hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    // CHECK: %alloc_5 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<4x7x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<4x7x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: %6 = hivm.hir.convert_layout %alloc_5 output_shape [112, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : (memref<4x7x16x16xf16, #hivm.address_space<cbuf>>) -> memref<112x64xf16, #hivm.address_space<cbuf>>
    // CHECK: %memspacecast = memref.memory_space_cast %6 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<112x64xf16, #hivm.address_space<cbuf>> to memref<112x64xf16>
    // CHECK: %7 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32} : memref<112x64xf16> to tensor<112x64xf16>
    // CHECK: %alloc_6 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xf16>
    // CHECK: %8 = bufferization.to_tensor %alloc_6 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xf16> to tensor<64x64xf16>
    // CHECK: %9 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<100x64xf32>
    // CHECK: %cst_7 = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f16
    // CHECK: %10 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_7 : f16) outs(%9 : tensor<100x64xf32>) -> tensor<100x64xf32>
    // CHECK: %cst_8 = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    // CHECK: %11 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<112x64xf32>
    // CHECK: %12 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_8 : f32) outs(%11 : tensor<112x64xf32>) -> tensor<112x64xf32>
    // CHECK: %13 = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%7, %8 : tensor<112x64xf16>, tensor<64x64xf16>) outs(%12 : tensor<112x64xf32>) -> tensor<112x64xf32>
    // CHECK: %extracted_slice = tensor.extract_slice %13[0, 0] [100, 64] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<112x64xf32> to tensor<100x64xf32>
    // CHECK: return