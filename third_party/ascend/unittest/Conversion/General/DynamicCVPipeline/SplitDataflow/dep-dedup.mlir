// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  // Test 1: C->V Deduplication (Multiple Vector ops in Block 2 using the same Cube output from Block 1)
  // CHECK-LABEL: func.func @test_c2v_dedup
  func.func @test_c2v_dedup(%arg0: memref<256x256xf16>) {
    %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16>
    %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16>
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<256x256xf32>
    %init = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>

    // Block 1 (CUBE) produces %mat1
    %mat1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t0, %t0 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>

    // Block 2 (VECTOR) uses %mat1 twice
    %exp1 = math.exp %mat1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x256xf32>
    %exp2 = math.absf %mat1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x256xf32>
    return
  }
  // CHECK: %[[MATMUL:[a-z0-9_]+]] = linalg.matmul
  // CHECK: %[[ALLOC_C2V:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}
  // CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32} ins(%[[MATMUL]] : tensor<256x256xf32>) outs(%[[ALLOC_C2V]] : memref<256x256xf32, #hivm.address_space<ub>>)
  // CHECK: hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
  // CHECK: hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
  // CHECK-NOT: hivm.hir.fixpipe
  // CHECK-NOT: hivm.hir.sync_block_set
  // CHECK-NOT: hivm.hir.sync_block_wait

  // Test 2: V->C Deduplication (Multiple Cube ops in Block 2 using the same Vector output from Block 1)
  // CHECK-LABEL: func.func @test_v2c_dedup
  func.func @test_v2c_dedup(%arg0: tensor<256x256xf32>) {
    // Block 1 (VECTOR) produces %exp
    %exp = math.exp %arg0 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x256xf32>

    // Block 2 (CUBE) uses %exp twice
    %t2 = tensor.empty() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : tensor<256x256xf32>
    %transposed1 = linalg.transpose ins(%exp : tensor<256x256xf32>) outs(%t2 : tensor<256x256xf32>) permutation = [1, 0]  {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"}
    %transposed2 = linalg.transpose ins(%exp : tensor<256x256xf32>) outs(%t2 : tensor<256x256xf32>) permutation = [1, 0]  {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"}
    return
  }
  // CHECK: %[[EXP:[a-z0-9_]+]] = math.exp
  // CHECK: %[[ALLOC_V2C:[a-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
  // CHECK: hivm.hir.copy ins({{.*}} : tensor<32x16x16x8xf32>) outs(%[[ALLOC_V2C]] : memref<32x16x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}
  // CHECK: hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
  // CHECK: hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
  // CHECK-NOT: hivm.hir.copy
  // CHECK-NOT: hivm.hir.sync_block_set
  // CHECK-NOT: hivm.hir.sync_block_wait
}
