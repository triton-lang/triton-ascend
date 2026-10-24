// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_v2c2v_transfer(%arg0: memref<256x256xf16>) {
    %c0_i32 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} 0 : index
    %c1_i32 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} 1 : index
    %c4_i32 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} 4 : index
    scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 {
      %alloc = memref.alloc() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : memref<256x256xf16>
      %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : memref<256x256xf16> to tensor<256x256xf16>
      %exp = math.exp %t0 {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x256xf16>

      %alloc_1 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16>
      %t1 = bufferization.to_tensor %alloc_1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16> to tensor<256x256xf16>
      %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<256x256xf32>
      %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
      %init = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>
      %mat1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%exp, %t1 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>

      %exp1 = math.exp %mat1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x256xf32>
    }
    return
  }
}




// CHECK-LABEL: func.func @test_v2c2v_transfer

// CHECK: math.exp
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = [[FLAG_1:[0-9]+]]
// CHECK: hivm.hir.copy
// CHECK: hivm.hir.sync_block_set {{.*}} flag = [[FLAG_1]]

// CHECK: hivm.hir.sync_block_wait {{.*}} flag = [[FLAG_1]]
// CHECK: linalg.matmul
// CHECK: hivm.hir.sync_block_set {{.*}} flag = [[FLAG_1]]
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = [[FLAG_2:[0-9]+]]
// CHECK: hivm.hir.fixpipe 
// CHECK: hivm.hir.sync_block_set {{.*}} flag = [[FLAG_2]]
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = [[FLAG_2]]

// CHECK: math.exp
// CHECK: hivm.hir.sync_block_set {{.*}} flag = [[FLAG_2]]
