// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_if_else_reuse(%arg0: memref<256x256xf16>, %arg13: i32) {
    %c0_i32 = arith.constant {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} 0 : index
    %c1_i32 = arith.constant {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} 1 : index
    %c4_i32 = arith.constant {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} 4 : index
    %c64_i32 = arith.constant {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} 64 : i32
    %10 = arith.muli %arg13, %c64_i32 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : i32
    %11 = arith.cmpi slt, %10, %c64_i32 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : i32
    scf.if %11 {
        scf.for %iv2 = %c0_i32 to %c4_i32 step %c1_i32 {
          %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<256x256xf32>
          %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16>
          %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16> to tensor<256x256xf16>
          %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
          %init = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>
          %mat1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%t0, %t0 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>

          %exp = math.exp %mat1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x256xf32>
        }
    } else {
        scf.for %iv3 = %c0_i32 to %c4_i32 step %c1_i32 {
          %empty = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<256x256xf32>
          %alloc = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16>
          %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<256x256xf16> to tensor<256x256xf16>
          %cst = arith.constant {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16
          %init = linalg.fill {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>
          %mat1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%t0, %t0 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>

          %exp = math.exp %mat1 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x256xf32>
        }
    } {ssbuffer.block_id = 5 : i32}
    return
  }
}

// CHECK-LABEL: func.func @test_if_else_reuse

// CHECK: hivm.hir.sync_block_set {{.*}} flag = 0

// CHECK: scf.if

// CHECK: memref.alloc()
// CHECK: annotation.mark
// CHECK: memref.alloc()
// CHECK: annotation.mark

// CHECK: scf.for

// CHECK: linalg.matmul
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 0
// CHECK: hivm.hir.fixpipe
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 0

// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 0
// CHECK: bufferization.to_tensor
// CHECK: math.exp
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 0

// CHECK: } else {

// CHECK: memref.alloc()
// CHECK: annotation.mark
// CHECK: memref.alloc()
// CHECK: annotation.mark

// CHECK: scf.for

// CHECK: linalg.matmul
// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 0
// CHECK: hivm.hir.fixpipe
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 0

// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 0
// CHECK: bufferization.to_tensor
// CHECK: math.exp
// CHECK: hivm.hir.sync_block_set {{.*}} flag = 0

// CHECK: } {ssbuffer.block_id = 5 : i32}

// CHECK: hivm.hir.sync_block_wait {{.*}} flag = 0
// CHECK-NEXT: return
