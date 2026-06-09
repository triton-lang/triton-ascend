// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s --implicit-check-not="flag = -1" --implicit-check-not="flag = 15"

module {
  func.func @flag_reuse_over_limit() {
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf32>
    %ta = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf32>
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %mat = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%ta, %ta : tensor<32x32xf32>, tensor<32x32xf32>) outs(%fill : tensor<32x32xf32>) -> tensor<32x32xf32>
    %v2 = math.exp %mat {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %empty_3 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %c3 = linalg.transpose ins(%v2 : tensor<32x32xf32>) outs(%empty_3 : tensor<32x32xf32>) permutation = [1, 0] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"}
    %v4 = math.exp %c3 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %empty_5 = tensor.empty() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %c5 = linalg.transpose ins(%v4 : tensor<32x32xf32>) outs(%empty_5 : tensor<32x32xf32>) permutation = [1, 0] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"}
    %v6 = math.exp %c5 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %empty_7 = tensor.empty() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %c7 = linalg.transpose ins(%v6 : tensor<32x32xf32>) outs(%empty_7 : tensor<32x32xf32>) permutation = [1, 0] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
    %v8 = math.exp %c7 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %empty_9 = tensor.empty() {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %c9 = linalg.transpose ins(%v8 : tensor<32x32xf32>) outs(%empty_9 : tensor<32x32xf32>) permutation = [1, 0] {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"}
    %v10 = math.exp %c9 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %empty_11 = tensor.empty() {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %c11 = linalg.transpose ins(%v10 : tensor<32x32xf32>) outs(%empty_11 : tensor<32x32xf32>) permutation = [1, 0] {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"}
    %v12 = math.exp %c11 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %empty_13 = tensor.empty() {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %c13 = linalg.transpose ins(%v12 : tensor<32x32xf32>) outs(%empty_13 : tensor<32x32xf32>) permutation = [1, 0] {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "CUBE"}
    %v14 = math.exp %c13 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %empty_15 = tensor.empty() {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %c15 = linalg.transpose ins(%v14 : tensor<32x32xf32>) outs(%empty_15 : tensor<32x32xf32>) permutation = [1, 0] {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"}
    %v16 = math.exp %c15 {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    return
  }
}

// CHECK-LABEL: func.func @flag_reuse_over_limit
// CHECK: {{flag = }}[[REUSED_FLAG:[0-9]+]]{{$}}
// CHECK-COUNT-29: {{flag = }}[[REUSED_FLAG]]{{$}}
// CHECK: return
