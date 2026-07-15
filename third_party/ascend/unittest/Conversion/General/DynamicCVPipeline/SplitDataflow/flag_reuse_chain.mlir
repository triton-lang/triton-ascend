// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s --implicit-check-not="flag = -1"

module {
  func.func @flag_reuse_chain() {
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf32>
    %ta = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf32>
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %mat = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%ta, %ta : tensor<32x32xf32>, tensor<32x32xf32>) outs(%fill : tensor<32x32xf32>) -> tensor<32x32xf32>
    %exp0 = math.exp %mat {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %empty_0 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %tr = linalg.transpose ins(%exp0 : tensor<32x32xf32>) outs(%empty_0 : tensor<32x32xf32>) permutation = [1, 0] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"}
    %exp1 = math.exp %tr {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    return
  }
}

// CHECK-LABEL: func.func @flag_reuse_chain
// CHECK: {{flag = }}[[REUSED_FLAG:[0-9]+]]{{$}}
// CHECK-COUNT-5: {{flag = }}[[REUSED_FLAG]]{{$}}
// CHECK: return
