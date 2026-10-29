// RUN: triton-opt --data-dependency-analysis --inter-core-transfer-and-sync %s | FileCheck %s

// A direct V->C edge must remain synchronized when the only alternative path
// reaches the same C block through another V->C edge.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @keep_direct_v2c_edge() {
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f32
    %empty1 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %seed = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%empty1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %v1 = math.exp %seed {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %v2 = math.exp %v1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    %empty3 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %m3 = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%empty3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return
  }
}

// CHECK-LABEL: func.func @keep_direct_v2c_edge

// CHECK: math.exp
// CHECK: hivm.hir.copy
// CHECK: hivm.hir.sync_block_set

// CHECK: math.exp
// CHECK: hivm.hir.copy
// CHECK: hivm.hir.sync_block_set

// CHECK: tensor.empty()
// CHECK: hivm.hir.sync_block_wait
// CHECK: bufferization.to_tensor
// CHECK: hivm.hir.sync_block_wait
// CHECK: bufferization.to_tensor
// CHECK: linalg.matmul
// CHECK: return
