// RUN: triton-opt --data-dependency-analysis --inter-core-transfer-and-sync %s | FileCheck %s

// The direct 1V->3C edge is redundant when another path reaches 3C through
// 1V->4C->3C, even though 1V->2V->3C is also present.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @drop_redundant_direct_v2c_edge() {
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f32
    %empty1 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %seed = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%empty1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %v1 = math.exp %seed {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %v2 = math.exp %v1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    %empty4 = tensor.empty() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %m4 = linalg.matmul {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%v1, %v1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%empty4 : tensor<32x32xf32>) -> tensor<32x32xf32>

    %empty3 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %m3 = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%empty3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return
  }
}

// CHECK-LABEL: func.func @drop_redundant_direct_v2c_edge

// CHECK: math.exp
// CHECK: hivm.hir.copy{{.*}}ssbuffer.transfer_id = 0
// CHECK: hivm.hir.sync_block_set{{.*}}ssbuffer.transfer_id = 0{{.*}}flag = 0

// CHECK: hivm.hir.copy{{.*}}ssbuffer.transfer_id = 1
// CHECK-NOT: hivm.hir.sync_block_set

// CHECK: math.exp
// CHECK: hivm.hir.copy{{.*}}ssbuffer.transfer_id = 2
// CHECK: hivm.hir.sync_block_set{{.*}}ssbuffer.transfer_id = 2{{.*}}flag = 1

// CHECK: tensor.empty()
// CHECK: hivm.hir.sync_block_wait{{.*}}ssbuffer.transfer_id = 0{{.*}}flag = 0
// CHECK: bufferization.to_tensor{{.*}}ssbuffer.transfer_id = 0
// CHECK: linalg.matmul


// CHECK: tensor.empty()
// CHECK-NOT: hivm.hir.sync_block_wait
// CHECK: bufferization.to_tensor{{.*}}ssbuffer.transfer_id = 1

// CHECK: hivm.hir.sync_block_wait{{.*}}ssbuffer.transfer_id = 2{{.*}}flag = 1
// CHECK: bufferization.to_tensor{{.*}}ssbuffer.transfer_id = 2
// CHECK: linalg.matmul

// CHECK: return
