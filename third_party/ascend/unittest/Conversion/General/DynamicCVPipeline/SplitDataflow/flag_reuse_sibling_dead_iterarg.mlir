// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s --implicit-check-not="flag = -1"

// The outer loop carries a value whose yield does not depend on the iter_arg, but
// loop-carried analysis is no longer needed: loop iterations execute serially, so
// loop(3)'s transfer is released before loop(5)'s transfer is acquired in program
// order. Both sibling transfers reuse one common flag.

module {
  func.func @flag_reuse_sibling_dead_iterarg(%arg0: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %c1 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c8 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 8 : i32
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %init = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %initv = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>

    %outer = scf.for %i = %c0 to %c8 step %c1 iter_args(%carry = %initv) -> (tensor<32x32xf32>) : i32 {
      scf.for %j = %c0 to %c8 step %c1 : i32 {
        %e3 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
        %f3 = linalg.fill {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%e3 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %m3 = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%f3, %f3 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%f3 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %v3 = math.exp %m3 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
      } {ssbuffer.core_type = "VECTOR"}
      scf.for %k = %c0 to %c8 step %c1 : i32 {
        %e5 = tensor.empty() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
        %f5 = linalg.fill {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%e5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %m5 = linalg.matmul {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} ins(%f5, %f5 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%f5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %v5 = math.exp %m5 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
      } {ssbuffer.core_type = "VECTOR"}
      %next = math.exp %initv {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
      scf.yield {ssbuffer.core_type = "VECTOR"} %next : tensor<32x32xf32>
    } {ssbuffer.core_type = "VECTOR"}
    return
  }
}

// CHECK-LABEL: func.func @flag_reuse_sibling_dead_iterarg
// CHECK: {{flag = }}[[REUSED_FLAG:[0-9]+]]{{$}}
// CHECK-COUNT-11: {{flag = }}[[REUSED_FLAG]]{{$}}
// CHECK: return
