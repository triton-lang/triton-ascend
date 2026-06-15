// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s --implicit-check-not="flag = -1"

// Two sibling inner loops (3) and (4) live in the body of a common outer loop (2).
// The outer loop is serialized: its iter_arg %acc is threaded loop(3) -> loop(4) ->
// yield -> %acc, so iteration N+1 of loop(3) has a true data dependency on the
// loop(4) result of iteration N. loop(4)@N therefore always completes before
// loop(3)@N+1 starts, so the flag of the loop(3) transfer can be safely reused by
// the loop(4) transfer. Both transfers must collapse onto one common flag.

module {
  func.func @flag_reuse_sibling_loops(%arg0: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %c1 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c8 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 8 : i32
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %init = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %initv = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>

    %outer = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %initv) -> (tensor<32x32xf32>) : i32 {
      // ---- inner loop (3): CUBE produce -> VECTOR consume ----
      %r3 = scf.for %j = %c0 to %c8 step %c1 iter_args(%a3 = %acc) -> (tensor<32x32xf32>) : i32 {
        %e3 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
        %f3 = linalg.fill {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%e3 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %m3 = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%f3, %f3 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%f3 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %x3 = math.exp %m3 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %v3 = arith.addf %x3, %a3 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        scf.yield {ssbuffer.core_type = "VECTOR"} %v3 : tensor<32x32xf32>
      } {ssbuffer.core_type = "VECTOR"}
      // ---- inner loop (4): CUBE produce -> VECTOR consume ----
      %r4 = scf.for %k = %c0 to %c8 step %c1 iter_args(%a4 = %r3) -> (tensor<32x32xf32>) : i32 {
        %e4 = tensor.empty() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
        %f4 = linalg.fill {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%e4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %m4 = linalg.matmul {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} ins(%f4, %f4 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%f4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %x4 = math.exp %m4 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        %v4 = arith.addf %x4, %a4 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
        scf.yield {ssbuffer.core_type = "VECTOR"} %v4 : tensor<32x32xf32>
      } {ssbuffer.core_type = "VECTOR"}
      scf.yield {ssbuffer.core_type = "VECTOR"} %r4 : tensor<32x32xf32>
    } {ssbuffer.core_type = "VECTOR"}
    return
  }
}

// CHECK-LABEL: func.func @flag_reuse_sibling_loops
// CHECK: {{flag = }}[[REUSED_FLAG:[0-9]+]]{{$}}
// CHECK-COUNT-11: {{flag = }}[[REUSED_FLAG]]{{$}}
// CHECK: return
