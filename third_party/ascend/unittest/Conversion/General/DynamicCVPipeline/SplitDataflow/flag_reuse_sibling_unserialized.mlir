// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s --implicit-check-not="flag = -1"

// Same sibling-loop shape as flag_reuse_sibling_loops.mlir, with an outer loop (2)
// that carries nothing. Loop iterations still execute serially, so loop(3)'s
// transfer is fully released (its post-loop wait) before loop(4)'s transfer is
// acquired (its pre-loop set) in program order. No loop-carried-dependency analysis
// is needed: the two sibling transfers reuse one common flag.

module {
  func.func @flag_reuse_sibling_unserialized(%arg0: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %c1 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c8 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 8 : i32
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32

    scf.for %i = %c0 to %c8 step %c1 : i32 {
      // ---- inner loop (3): self-contained CUBE -> VECTOR, no carry ----
      scf.for %j = %c0 to %c8 step %c1 : i32 {
        %e3 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
        %f3 = linalg.fill {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%e3 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %m3 = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%f3, %f3 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%f3 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %v3 = math.exp %m3 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
      } {ssbuffer.core_type = "VECTOR"}
      // ---- inner loop (4): self-contained CUBE -> VECTOR, no carry ----
      scf.for %k = %c0 to %c8 step %c1 : i32 {
        %e4 = tensor.empty() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
        %f4 = linalg.fill {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%e4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %m4 = linalg.matmul {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} ins(%f4, %f4 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%f4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %v4 = math.exp %m4 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
      } {ssbuffer.core_type = "VECTOR"}
    } {ssbuffer.core_type = "VECTOR"}
    return
  }
}

// CHECK-LABEL: func.func @flag_reuse_sibling_unserialized
// Both sibling transfers collapse onto one common flag (loop iterations are serial).
// CHECK: {{flag = }}[[REUSED_FLAG:[0-9]+]]{{$}}
// CHECK-COUNT-11: {{flag = }}[[REUSED_FLAG]]{{$}}
// CHECK: return
