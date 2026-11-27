// RUN: triton-opt --ub-usage-opt %s | FileCheck %s

// A CUBE dependency without incoming edges must still prevent pulling the
// reduction and its initialization into VECTOR block 7. Moving the CUBE
// constant with them would mix core types in one block and affect C/V pipelining.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @avoid_cv_pipeline
  func.func @avoid_cv_pipeline(%input: tensor<64x64xf32>, %init: tensor<64xf32>, %lb: index, %ub: index, %step: index) -> tensor<64xf32> {
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> tensor<64xf32> {
      // CHECK: math.exp {{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"}
      %large = math.exp %input {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      // CHECK: arith.constant {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"}
      %zero = arith.constant {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f32
      // CHECK: linalg.fill {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"}
      %seed = linalg.fill {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} ins(%zero : f32) outs(%acc : tensor<64xf32>) -> tensor<64xf32>
      // CHECK: linalg.reduce
      // CHECK-SAME: ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"
      %small = linalg.reduce ins(%large : tensor<64x64xf32>) outs(%seed : tensor<64xf32>) dimensions = [1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"}
        (%in: f32, %out: f32) {
          %sum = arith.addf %in, %out : f32
          linalg.yield %sum : f32
        }
      scf.yield %small : tensor<64xf32>
    } {ssbuffer.core_type = "VECTOR"}
    return %result : tensor<64xf32>
  }
}
