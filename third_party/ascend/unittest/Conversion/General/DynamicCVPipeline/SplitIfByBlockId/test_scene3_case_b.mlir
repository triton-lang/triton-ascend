// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // 场景3 Case B: then 有 block_id={10, 11}, else 有 block_id={5} actual ops
  // then 拆分为 2 个 split if, 最后 if 的 else 吸收 block_id=5 的 ops
  // ============================================================================

  // CHECK-LABEL: func.func @test_scene3_case_b
  // Placeholder:
  // CHECK: tensor.empty() {ssbuffer.block_id = -1 : i32}
  // 第一个 split if (block_id=10, A group):
  // CHECK: [[SF0:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // 第二个 split if (block_id=11, B group): then normal, else 吸收 C
  // CHECK: [[SF1:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.mulf [[SF0]]#0, {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: scf.yield [[SF0]]#0,
  // CHECK: else
  // C's ops absorbed into last if's else:
  // CHECK-NEXT: arith.subf {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // Final result from last split if:
  // CHECK: return [[SF1]]#0, [[SF1]]#1

  func.func @test_scene3_case_b(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.0 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %1 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %cst2 = arith.constant {ssbuffer.block_id = 13 : i32} 1.0 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %f0 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst2 : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>

    %res:2 = scf.if %cond -> (tensor<16xf32>, tensor<16xf32>) {
      // block_id=10: then ops group A
      %a = arith.addf %1, %f0 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
      // block_id=11: then ops group B (uses %a -> cross-group dep)
      %b = arith.mulf %a, %f0 {ssbuffer.block_id = 11 : i32} : tensor<16xf32>
      scf.yield %a, %b : tensor<16xf32>, tensor<16xf32>
    } else {
      // block_id=5: else ops (group C)
      %c = arith.subf %f0, %f0 {ssbuffer.block_id = 5 : i32} : tensor<16xf32>
      scf.yield %c, %c : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.block_id = 12 : i32}

    return %res#0, %res#1 : tensor<16xf32>, tensor<16xf32>
  }
}
