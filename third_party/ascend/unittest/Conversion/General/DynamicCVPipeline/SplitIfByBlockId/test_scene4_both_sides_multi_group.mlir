// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // 场景4 Case B: then 有 block_id={10, 11}, else 有 block_id={5, 3}
  // 两边都有 >=2 组, 需要两轮迭代:
  //   Round 1: 拆分 then 侧, 最后一个 split if 的 else 吸收 else 侧所有 ops
  //   Round 2: 吸收后 else 内有 2 组, 触发 else 侧拆分
  // ============================================================================

  // CHECK-LABEL: func.func @test_scene4_case_b

  func.func @test_scene4_case_b(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.0 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %1 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %cst2 = arith.constant {ssbuffer.block_id = 13 : i32} 1.0 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %f0 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst2 : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>

    %res:2 = scf.if %cond -> (tensor<16xf32>, tensor<16xf32>) {
      // block_id=10: then group A
      %a = arith.addf %1, %f0 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
      // block_id=11: then group B (uses %a -> cross-group dep)
      %b = arith.mulf %a, %f0 {ssbuffer.block_id = 11 : i32} : tensor<16xf32>
      scf.yield %a, %b : tensor<16xf32>, tensor<16xf32>
    } else {
      // block_id=5: else group C
      %c = arith.subf %f0, %f0 {ssbuffer.block_id = 5 : i32} : tensor<16xf32>
      // block_id=3: else group D (uses %c -> cross-group dep)
      %d = arith.addf %c, %1 {ssbuffer.block_id = 3 : i32} : tensor<16xf32>
      scf.yield %c, %d : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.block_id = 12 : i32}

    return %res#0, %res#1 : tensor<16xf32>, tensor<16xf32>
  }

  // Round 1 + Round 2 期望结构:
  //
  // Round 1 — 拆分 then 侧 (block_id=10, 11):
  //
  // CHECK: tensor.empty() {ssbuffer.block_id = -1 : i32}
  //
  // Round 1 第一个 split if: block_id=10 (then)
  // CHECK: [[R1:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  //
  // Round 2 — 拆分 else 侧 (block_id=5, 3), 条件取反:
  //
  // CHECK: arith.constant true
  // CHECK: [[NEG:%.*]] = arith.xori
  // CHECK: tensor.empty() {ssbuffer.block_id = -1 : i32}
  //
  // Round 2 第一个 split if: block_id=5 (then, 取反条件)
  // CHECK: [[R2:%.*]]:2 = scf.if [[NEG]]
  // CHECK-NEXT: arith.subf {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  //
  // Round 2 第二个 split if: block_id=3 (then) + block_id=11 吸收到 else
  // CHECK: [[R3:%.*]]:2 = scf.if [[NEG]]
  // CHECK-NEXT: arith.addf [[R2]]#0, {{.*}} {ssbuffer.block_id = 3 : i32}
  // CHECK: scf.yield [[R2]]#0,
  // CHECK: else
  // 原来 then 侧 block_id=11 的 ops 作为 else 被吸收到这里:
  // CHECK-NEXT: arith.mulf [[R1]]#0, {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: scf.yield [[R1]]#0,
  //
  // 最终 return 使用最后一个 split if 的 result:
  // CHECK: return [[R3]]#0, [[R3]]#1
}
