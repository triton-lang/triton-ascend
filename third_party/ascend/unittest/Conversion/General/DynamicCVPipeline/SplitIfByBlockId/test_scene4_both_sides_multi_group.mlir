// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // Scene 4 Case B: then has block_id={10, 11}, else has block_id={5, 3}
  // Both sides have >=2 groups, needs two iterations:
  //   Round 1: split then side, last split if's else absorbs all else-side ops
  //   Round 2: after absorption, else has 2 groups, triggers else-side split
  // ============================================================================

  // CHECK-LABEL: func.func @test_scene4_case_b

  func.func @test_scene4_case_b(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.0 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %1 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %cst2 = arith.constant {ssbuffer.block_id = 13 : i32} 1.0 : f32
    %2 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<16xf32>
    %f0 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst2 : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>

    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    %for:2 = scf.for %iv = %lb to %ub step %step iter_args(%a0 = %1, %a1 = %1) -> (tensor<16xf32>, tensor<16xf32>) {
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
      scf.yield %res#0, %res#1 : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.main_loop = 0 : i64}

    return %for#0, %for#1 : tensor<16xf32>, tensor<16xf32>
  }

  // Round 1 + Round 2 expected structure:
  //
  // for inside main_loop:
  // CHECK: scf.for
  //
  // Round 1 — split then side (block_id=10, 11):
  //
  // CHECK: tensor.empty() {ssbuffer.block_id = -1 : i32}
  //
  // Round 1 first split if: block_id=10 (then)
  // CHECK: [[R1:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  //
  // Round 2 — split else side (block_id=5, 3), negated condition:
  //
  // CHECK: arith.constant true
  // CHECK: [[NEG:%.*]] = arith.xori
  // CHECK: tensor.empty() {ssbuffer.block_id = -1 : i32}
  //
  // Round 2 first split if: block_id=5 (then, negated condition)
  // CHECK: [[R2:%.*]]:2 = scf.if [[NEG]]
  // CHECK-NEXT: arith.subf {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  //
  // Round 2 second split if: block_id=3 (then) + block_id=11 absorbed into else
  // CHECK: [[R3:%.*]]:2 = scf.if [[NEG]]
  // CHECK-NEXT: arith.addf [[R2]]#0, {{.*}} {ssbuffer.block_id = 3 : i32}
  // CHECK: scf.yield [[R2]]#0,
  // CHECK: else
  // original then-side block_id=11 ops absorbed here as else:
  // CHECK-NEXT: arith.mulf [[R1]]#0, {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: scf.yield [[R1]]#0,
  //
  // for yields the last split if's result:
  // CHECK: scf.yield [[R3]]#0, [[R3]]#1
  // final return:
  // CHECK: return
}
