// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // Scene 3 Case B: then has block_id={10, 11}, else has block_id={5} actual ops
  // then splits into 2 split-ifs, last if's else absorbs block_id=5 ops
  // ============================================================================

  // CHECK-LABEL: func.func @test_scene3_case_b
  // for inside main_loop:
  // CHECK: scf.for
  // Placeholder:
  // CHECK: tensor.empty() {ssbuffer.block_id = -1 : i32}
  // first split if (block_id=10, group A):
  // CHECK: [[SF0:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // second split if (block_id=11, group B): then normal, else absorbs C
  // CHECK: [[SF1:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.mulf [[SF0]]#0, {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: scf.yield [[SF0]]#0,
  // CHECK: else
  // C's ops absorbed into last if's else:
  // CHECK-NEXT: arith.subf {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // for yields the last split if's result:
  // CHECK: scf.yield [[SF1]]#0, [[SF1]]#1
  // CHECK: return

  func.func @test_scene3_case_b(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
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
      scf.yield %res#0, %res#1 : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.main_loop = 0 : i64}

    return %for#0, %for#1 : tensor<16xf32>, tensor<16xf32>
  }
}
