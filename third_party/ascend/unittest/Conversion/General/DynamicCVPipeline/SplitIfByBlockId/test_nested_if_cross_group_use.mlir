// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // Case A: outer if has no yield, then region has block_id={40, 26}
  // block 40's first op is a nested scf.if (no prior op to set currentId),
  // its result is consumed by block 26
  // Key: (1) groupOpsInBlock must use nested if's own block_id to create group
  // when currentId==-1; (2) planYieldCaseA must scan nestedIfs' results
  // Expected: split into 2 chained ifs, group 40 passes nested result via augmented yield slot
  // ============================================================================

  // CHECK-LABEL: func.func @test_nested_if_result_cross_group_use
  // for inside main_loop:
  // CHECK: scf.for
  // first split if: group 40, -> (index), yields nested if's result
  // CHECK: [[FIRST:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK: [[NESTED:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 18 : i32}
  // CHECK: scf.yield {{.*}} : index
  // CHECK: else
  // CHECK: scf.yield {{.*}} : index
  // CHECK: scf.yield [[NESTED]] : index
  // CHECK: else
  // CHECK: scf.yield
  // second split if: group 26, void, consumes first if's result
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: arith.muli [[FIRST]], {{.*}} {ssbuffer.block_id = 26 : i32}

  func.func @test_nested_if_result_cross_group_use(%cond1: i1, %cond2: i1) {
    %c0 = arith.constant {ssbuffer.block_id = 13 : i32} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 13 : i32} 1 : index
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %cond1 {
        // block 40: nested if as first op in thenBlock (no prior op to set currentId)
        %nested_res = scf.if %cond2 -> (index) {
          %v = arith.addi %c0, %c1 {ssbuffer.block_id = 18 : i32} : index
          scf.yield %v : index
        } else {
          scf.yield %c0 : index
        } {ssbuffer.block_id = 40 : i32}
        // block 26: consumes nested_res (cross-group)
        %v2 = arith.muli %nested_res, %c1 {ssbuffer.block_id = 26 : i32} : index
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }

  // ============================================================================
  // Case B: outer if has 1 original yield slot (N=1), then region has block_id={40, 26}
  // block 40 contains nested if + op producing original yield slot, nested if result consumed by block 26
  // Key: planYieldCaseB must scan nestedIfs' results
  // Expected: K=2 (N=1 original + M=1 augmented), nested result in augmented slot
  // ============================================================================

  // CHECK-LABEL: func.func @test_nested_if_result_cross_group_case_b
  // for inside main_loop:
  // CHECK: scf.for
  // first split if: K=2, group 40, slot 0 original + slot 1 augmented (nested result)
  // CHECK: [[FIRST_B:%.*]]:2 = scf.if
  // CHECK-SAME: -> (index, index)
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 40 : i32}
  // CHECK: [[NESTED_B:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 18 : i32}
  // CHECK: scf.yield {{.*}} : index
  // CHECK: else
  // CHECK: scf.yield {{.*}} : index
  // CHECK: scf.yield {{.*}}, [[NESTED_B]] : index, index
  // CHECK: else
  // CHECK: scf.yield
  // second split if: K=2, group 26, consumes augmented slot [[FIRST_B]]#1
  // CHECK: [[SECOND_B:%.*]]:2 = scf.if
  // CHECK-SAME: -> (index, index)
  // CHECK: arith.muli [[FIRST_B]]#1, {{.*}} {ssbuffer.block_id = 26 : i32}
  // for yields last split if's slot 0, slot 1 is passthrough initial value:
  // CHECK: scf.yield [[SECOND_B]]#0,
  // CHECK: return

  func.func @test_nested_if_result_cross_group_case_b(%cond1: i1, %cond2: i1) -> index {
    %c0 = arith.constant {ssbuffer.block_id = 13 : i32} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 13 : i32} 1 : index
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    %for:2 = scf.for %iv = %lb to %ub step %step iter_args(%a0 = %c0, %a1 = %c0) -> (index, index) {
      %result = scf.if %cond1 -> (index) {
        // block 40: produces original yield slot + nested if result
        %orig_yield = arith.addi %c0, %c1 {ssbuffer.block_id = 40 : i32} : index
        %nested_res = scf.if %cond2 -> (index) {
          %v = arith.addi %c0, %c1 {ssbuffer.block_id = 18 : i32} : index
          scf.yield %v : index
        } else {
          scf.yield %c0 : index
        } {ssbuffer.block_id = 40 : i32}
        // block 26: consumes nested_res (cross-group)
        %v2 = arith.muli %nested_res, %c1 {ssbuffer.block_id = 26 : i32} : index
        scf.yield %orig_yield : index
      } else {
        scf.yield %c0 : index
      } {ssbuffer.block_id = 16 : i32}
      scf.yield %result, %c0 : index, index
    } {ssbuffer.main_loop = 0 : i64}
    return %for#0 : index
  }
}
