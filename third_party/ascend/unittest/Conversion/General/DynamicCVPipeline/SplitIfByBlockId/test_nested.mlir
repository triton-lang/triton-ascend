// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ==========================================================================
  // Nested if — scf.if inside scf.if, with multi-group ops in inner regions
  //
  // The pass processes nested candidates outermost-first (sorted by ifDepth),
  // locks each inner candidate's ops under its parent's group, and iterates
  // until no more candidates remain.
  // ==========================================================================

  // --------------------------------------------------------------------------
  // Nested then split: outer if then region has block_id={18, 9} + inner if
  // Inner if then region has block_id={7, 8}
  // Cross-group deps: block 18→9 (outer), block 7→8 (inner)
  // Iteration 1: outer split (bid=18 result if + bid=9 wraps inner if),
  //              inner split (bid=7 result if + bid=8 void if)
  // Iteration 2: bid=9 region gets nested split-ifs (bid=23,24) → split again
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_nested_then_split
  // CHECK: scf.for
  // outer split-if 1: bid=18, result if, 1 output (cross-group %v1)
  // CHECK: [[R0:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 18 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // outer split-if 2: bid=9, void if (no cross-group output, consumes R0)
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: arith.muli [[R0]], {{.*}} {ssbuffer.block_id = 9 : i32}
  // outer split-if 3: wraps inner result if (bid=7), 1 result
  // CHECK: [[R1:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // inner result if (bid=7)
  // CHECK: [[INNER_R:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK: arith.addi [[R0]], {{.*}} {ssbuffer.block_id = 7 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // CHECK: scf.yield [[INNER_R]]
  // CHECK: else
  // CHECK: scf.yield
  // outer split-if 4: void if wraps inner void if (bid=8), consumes R1
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: scf.if
  // CHECK: arith.muli [[R1]], {{.*}} {ssbuffer.block_id = 8 : i32}

  func.func @test_nested_then_split(%cond1: i1, %cond2: i1) {
    %c0 = arith.constant {ssbuffer.block_id = 13 : i32} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 13 : i32} 1 : index
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %cond1 {
        // block 18: produce %v1
        %v1 = arith.addi %c0, %c1 {ssbuffer.block_id = 18 : i32} : index
        // block 9: consume %v1 (cross-group)
        %v2 = arith.muli %v1, %c1 {ssbuffer.block_id = 9 : i32} : index
        // nested if
        scf.if %cond2 {
          // block 7: produce %v3, also uses %v1 from outer scope
          %v3 = arith.addi %v1, %c1 {ssbuffer.block_id = 7 : i32} : index
          // block 8: consume %v3 (cross-group)
          %v4 = arith.muli %v3, %c1 {ssbuffer.block_id = 8 : i32} : index
        }
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }

  // --------------------------------------------------------------------------
  // Nested then split (no cross-group deps): outer block_id={18, 9},
  // inner block_id={7, 8}. Each group independent → after iteration 2,
  // inner ifs are wrapped by outer void ifs.
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_nested_then_no_cross_dep
  // for inside main_loop:
  // CHECK: scf.for
  // outer: block 18 void if
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 18 : i32}
  // outer: block 9 void if (inner already separated)
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 9 : i32}
  // outer void if wrapping inner block 7 void if
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: scf.if
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 7 : i32}
  // outer void if wrapping inner block 8 void if
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: scf.if
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 8 : i32}

  func.func @test_nested_then_no_cross_dep(%cond1: i1, %cond2: i1) {
    %c0 = arith.constant {ssbuffer.block_id = 13 : i32} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 13 : i32} 1 : index
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %cond1 {
        %v1 = arith.addi %c0, %c1 {ssbuffer.block_id = 18 : i32} : index
        %v2 = arith.muli %c0, %c1 {ssbuffer.block_id = 9 : i32} : index
        scf.if %cond2 {
          %v3 = arith.addi %c0, %c1 {ssbuffer.block_id = 7 : i32} : index
          %v4 = arith.muli %c0, %c1 {ssbuffer.block_id = 8 : i32} : index
        }
      }
    } {ssbuffer.main_loop = 0 : i64}
    return
  }

  // --------------------------------------------------------------------------
  // Case A: nested if's result is used across groups in outer scope
  // Outer if is void (Case A), then region has block_id={40, 26}
  // block 40's first op is a nested scf.if (no prior op to set currentId),
  // its result is consumed by block 26
  // Key: groupOpsInBlock must use nested if's own block_id to create group
  // when currentId==-1; planYieldCaseA must scan nestedIfs' results
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_nested_cross_group_use_case_a
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

  func.func @test_nested_cross_group_use_case_a(%cond1: i1, %cond2: i1) {
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

  // --------------------------------------------------------------------------
  // Case B: nested if's result used across groups in outer scope
  // Outer if has 1 original yield slot (N=1), then region has block_id={40, 26}
  // block 40 contains nested if + op producing original yield slot,
  // nested if result consumed by block 26 (cross-group)
  // Result: K=2 (N=1 original + M=1 augmented for nested result)
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_nested_cross_group_use_case_b
  // for inside main_loop:
  // CHECK: scf.for
  // first split if: group 40, 2 outputs (augmented cross-group + original slot 0)
  // CHECK: [[FIRST_B:%.*]]:2 = scf.if
  // CHECK-SAME: -> (index, index)
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 40 : i32}
  // CHECK: [[NESTED_B:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 18 : i32}
  // CHECK: scf.yield {{.*}} : index
  // CHECK: else
  // CHECK: scf.yield {{.*}} : index
  // CHECK: scf.yield [[NESTED_B]], {{.*}} : index, index
  // CHECK: else
  // CHECK: scf.yield
  // second split if: group 26, last if carries original result type (N=1)
  // CHECK: [[SECOND_B:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK: arith.muli [[FIRST_B]]#0, {{.*}} {ssbuffer.block_id = 26 : i32}
  // for yields last split if's slot 0:
  // CHECK: scf.yield [[SECOND_B]],
  // CHECK: return

  func.func @test_nested_cross_group_use_case_b(%cond1: i1, %cond2: i1) -> index {
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

  // --------------------------------------------------------------------------
  // Nested else split: outer if's else region contains another scf.if,
  // inner if's else has block_id={5, 3}
  // Iteration 1: inner if split into bid=5 and bid=3 split-ifs
  // Iteration 2: outer if's else block now has multiple block_id nested ifs
  //              → outer also splits
  // --------------------------------------------------------------------------

  // CHECK-LABEL: func.func @test_nested_else_split
  // for inside main_loop:
  // CHECK: scf.for
  // outer negated condition
  // CHECK: arith.xori
  // outer result if: wraps block_id=5 split-if + inner condition + inner result
  // CHECK: [[R0:%.*]]:2 = scf.if
  // CHECK-SAME: -> (i1, i32)
  // inner negated condition
  // CHECK: [[INNER_COND:%.*]] = arith.xori
  // inner result if (block_id=5)
  // CHECK: [[INNER_R:%.*]] = scf.if [[INNER_COND]] -> (i32)
  // CHECK: arith.maxsi {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // outer then yield: inner condition + inner result
  // CHECK: scf.yield [[INNER_COND]], [[INNER_R]]
  // CHECK: else
  // CHECK: scf.yield
  // outer void if wrapping block_id=3 void if: consumes [[R0]]#0 + [[R0]]#1
  // CHECK: scf.if
  // CHECK-NOT: ->
  // inner void if (block_id=3)
  // CHECK: scf.if [[R0]]#0
  // CHECK: arith.index_cast [[R0]]#1 {ssbuffer.block_id = 3 : i32}

  func.func @test_nested_else_split(%outer: i1, %inner: i1) {
    %c64_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : i32
    %c64 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : index
    %c0 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : index
    %alloc = memref.alloc() {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      scf.if %outer {
      } else {
        scf.if %inner {
        } else {
          %v1 = arith.maxsi %c64_i32, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
          %v2 = arith.index_cast %v1 {ssbuffer.block_id = 3 : i32} : i32 to index
          %v3 = arith.muli %v2, %c64 {ssbuffer.block_id = 3 : i32} : index
          memref.dealloc %alloc {ssbuffer.block_id = 3 : i32} : memref<64x64xf16>
        } {ssbuffer.block_id = 16 : i32}
      } {ssbuffer.block_id = 17 : i32}
    } {ssbuffer.main_loop = 0 : i64}
    memref.dealloc %alloc {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    return
  }
}
