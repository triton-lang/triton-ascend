// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // Nested then split: outer if then region has block_id={18, 9} + inner if
  // Inner if then region has block_id={7, 8}
  // Cross-group deps: block 18→9 (outer), block 7→8 (inner)
  // Iteration 1: outer split (bid=18 result if + bid=9 wraps inner if),
  //              inner split (bid=7 result if + bid=8 void if)
  // Iteration 2: bid=9 region gets nested split-ifs (bid=23,24) → split again
  // Result: each outer split-if contains only one block_id
  // ============================================================================

  // CHECK-LABEL: func.func @test_nested_then_split
  // CHECK: scf.for
  // outer split-if 1: bid=18, result if, 1 result
  // CHECK: [[R0:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 18 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // outer split-if 2: bid=9, result if, 1 result, consumes R0
  // CHECK: [[R1:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK: arith.muli [[R0]], {{.*}} {ssbuffer.block_id = 9 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // outer split-if 3: wraps inner result if (bid=7), 1 result, else passthrough R1
  // CHECK: [[R2:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // inner result if (bid=7)
  // CHECK: [[R3:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK: arith.addi [[R0]], {{.*}} {ssbuffer.block_id = 7 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // CHECK: scf.yield [[R3]]
  // CHECK: else
  // CHECK: scf.yield [[R1]]
  // outer split-if 4: void if wraps inner void if (bid=8), consumes R2
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: scf.if
  // CHECK: arith.muli [[R2]], {{.*}} {ssbuffer.block_id = 8 : i32}

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

  // ============================================================================
  // Nested then split (no cross-group deps): outer block_id={18, 9}, inner block_id={7, 8}
  // Each group is independent (no data deps) → after iteration 2, inner ifs are wrapped by outer void ifs
  // ============================================================================

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
}
