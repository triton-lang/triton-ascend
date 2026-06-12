// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // 嵌套 then 拆分: 外层 if then 内有 block_id={18, 9} + 内层 if
  // 内层 if then 内有 block_id={7, 8}
  // 跨组依赖: block 18→9 (外层), block 7→8 (内层)
  // 期望: 外层先拆 (深度0), 内层后拆 (深度1), 原始 if 均移除
  // ============================================================================

  // CHECK-LABEL: func.func @test_nested_then_split
  // 外层: block 18 产出 %v1, 被 block 9 和内层 block 7 消费 → yield slot
  // CHECK: [[R0:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK-NEXT: arith.addi {{.*}} {ssbuffer.block_id = 18 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // 外层: block 9 void if, 包裹 block 9 op + 内层 split ifs
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK-NEXT: arith.muli [[R0]], {{.*}} {ssbuffer.block_id = 9 : i32}
  // 内层: block 7 产出, block 8 消费
  // CHECK: [[R1:%.*]] = scf.if
  // CHECK-SAME: -> (index)
  // CHECK-NEXT: arith.addi [[R0]], {{.*}} {ssbuffer.block_id = 7 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // 内层: block 8 void if
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK-NEXT: arith.muli [[R1]], {{.*}} {ssbuffer.block_id = 8 : i32}
  // 不应该再有原始嵌套 if 结构 (原 outer/inner if 均被移除)
  // CHECK-NOT: scf.if {

  func.func @test_nested_then_split(%cond1: i1, %cond2: i1) {
    %c0 = arith.constant {ssbuffer.block_id = 13 : i32} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 13 : i32} 1 : index
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
    return
  }

  // ============================================================================
  // 嵌套 then 拆分 (无跨组依赖): 外层 block_id={18, 9}, 内层 block_id={7, 8}
  // 各组独立无数据依赖 → 全 void split if
  // ============================================================================

  // CHECK-LABEL: func.func @test_nested_then_no_cross_dep
  // 外层: block 18 void if
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 18 : i32}
  // 外层: block 9 void if, 包含内层 split ifs
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 9 : i32}
  // 内层: block 7 void if
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 7 : i32}
  // 内层: block 8 void if (最后一个 group, 无结果)
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 8 : i32}
  // 无嵌套 scf.if 剩余
  // CHECK-NOT: scf.if {

  func.func @test_nested_then_no_cross_dep(%cond1: i1, %cond2: i1) {
    %c0 = arith.constant {ssbuffer.block_id = 13 : i32} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 13 : i32} 1 : index
    scf.if %cond1 {
      %v1 = arith.addi %c0, %c1 {ssbuffer.block_id = 18 : i32} : index
      %v2 = arith.muli %c0, %c1 {ssbuffer.block_id = 9 : i32} : index
      scf.if %cond2 {
        %v3 = arith.addi %c0, %c1 {ssbuffer.block_id = 7 : i32} : index
        %v4 = arith.muli %c0, %c1 {ssbuffer.block_id = 8 : i32} : index
      }
    }
    return
  }
}
