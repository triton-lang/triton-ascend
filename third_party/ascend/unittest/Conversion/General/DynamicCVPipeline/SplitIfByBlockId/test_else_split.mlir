// RUN: triton-opt --split-if-by-block-id %s | FileCheck %s

module {
  // ============================================================================
  // Else 拆分 (Case A): then 为空, else 内有 block_id={5, 3}
  // 期望: 条件取反, 原外层空 if 移除, 用取反条件包裹 else ops 并拆分
  // ============================================================================

  // CHECK-LABEL: func.func @test_else_split_basic
  // 取反条件
  // CHECK: arith.xori
  // 第一个 split if (block_id=5): 新增 yield slot
  // CHECK: [[R0:%.*]] = scf.if
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // 第二个 split if (block_id=3): 纯副作用, 消费 [[R0]]
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK-NEXT: arith.index_cast [[R0]] {ssbuffer.block_id = 3 : i32}
  // CHECK: memref.dealloc {{.*}} {ssbuffer.block_id = 3 : i32}
  // 注意: 原来的空 then if 已消失, 外层不应该再有嵌套 scf.if 包裹

  func.func @test_else_split_basic(%cond: i1) {
    %c64_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : i32
    %c64 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : index
    %c0 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : index
    %alloc = memref.alloc() {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    scf.if %cond {
    } else {
      %v1 = arith.maxsi %c64_i32, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
      %v2 = arith.index_cast %v1 {ssbuffer.block_id = 3 : i32} : i32 to index
      %v3 = arith.muli %v2, %c64 {ssbuffer.block_id = 3 : i32} : index
      memref.dealloc %alloc {ssbuffer.block_id = 3 : i32} : memref<64x64xf16>
    } {ssbuffer.block_id = 17 : i32}
    memref.dealloc %alloc {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    return
  }

  // ============================================================================
  // Else 拆分 (Case B): 原始 if 有 yield (N=2), then 的 yield 传原始空 then 值
  // else 内 block_id={10, 11}, 跨组依赖: block 10 → block 11 (在 slot 0 中)
  // ============================================================================

  // CHECK-LABEL: func.func @test_else_split_with_yield
  // 条件取反
  // CHECK: arith.xori
  // 第一个 split if (block_id=10): 替代原 else→then
  // CHECK: [[R0:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.addf {{.*}} {ssbuffer.block_id = 10 : i32}
  // CHECK: scf.yield
  // 第二个 split if (block_id=11): 消费 [[R0]]#0
  // CHECK: [[R1:%.*]]:2 = scf.if
  // CHECK-NEXT: arith.mulf [[R0]]#0, {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK: scf.yield [[R0]]#0,
  // 原始 uses 替换为最后一个 if 的 result
  // CHECK: return [[R1]]#0, [[R1]]#1

  func.func @test_else_split_with_yield(%cond: i1) -> (tensor<16xf32>, tensor<16xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 14 : i32} 0.0 : f32
    %t0 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<16xf32>
    %f0 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst : f32) outs(%t0 : tensor<16xf32>) -> tensor<16xf32>
    %cst2 = arith.constant {ssbuffer.block_id = 14 : i32} 1.0 : f32
    %t1 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<16xf32>
    %f1 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst2 : f32) outs(%t1 : tensor<16xf32>) -> tensor<16xf32>
    %4:2 = scf.if %cond -> (tensor<16xf32>, tensor<16xf32>) {
      scf.yield %f0, %f0 : tensor<16xf32>, tensor<16xf32>
    } else {
      %v1 = arith.addf %f1, %f0 {ssbuffer.block_id = 10 : i32} : tensor<16xf32>
      %v2 = arith.mulf %v1, %f1 {ssbuffer.block_id = 11 : i32} : tensor<16xf32>
      scf.yield %v1, %v2 : tensor<16xf32>, tensor<16xf32>
    } {ssbuffer.block_id = 17 : i32}
    return %4#0, %4#1 : tensor<16xf32>, tensor<16xf32>
  }

  // ============================================================================
  // 嵌套 else 拆分: 外层 if else 内有嵌套 if, 嵌套 if else 内有不同 block_id ops
  // 外层 if 保留 (then 空, else 有嵌套 if 的 block_id 结果)
  // 内层 scf.if 的 else→then 取反拆分
  // ============================================================================

  // CHECK-LABEL: func.func @test_nested_else_split
  // 外层 if 保留
  // CHECK: scf.if
  // CHECK: } else {
  // 内层取反条件
  // CHECK: arith.xori
  // block_id=5 的 split if
  // CHECK: [[R0:%.*]] = scf.if
  // CHECK-SAME: -> (i32)
  // CHECK-NEXT: arith.maxsi {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: scf.yield
  // block_id=3 的 split if (纯副作用)
  // CHECK: scf.if
  // CHECK-NOT: ->
  // CHECK-NEXT: arith.index_cast [[R0]] {ssbuffer.block_id = 3 : i32}

  func.func @test_nested_else_split(%outer: i1, %inner: i1) {
    %c64_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : i32
    %c64 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : index
    %c0 = arith.constant {ssbuffer.block_id = 14 : i32} 0 : index
    %alloc = memref.alloc() {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
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
    memref.dealloc %alloc {ssbuffer.block_id = 14 : i32} : memref<64x64xf16>
    return
  }
}
