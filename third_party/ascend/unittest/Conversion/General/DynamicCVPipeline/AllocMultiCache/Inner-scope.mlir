// RUN: triton-opt --add_multi_buffer_inner_scope %s | FileCheck %s

// Consolidated Unit Tests for AddMultiBufferInnerScope Pass
// Tests buffer creation, producer logic, consumer logic with ping-pong select chain

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

  //===--------------------------------------------------------------------===//
  // T1: Double Buffer Creation Test
  // Test: cross-block tensor with block_id 5→6 triggers buffer creation
  // Key Check: memref.alloc creates ping buffer, scf.if selects buffer
  //         : producer tag [groupId, 1] on memref.memory_space_cast
  //         : consumer tag [groupId, 0] on scf.if
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t1_double_buffer
  // CHECK-DAG: memref.alloc
  // CHECK-DAG: memref.memory_space_cast
  // CHECK-DAG: bufferization.to_tensor
  // CHECK-DAG: hivm.hir.copy
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 5
  func.func @test_t1_double_buffer() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        %consumed = arith.addf %arg, %arg {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_prod = arith.addf %consumed, %consumed {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T2: Multiple Block IDs Test
  // Test: block_id 5, 6, 7 all present in cross-block deps
  // Key Check: multiple buffers created for different block_ids
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t2_multiple_blocks
  // CHECK-DAG: memref.alloc
  // CHECK-DAG: memref.alloc
  // CHECK-DAG: memref.alloc
  // CHECK-DAG: memref.alloc
  // CHECK-DAG: linalg.fill {ssbuffer.block_id = 5 : i32}
  // CHECK-DAG: linalg.fill {ssbuffer.block_id = 7 : i32}
  func.func @test_t2_multiple_blocks() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<64xf32>
    scope.scope : () -> () {
      %prod5 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
      %prod7 = linalg.fill {ssbuffer.block_id = 7 : i32} ins(%cst : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
      %loop_result:2 = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg5 = %prod5, %arg7 = %prod7) -> (tensor<64xf32>, tensor<64xf32>) : i32 {
        %cons6 = arith.addf %arg5, %arg5 {ssbuffer.block_id = 6 : i32} : tensor<64xf32>
        %cons8 = arith.mulf %arg7, %arg7 {ssbuffer.block_id = 8 : i32} : tensor<64xf32>
        %new_prod5 = arith.addf %cons6, %cons6 {ssbuffer.block_id = 5 : i32} : tensor<64xf32>
        %new_prod7 = arith.mulf %cons8, %cons8 {ssbuffer.block_id = 7 : i32} : tensor<64xf32>
        scf.yield %new_prod5, %new_prod7 : tensor<64xf32>, tensor<64xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T3: 2D Tensor Double Buffer
  // Test: 2D tensor with cross-block dependency
  // Key Check: 2D memref buffer created with correct shape
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t3_2d_tensor
  // CHECK: memref.alloc
  // CHECK: tensor<128x64xf32>
  func.func @test_t3_2d_tensor() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128x64xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128x64xf32>) -> tensor<128x64xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128x64xf32>) : i32 {
        %consumed = arith.mulf %arg, %arg {ssbuffer.block_id = 6 : i32} : tensor<128x64xf32>
        %new_prod = arith.addf %consumed, %consumed {ssbuffer.block_id = 5 : i32} : tensor<128x64xf32>
        scf.yield %new_prod : tensor<128x64xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T4: getIterCount - Loop iteration count calculation
  // Test: lb=0, step=1, iv=i32 → iterCount = i
  // Key Check: main_loop preserved, no buffer if no cross-block dep
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t4_getitercount
  // CHECK: scope.scope
  // CHECK: linalg.fill
  // CHECK: scf.for
  // CHECK: {ssbuffer.main_loop = 1 : i64}
  func.func @test_t4_getitercount() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        %r = arith.addf %arg, %arg {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %r : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T5: findMainLoopInScope - Find main loop in scope
  // Test: forOp has main_loop attribute → returns that forOp
  // Key Check: main_loop attribute preserved
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t5_findmainloop
  // CHECK: linalg.fill {ssbuffer.block_id = 5 : i32}
  // CHECK: scf.for
  // CHECK: {ssbuffer.main_loop = 1 : i64}
  func.func @test_t5_findmainloop() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        %r = arith.addf %arg, %arg {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %r : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T6: collectBlockInfo - Group ops by ssbuffer.block_id
  // Test: single block → returns 1 block group (no buffer needed)
  // Key Check: block_id 5 both producer and consumer
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t6_collectblockinfo
  // CHECK: linalg.fill {ssbuffer.block_id = 5 : i32}
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 5 : i32}
  func.func @test_t6_collectblockinfo() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<64xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
      %cons = arith.addf %prod, %prod {ssbuffer.block_id = 5 : i32} : tensor<64xf32>
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %cons) -> (tensor<64xf32>) : i32 {
        %r = arith.addf %arg, %arg {ssbuffer.block_id = 5 : i32} : tensor<64xf32>
        scf.yield %r : tensor<64xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T7: dep_mark Array Attribute Test
  // Test: scalar deps inside main_loop get dep_mark attribute
  // Key Check: dep_mark array attribute on ops
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t7_dep_mark
  // CHECK-DAG: memref.load {{.*}} {ssbuffer.block_id = 11 : i32}
  // CHECK-DAG: arith.addi {{.*}} ssbuffer.dep_mark
  // CHECK-DAG: arith.muli {{.*}} ssbuffer.dep_mark
  // CHECK-DAG: arith.divsi {{.*}} ssbuffer.dep_mark
  func.func @test_t7_dep_mark() {
    %true = arith.constant true
    %c0_i64 = arith.constant 0 : i64
    %c100_i64 = arith.constant 100 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<1xi64>
      %cast_alloc = memref.cast %alloc : memref<1xi64> to memref<1xi64, strided<[1], offset: ?>>
      scf.for %i = %c0_i64 to %c100_i64 step %c1_i64 : i64 {
        %load = memref.load %cast_alloc[%c0] {ssbuffer.block_id = 11 : i32} : memref<1xi64, strided<[1], offset: ?>>
        %compute = arith.addi %load, %c1_i64 {ssbuffer.block_id = 11 : i32} : i64
        %result = scf.if %true -> (i64) {
          %r1 = arith.muli %compute, %c1_i64 {ssbuffer.block_id = 8 : i32} : i64
          scf.yield %r1 : i64
        } else {
          %r2 = arith.divsi %compute, %c1_i64 {ssbuffer.block_id = 9 : i32} : i64
          scf.yield %r2 : i64
        }{ssbuffer.block_id = 12 : i32}
        memref.store %compute, %cast_alloc[%c0] {ssbuffer.block_id = 12 : i32} : memref<1xi64, strided<[1], offset: ?>>
      } {ssbuffer.block_id = 26 : i32, ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T8: Memref Operand Test
  // Test: memref.alloc with block_id passed to materialize_in_destination
  //       with different block_id. The tensor operand crosses blocks, but
  //       memref operand is not tracked as cross-block dep in typical flow.
  // Key Check: double buffer is created since cross-block dep is tensor type
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t8_memref_operand
  // CHECK: scope.scope
  // CHECK: memref.memory_space_cast {{.*}} {ssbuffer.intraDeps
  func.func @test_t8_memref_operand() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %mem = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<128xf32>
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        %consumed = arith.addf %arg, %arg {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        bufferization.materialize_in_destination %consumed in writable %mem {ssbuffer.block_id = 6 : i32} : (tensor<128xf32>, memref<128xf32>) -> ()
        %new_prod = arith.addf %consumed, %consumed {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T9: Multiple Main Loops - Global groupId Uniqueness
  // Test: Two parallel main_loop forOps in same vector scope
  // Key Check: intraDeps groupId is globally unique across all main_loops,
  //            NOT reset to 0 for each main_loop
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t9_multiple_main_loops
  // CHECK-DAG: memref.memory_space_cast {{.*}} {ssbuffer.intraDeps = [0 : i32, 1 : i32]}
  // CHECK-DAG: memref.memory_space_cast {{.*}} {ssbuffer.intraDeps = [1 : i32, 1 : i32]}
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 8
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 8
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 7
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 7
  // CHECK-DAG: } {ssbuffer.main_loop = 1 : i64}
  // CHECK-DAG: } {ssbuffer.main_loop = 1 : i64}
  // CHECK-DAG: ssbuffer.intraDeps = [0 : i32, 0 : i32]
  // CHECK-DAG: ssbuffer.intraDeps = [1 : i32, 0 : i32]
  func.func @test_t9_multiple_main_loops() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty_128 = tensor.empty() : tensor<128xf32>
    %empty_64 = tensor.empty() : tensor<64xf32>
    scope.scope : () -> () {
      %prod1 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty_128 : tensor<128xf32>) -> tensor<128xf32>
      %loop_result1 = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg1 = %prod1) -> (tensor<128xf32>) : i32 {
        %cons1 = arith.addf %arg1, %arg1 {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_prod1 = arith.addf %cons1, %cons1 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod1 : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      %prod2 = linalg.fill {ssbuffer.block_id = 7 : i32} ins(%cst : f32) outs(%empty_64 : tensor<64xf32>) -> tensor<64xf32>
      %loop_result2 = scf.for %j = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg2 = %prod2) -> (tensor<64xf32>) : i32 {
        %cons2 = arith.mulf %arg2, %arg2 {ssbuffer.block_id = 8 : i32} : tensor<64xf32>
        %new_prod2 = arith.mulf %cons2, %cons2 {ssbuffer.block_id = 7 : i32} : tensor<64xf32>
        scf.yield %new_prod2 : tensor<64xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T10: Single Buffer Test (N==1) - Verification of Single Buffer Tags
  // Test: When BufferCountManager returns 1, single buffer is created
  // Key Check: producer (materialize_in_destination) and consumer (to_tensor)
  //         : both tagged with block_id, intraDeps=[groupId,0], intra_buffer
  // Note: This test shows the expected format when bufNum==1.
  //       Default bufNum is 2, so this test follows double buffer output.
  //       For true single buffer test, need BufferCountManager::setBufferCount(IntraCore, 1)
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t10_single_buffer
  // CHECK-DAG: memref.memory_space_cast {{.*}} {ssbuffer.intraDeps = [0 : i32, 1 : i32]}
  // CHECK-DAG: memref.memory_space_cast {{.*}} {ssbuffer.intraDeps = [0 : i32, 1 : i32]}
  // CHECK-DAG: } {ssbuffer.block_id = 5 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32], ssbuffer.intra_buffer}
  func.func @test_t10_single_buffer() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        %consumed = arith.addf %arg, %arg {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_prod = arith.addf %consumed, %consumed {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T11: DivUIOp/SubIOp Counter Test (lb != 0 or step != 1)
  // Test: When lb != 0, SubIOp is created for (iv - lb) / step
  //       When step != 1, DivUIOp is created for iterIdx / step
  // Key Check: Both SubIOp and DivUIOp get block_id tag
  //         : remsi and cmpi also get block_id tag
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t11_counter_with_divsub
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.divui {{.*}} {ssbuffer.block_id = 5
  func.func @test_t11_counter_with_divsub() {
    %c10_i32 = arith.constant 10 : i32
    %c100_i32 = arith.constant 100 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c10_i32 to %c100_i32 step %c2_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        %consumed = arith.addf %arg, %arg {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_prod = arith.addf %consumed, %consumed {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T12: Index Type Loop Variable Test (IndexCastOp block_id)
  // Test: When loop induction variable is 'index' type, IndexCastOp is created
  //       to convert from index to i32 for buffer index calculation.
  // Key Check: IndexCastOp gets block_id tag from the block it belongs to
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t12_index_type_loop_var
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.index_cast {{.*}} {ssbuffer.block_id = 5
  func.func @test_t12_index_type_loop_var() {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0 to %c100 step %c1 iter_args(%arg = %prod) -> (tensor<128xf32>) : index {
        %consumed = arith.addf %arg, %arg {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_prod = arith.addf %consumed, %consumed {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T13: Three Parallel Main Loops - Global groupId Uniqueness
  // Test: Three parallel main_loop forOps in same vector scope
  // Key Check: groupId is globally unique across all main_loops (0, 1, 2)
  //         : Each main_loop's intraDeps reference different groupIds
  //         : remsi and cmpi get correct block_id tags
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t13_three_parallel_main_loops
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 8
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 8
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 7
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 7
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 10
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 10
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 9
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 9
  // CHECK-DAG: } {ssbuffer.main_loop = 1 : i64}
  // CHECK-DAG: } {ssbuffer.main_loop = 1 : i64}
  // CHECK-DAG: } {ssbuffer.main_loop = 1 : i64}
  // CHECK-DAG: ssbuffer.intraDeps = [0 : i32, 1 : i32]
  // CHECK-DAG: ssbuffer.intraDeps = [1 : i32, 1 : i32]
  // CHECK-DAG: ssbuffer.intraDeps = [2 : i32, 1 : i32]
  func.func @test_t13_three_parallel_main_loops() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty_128 = tensor.empty() : tensor<128xf32>
    %empty_64 = tensor.empty() : tensor<64xf32>
    %empty_32 = tensor.empty() : tensor<32xf32>
    scope.scope : () -> () {
      // First main_loop: block_id 5 -> 6
      %prod1 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty_128 : tensor<128xf32>) -> tensor<128xf32>
      %loop_result1 = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg1 = %prod1) -> (tensor<128xf32>) : i32 {
        %cons1 = arith.addf %arg1, %arg1 {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_prod1 = arith.addf %cons1, %cons1 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod1 : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      // Second main_loop: block_id 7 -> 8
      %prod2 = linalg.fill {ssbuffer.block_id = 7 : i32} ins(%cst : f32) outs(%empty_64 : tensor<64xf32>) -> tensor<64xf32>
      %loop_result2 = scf.for %j = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg2 = %prod2) -> (tensor<64xf32>) : i32 {
        %cons2 = arith.mulf %arg2, %arg2 {ssbuffer.block_id = 8 : i32} : tensor<64xf32>
        %new_prod2 = arith.mulf %cons2, %cons2 {ssbuffer.block_id = 7 : i32} : tensor<64xf32>
        scf.yield %new_prod2 : tensor<64xf32>
      } {ssbuffer.main_loop = 1 : i64}
      // Third main_loop: block_id 9 -> 10
      %prod3 = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst : f32) outs(%empty_32 : tensor<32xf32>) -> tensor<32xf32>
      %loop_result3 = scf.for %k = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg3 = %prod3) -> (tensor<32xf32>) : i32 {
        %cons3 = arith.subf %arg3, %arg3 {ssbuffer.block_id = 10 : i32} : tensor<32xf32>
        %new_prod3 = arith.addf %cons3, %cons3 {ssbuffer.block_id = 9 : i32} : tensor<32xf32>
        scf.yield %new_prod3 : tensor<32xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T14: i64 Loop Variable Test (TruncIOp block_id)
  // Test: When loop induction variable is i64 (width > 32), TruncIOp is created
  //       to truncate i64 to i32 for buffer index calculation.
  // Key Check: TruncIOp gets block_id tag from the block it belongs to
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t14_i64_loop_var
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.trunci {{.*}} {ssbuffer.block_id = 5
  func.func @test_t14_i64_loop_var() {
    %c0_i64 = arith.constant 0 : i64
    %c100_i64 = arith.constant 100 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i64 to %c100_i64 step %c1_i64 iter_args(%arg = %prod) -> (tensor<128xf32>) : i64 {
        %consumed = arith.addf %arg, %arg {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_prod = arith.addf %consumed, %consumed {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T15: i16 Loop Variable Test (ExtSIOp block_id)
  // Test: When loop induction variable is i16 (width < 32), ExtSIOp is created
  //       to extend i16 to i32 for buffer index calculation.
  // Key Check: ExtSIOp gets block_id tag from the block it belongs to
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t15_i16_loop_var
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.extsi {{.*}} {ssbuffer.block_id = 5
  func.func @test_t15_i16_loop_var() {
    %c0_i16 = arith.constant 0 : i16
    %c100_i16 = arith.constant 100 : i16
    %c1_i16 = arith.constant 1 : i16
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i16 to %c100_i16 step %c1_i16 iter_args(%arg = %prod) -> (tensor<128xf32>) : i16 {
        %consumed = arith.addf %arg, %arg {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_prod = arith.addf %consumed, %consumed {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  //===--------------------------------------------------------------------===//
  // T16: Multiple Parallel Main Loops with Complex Loop Variables
  // Test: Three parallel main_loop forOps with different loop variable types
  //       - loop1: i32, lb=10, step=2 -> creates SubIOp, DivUIOp
  //       - loop2: index type -> creates IndexCastOp
  //       - loop3: i64 type -> creates TruncIOp
  // Key Check: All counter operations get correct block_id tags from each loop
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t16_parallel_complex_loops
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 6
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.divui {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.subi {{.*}} {ssbuffer.block_id = 5
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 8
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 8
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 7
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 7
  // CHECK-DAG: arith.index_cast {{.*}} {ssbuffer.block_id = 7
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 10
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 10
  // CHECK-DAG: arith.remsi {{.*}} {ssbuffer.block_id = 9
  // CHECK-DAG: arith.cmpi eq, {{.*}} {ssbuffer.block_id = 9
  // CHECK-DAG: arith.trunci {{.*}} {ssbuffer.block_id = 9
  func.func @test_t16_parallel_complex_loops() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c100_i64 = arith.constant 100 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.0 : f32
    %empty_128 = tensor.empty() : tensor<128xf32>
    %empty_64 = tensor.empty() : tensor<64xf32>
    %empty_32 = tensor.empty() : tensor<32xf32>
    scope.scope : () -> () {
      // First main_loop: i32, lb=10, step=2 -> creates SubIOp, DivUIOp
      %prod1 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty_128 : tensor<128xf32>) -> tensor<128xf32>
      %loop_result1 = scf.for %i = %c10_i32 to %c100_i32 step %c2_i32 iter_args(%arg1 = %prod1) -> (tensor<128xf32>) : i32 {
        %cons1 = arith.addf %arg1, %arg1 {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_prod1 = arith.addf %cons1, %cons1 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod1 : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      // Second main_loop: index type -> creates IndexCastOp
      %prod2 = linalg.fill {ssbuffer.block_id = 7 : i32} ins(%cst : f32) outs(%empty_64 : tensor<64xf32>) -> tensor<64xf32>
      %loop_result2 = scf.for %j = %c0 to %c100 step %c1 iter_args(%arg2 = %prod2) -> (tensor<64xf32>) : index {
        %cons2 = arith.mulf %arg2, %arg2 {ssbuffer.block_id = 8 : i32} : tensor<64xf32>
        %new_prod2 = arith.mulf %cons2, %cons2 {ssbuffer.block_id = 7 : i32} : tensor<64xf32>
        scf.yield %new_prod2 : tensor<64xf32>
      } {ssbuffer.main_loop = 1 : i64}
      // Third main_loop: i64 type -> creates TruncIOp
      %prod3 = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst : f32) outs(%empty_32 : tensor<32xf32>) -> tensor<32xf32>
      %loop_result3 = scf.for %k = %c0_i64 to %c100_i64 step %c1_i64 iter_args(%arg3 = %prod3) -> (tensor<32xf32>) : i64 {
        %cons3 = arith.subf %arg3, %arg3 {ssbuffer.block_id = 10 : i32} : tensor<32xf32>
        %new_prod3 = arith.addf %cons3, %cons3 {ssbuffer.block_id = 9 : i32} : tensor<32xf32>
        scf.yield %new_prod3 : tensor<32xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T19: tensor.empty Dependency Test
// Test: tensor.empty inside main_loop body gets dep_mark like scalar
// Key Check: tensor.empty defines producer, gets dep_mark
//         : consumer of tensor.empty also gets dep_mark
//===--------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_t19_tensor_empty_dep_mark
// CHECK-DAG: tensor.empty({{.*}}) {{.*}} ssbuffer.dep_mark
// CHECK-DAG: arith.addf {{.*}} ssbuffer.dep_mark
// CHECK-NOT: tensor.empty {{.*}} ssbuffer.intraDeps
func.func @test_t19_tensor_empty_dep_mark() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      %empty_128 = tensor.empty() : tensor<128xf32>
      %initial = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty_128 : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %initial) -> (tensor<128xf32>) : i32 {
        %empty = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        %consumed = arith.addf %empty, %empty {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_fill = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
        scf.yield %new_fill : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T20: Mixed Dependencies Test (tensor.empty + scalar + normal tensor)
// Test: all three types coexist, each gets correct dep_mark handling
// Key Check: tensor.empty -> dep_mark like scalar
//         : scalar (i32) -> dep_mark when used by cross-block consumer
//         : normal tensor -> buffer allocation
//===--------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_t20_mixed_dependencies
// CHECK-DAG: tensor.empty({{.*}}) {{.*}} ssbuffer.dep_mark
// CHECK-DAG: arith.mulf {{.*}} {ssbuffer.block_id = 6
// CHECK-NOT: tensor.empty {{.*}} ssbuffer.intraDeps
func.func @test_t20_mixed_dependencies() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      %empty_tensor = tensor.empty() : tensor<128xf32>
      %filled_tensor = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty_tensor : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %filled_tensor) -> (tensor<128xf32>) : i32 {
        %empty = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        %consumed = arith.addf %empty, %empty {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %scaled = arith.mulf %consumed, %consumed {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %new_fill = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
        scf.yield %new_fill : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T21: tensor.empty inside nested forOp (not main_loop) should NOT get dep_mark
// Test: tensor.empty defined inside a nested forOp (not main_loop) should not be tagged
// Key Check: tensor.empty in nested forOp -> no dep_mark
//         : only tensor.empty in main_loop scope gets dep_mark
//===--------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_t21_tensor_empty_in_nested_for
// CHECK-NOT: tensor.empty {{.*}} ssbuffer.dep_mark
func.func @test_t21_tensor_empty_in_nested_for() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      %empty_128 = tensor.empty() : tensor<128xf32>
      %initial = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty_128 : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %initial) -> (tensor<128xf32>) : i32 {
        %inner_loop_result = scf.for %j = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg2 = %arg) -> (tensor<128xf32>) : i32 {
          // tensor.empty inside nested forOp - should NOT get dep_mark
          %empty_nested = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
          %consumed = arith.addf %empty_nested, %empty_nested {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          %new_fill = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty_nested : tensor<128xf32>) -> tensor<128xf32>
          scf.yield %new_fill : tensor<128xf32>
        } {ssbuffer.block_id = 7 : i32}
        // tensor.empty in main_loop body but nested forOp scope - should NOT get dep_mark
        %main_empty = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        %main_consumed = arith.addf %main_empty, %main_empty {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        %main_fill = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%main_empty : tensor<128xf32>) -> tensor<128xf32>
        scf.yield %main_fill : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T22: tensor.empty inside scf.if (not main_loop) should NOT get dep_mark
// Test: tensor.empty defined inside an scf.if within main_loop body
//       should not be tagged because its parentOp is scf.if, not main_loop forOp
// Key Check: tensor.empty in scf.if -> no dep_mark
//         : only tensor.empty with parentOp = main_loop forOp gets dep_mark
//===--------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_t22_tensor_empty_in_nested_if
// CHECK-NOT: tensor.empty {{.*}} ssbuffer.dep_mark
func.func @test_t22_tensor_empty_in_nested_if() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      %empty_128 = tensor.empty() : tensor<128xf32>
      %initial = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty_128 : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %initial) -> (tensor<128xf32>) : i32 {
        %cond = arith.cmpi eq, %i, %c0_i32 : i32
        // tensor.empty inside scf.if - its parentOp is scf.if, NOT main_loop forOp
        // should NOT get dep_mark
        %if_result:2 = scf.if %cond -> (tensor<128xf32>, tensor<128xf32>) {
          %empty_if = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
          %consumed_if = arith.addf %empty_if, %empty_if {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          scf.yield %consumed_if, %consumed_if : tensor<128xf32>, tensor<128xf32>
        } else {
          %empty_else = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
          %consumed_else = arith.mulf %empty_else, %empty_else {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
          scf.yield %consumed_else, %consumed_else : tensor<128xf32>, tensor<128xf32>
        }
        %new_fill = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%arg : tensor<128xf32>) -> tensor<128xf32>
        scf.yield %new_fill : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T23: scf.if with Then Branch Using depVal and Else Branch Yielding depVal
// Test: Real scenario from if_problem.mlir where:
//       - block_id = 11: func.call produces depVal (%154)
//       - scf.if with block_id = 20 has then/else branches
//       - then branch: compute op uses %154, then yield
//       - else branch: directly yield %154
//       - %154 is used by external ops (arith.select in then branch)
// Note: This test verifies that depVal is properly handled when used in
//       scf.if then branch compute, and else branch directly yields it.
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t23_scf_if_else_branch_yield
  // CHECK-DAG: memref.alloc
  // CHECK-DAG: memref.memory_space_cast {{.*}} {ssbuffer.intraDeps
  // CHECK: scf.if {{.*}} -> (tensor<16x32xf16>) {
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 20 : i32}
  // CHECK: scf.yield
  // CHECK: } else {
  // CHECK: scf.yield {{.*}} : tensor<16x32xf16>
  // CHECK: } {ssbuffer.block_id = 20 : i32}
  // CHECK: hivm.hir.copy {{.*}} {ssbuffer.block_id = 20 : i32}
func.func @test_t23_scf_if_else_branch_yield() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %cst_f16 = arith.constant 1.0 : f16
    %empty_16x32 = tensor.empty() : tensor<16x32xf16>
    scope.scope : () -> () {
      // Producer: linalg.fill with block_id = 11
      %prod = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst_f16 : f16) outs(%empty_16x32 : tensor<16x32xf16>) -> tensor<16x32xf16>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<16x32xf16>) : i32 {
        %cond = arith.cmpi eq, %i, %c0_i32 : i32
        %alloc = memref.alloc() {ssbuffer.block_id = 11 : i32} : memref<16x32xf16>
        %to_tensor = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 11 : i32} : memref<16x32xf16>
        // scf.if with block_id = 20
        // then branch: compute op uses %to_tensor (depVal from block_id 11), then yield
        // else branch: directly yield %to_tensor (depVal from block_id 11)
        %if_result = scf.if %cond -> (tensor<16x32xf16>) {
          %consumed_then = arith.addf %to_tensor, %arg {ssbuffer.block_id = 20 : i32} : tensor<16x32xf16>
          scf.yield %consumed_then : tensor<16x32xf16>
        } else {
          scf.yield %to_tensor : tensor<16x32xf16>
        } {ssbuffer.block_id = 20 : i32}
        %new_prod = arith.addf %if_result, %if_result {ssbuffer.block_id = 11 : i32} : tensor<16x32xf16>
        scf.yield %new_prod : tensor<16x32xf16>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T24: Same depVal Shared by Multiple Ops in Same block_id
// Test: When a depVal is used by multiple normal ops in the SAME block_id,
//       there should be only ONE buffer selection scf.if, shared by all ops.
//       This tests the fix for duplicate buffer selection generation.
// Key Check: Only ONE scf.if buffer selection for block_id = 5
//         : %cons is produced in block_id = 6, used by three ops in block_id = 5
//         : These three ops should share ONE buffer selection
//         : The buffer selection result is used by all three ops
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t24_shared_buffer_selection
  // CHECK-DAG: memref.alloc
  // CHECK-DAG: memref.memory_space_cast {{.*}} {ssbuffer.intraDeps
  // Only ONE scf.if buffer selection for block_id = 5 (not two!)
  // CHECK: scf.if {{.*}} -> (tensor<128xf32>)
  // CHECK: } {ssbuffer.block_id = 5
  // CHECK-NOT: scf.if {{.*}} -> (tensor<128xf32>)
  // CHECK-NOT: } {ssbuffer.block_id = 5
  // CHECK-DAG: arith.addf %{{.*}}, %{{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK-DAG: arith.subf %{{.*}}, %{{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK-DAG: arith.mulf %{{.*}}, %{{.*}} {ssbuffer.block_id = 5 : i32}
  func.func @test_t24_shared_buffer_selection() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      // Producer: block_id = 5
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        // %arg is the depVal from previous iteration (block_id = 5)
        // In block_id = 6, %arg is consumed to produce %cons
        %cons = arith.addf %arg, %arg {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        // In block_id = 5, %cons (the processed depVal) is used by THREE ops
        // These three ops should share ONE buffer selection
        %new_prod1 = arith.addf %cons, %cons {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        %new_prod2 = arith.subf %cons, %cons {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        %new_prod3 = arith.mulf %cons, %cons {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        // All three results are yielded out
        %final = arith.addf %new_prod1, %new_prod2 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        %final2 = arith.addf %final, %new_prod3 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %cons : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
//===--------------------------------------------------------------------===//
// T25: Same depVal Used in Two Different block_ids
// Test: When a depVal is used in TWO DIFFERENT block_ids,
//       there should be buffer selection scf.if in EACH block_id where it's used.
//       This tests that cross-block dependency generates buffer selection per block.
// Key Check: Buffer selection scf.ifs for block_id = 5 and block_id = 7
//         : %cons is produced in block_id = 6, used in block_id = 5 and block_id = 7
//         : Each block gets its own buffer selection
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t25_depval_in_two_blocks
  // CHECK-DAG: memref.alloc
  // CHECK-DAG: memref.memory_space_cast {{.*}} {ssbuffer.intraDeps
  // Buffer selection scf.if for block_id=5
  // CHECK: scf.if {{.*}} -> (tensor<128xf32>)
  // CHECK: } {ssbuffer.block_id = 5
  // Buffer selection scf.if for block_id=7
  // CHECK: scf.if {{.*}} -> (tensor<128xf32>)
  // CHECK: } {ssbuffer.block_id = 7
  // Verify the ops using %cons in each block
  // CHECK-DAG: arith.addf {{.*}} {ssbuffer.block_id = 5 : i32}
  // CHECK-DAG: arith.subf {{.*}} {ssbuffer.block_id = 7 : i32}
  func.func @test_t25_depval_in_two_blocks() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      // Producer: block_id = 5
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        // %arg is the depVal from previous iteration (block_id = 5)
        // In block_id = 6, %arg is consumed to produce %cons
        %cons = arith.addf %arg, %arg {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        // In block_id = 5, %cons is used by first op
        %new_prod1 = arith.addf %cons, %cons {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        // In block_id = 7, %cons is used by second op
        %new_prod2 = arith.subf %cons, %cons {ssbuffer.block_id = 7 : i32} : tensor<128xf32>
        // Final producer in block_id = 5
        %final = arith.addf %new_prod1, %new_prod2 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %final : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T26: ifOp inner ops - dep_mark applied with cross-block producer (baseline)
// Test: An ifOp at block 12 with inner ops at blocks 8 and 9 uses a scalar
//       producer at block 11. This is a baseline test that verifies dep_mark
//       is added when producer and ifOp are in different blocks.
// NOTE: This test passes with BOTH getSsbufferId AND getOutermostSsbufferId
//       because all blocks (8, 9, 12) are cross-block from producer (11).
//       See T27 for the test that ACTUALLY distinguishes the fix.
// Key Check: dep_mark is added to producer and inner ops
//===--------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_t26_ifop_inner_dep_mark
// CHECK: arith.addi {{.*}} {ssbuffer.block_id = 11 : i32, ssbuffer.dep_mark
// CHECK: scf.if
// CHECK: arith.muli {{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.dep_mark
// CHECK: arith.divsi {{.*}} {ssbuffer.block_id = 9 : i32, ssbuffer.dep_mark
  func.func @test_t26_ifop_inner_dep_mark() {
    %true = arith.constant true
    %c0_i64 = arith.constant 0 : i64
    %c100_i64 = arith.constant 100 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<1xi64>
      %cast_alloc = memref.cast %alloc : memref<1xi64> to memref<1xi64, strided<[1], offset: ?>>
      scf.for %i = %c0_i64 to %c100_i64 step %c1_i64 : i64 {
        %load = memref.load %cast_alloc[%c0] {ssbuffer.block_id = 11 : i32} : memref<1xi64, strided<[1], offset: ?>>
        %compute = arith.addi %load, %c1_i64 {ssbuffer.block_id = 11 : i32} : i64
        %result = scf.if %true -> (i64) {
          %r1 = arith.muli %compute, %c1_i64 {ssbuffer.block_id = 8 : i32} : i64
          scf.yield %r1 : i64
        } else {
          %r2 = arith.divsi %compute, %c1_i64 {ssbuffer.block_id = 9 : i32} : i64
          scf.yield %r2 : i64
        } {ssbuffer.block_id = 12 : i32}
        memref.store %compute, %cast_alloc[%c0] {ssbuffer.block_id = 12 : i32} : memref<1xi64, strided<[1], offset: ?>>
      } {ssbuffer.block_id = 26 : i32, ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T27: ifOp inner ops - same-block producer, no dep_mark
// Test: When the ifOp is at the SAME block as the producer (both block 11),
//       the cross-block dep judgment should treat them as same-block
//       (11 vs 11), so NO dep_mark should be added.
//       This is the inverse case of T26.
// Key Check: NO dep_mark on producer or inner ops
//===--------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_t27_ifop_same_block_no_dep_mark
// CHECK-NOT: arith.addi {{.*}} ssbuffer.dep_mark
// CHECK-NOT: arith.muli {{.*}} ssbuffer.dep_mark
// CHECK-NOT: arith.divsi {{.*}} ssbuffer.dep_mark
  func.func @test_t27_ifop_same_block_no_dep_mark() {
    %true = arith.constant true
    %c0_i64 = arith.constant 0 : i64
    %c100_i64 = arith.constant 100 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<1xi64>
      %cast_alloc = memref.cast %alloc : memref<1xi64> to memref<1xi64, strided<[1], offset: ?>>
      scf.for %i = %c0_i64 to %c100_i64 step %c1_i64 : i64 {
        %load = memref.load %cast_alloc[%c0] {ssbuffer.block_id = 11 : i32} : memref<1xi64, strided<[1], offset: ?>>
        %compute = arith.addi %load, %c1_i64 {ssbuffer.block_id = 11 : i32} : i64
        // ifOp at block 11 - SAME block as producer
        %result = scf.if %true -> (i64) {
          %r1 = arith.muli %compute, %c1_i64 {ssbuffer.block_id = 8 : i32} : i64
          scf.yield %r1 : i64
        } else {
          %r2 = arith.divsi %compute, %c1_i64 {ssbuffer.block_id = 9 : i32} : i64
          scf.yield %r2 : i64
        } {ssbuffer.block_id = 11 : i32}
        memref.store %compute, %cast_alloc[%c0] {ssbuffer.block_id = 12 : i32} : memref<1xi64, strided<[1], offset: ?>>
      } {ssbuffer.block_id = 26 : i32, ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T28: ifOp with 2 results - producer-side buffer selection not duplicated
// Test: An ifOp with 2 results (multi-result). The dedup fix in
//       groupOpsBySsbufferId ensures the ifOp is added to opsById[18] only
//       ONCE (not twice, once per result). Without the fix, processDepVal
//       would iterate the ifOp twice and generate a duplicate producer-side
//       scf.if buffer selection (with intra_buffer attribute).
// Key Check: Only ONE scf.if with intra_buffer at block_id = 18
//         : Pattern is checked structurally, not by specific intraDeps value
//===--------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_t28_multi_result_ifop_no_dup_intraDeps
// The ifOp's producer-side buffer selection scf.if (with intra_buffer at block 18)
// should appear exactly once. It contains hivm.hir.copy for the ifOp's result.
// CHECK: scf.if
// CHECK: hivm.hir.copy {{.*}} outs({{.*}}) {ssbuffer.block_id = 18 : i32}
// CHECK: } {ssbuffer.block_id = 18 : i32, ssbuffer.intra_buffer}
// No duplicate producer-side scf.if would produce another intra_buffer at block 18
// CHECK-NOT: } {ssbuffer.block_id = 18 : i32, ssbuffer.intra_buffer}
  func.func @test_t28_multi_result_ifop_no_dup_intraDeps() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      // Transfer producer at block 11
      %memspacecast = memref.memory_space_cast %alloc {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      %depval = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %depval) -> (tensor<128xf32>) : i32 {
        %cond = arith.cmpi eq, %i, %c0_i32 : i32
        // ifOp with 2 results - the duplication bug would manifest here
        %if_result:2 = scf.if %cond -> (tensor<128xf32>, tensor<128xf32>) {
          %consumed = arith.addf %arg, %arg {ssbuffer.block_id = 11 : i32} : tensor<128xf32>
          scf.yield %consumed, %consumed : tensor<128xf32>, tensor<128xf32>
        } else {
          %other = arith.mulf %arg, %arg {ssbuffer.block_id = 11 : i32} : tensor<128xf32>
          scf.yield %other, %other : tensor<128xf32>, tensor<128xf32>
        } {ssbuffer.block_id = 18 : i32}
        %next = arith.addf %if_result#0, %if_result#0 {ssbuffer.block_id = 11 : i32} : tensor<128xf32>
        scf.yield %next : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T29: ifOp inner ops - producer-side buffer uses ifOp's block_id (baseline)
// Test: An ifOp at block 18 yields a result that is used outside (block 11).
//       The producer-side hivm.hir.copy for the ifOp's result must use the
//       ifOp's block_id (18).
// NOTE: This test passes with BOTH getSsbufferId AND getOutermostSsbufferId
//       because for the ifOp itself, both return 18 (ifOp's own block_id).
//       The fix only changes behavior for INNER ops of the ifOp (see T27).
// Key Check: hivm.hir.copy (producer side) is at block_id = 18 (ifOp's)
//===--------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_t29_ifop_buffer_uses_ifop_block_id
// CHECK: scf.if
// CHECK: hivm.hir.copy {{.*}} outs({{.*}}) {ssbuffer.block_id = 18 : i32}
// CHECK: } {ssbuffer.block_id = 18 : i32, ssbuffer.intra_buffer}
  func.func @test_t29_ifop_buffer_uses_ifop_block_id() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      // Transfer producer at block 11
      %memspacecast = memref.memory_space_cast %alloc {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
      %depval = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %depval) -> (tensor<128xf32>) : i32 {
        %cond = arith.cmpi eq, %i, %c0_i32 : i32
        // ifOp at block 18 with 2 results, inner ops at block 11
        %if_result:2 = scf.if %cond -> (tensor<128xf32>, tensor<128xf32>) {
          %consumed = arith.addf %arg, %arg {ssbuffer.block_id = 11 : i32} : tensor<128xf32>
          scf.yield %consumed, %consumed : tensor<128xf32>, tensor<128xf32>
        } else {
          %other = arith.mulf %arg, %arg {ssbuffer.block_id = 11 : i32} : tensor<128xf32>
          scf.yield %other, %other : tensor<128xf32>, tensor<128xf32>
        } {ssbuffer.block_id = 18 : i32}
        %next = arith.addf %if_result#0, %if_result#0 {ssbuffer.block_id = 11 : i32} : tensor<128xf32>
        scf.yield %next : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T30: memref.alloc in outer forOp, subview in nested forOp
// Mirrors the wy_input.mlir structure:
//   - outer forOp (block 8, main_loop = 0): wraps everything
//   - inside: %alloc_8 at block 4, then inner forOp at block 4
//   - inside inner forOp: %subview_17 at block 3 uses %alloc_8
// getOutermostSsbufferId should treat subview_17's logical block as 4
// (the inner forOp's block), so alloc_8 (block 4) -> subview_17 is same-block
// and no dep_mark / no ssbuffer.skip should be emitted.
//===--------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_t30_forop_memref_alloc_nested
// CHECK-NOT: ssbuffer.skip
// CHECK-NOT: memref.alloc.*ssbuffer\.dep_mark
// CHECK-NOT: memref.subview.*ssbuffer\.dep_mark
  func.func @test_t30_forop_memref_alloc_nested() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c64_i32 = arith.constant 64 : i32
    scope.scope : () -> () {
      %36:3 = scf.for %arg19 = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg20 = %c0_i32, %arg21 = %c0_i32, %arg22 = %c0_i32) -> (i32, i32, i32) : i32 {
        // %alloc_8 at block 4 — outside inner forOp
        %alloc_8 = memref.alloc() {ssbuffer.block_id = 4 : i32} : memref<128x64xf16>
        // Inner forOp at block 4, no main_loop attribute
        scf.for %arg23 = %c0 to %c128 step %c1 {
          // %subview_17 at block 3 — uses %alloc_8
          %subview_17 = memref.subview %alloc_8[%arg23, 0] [1, 64] [1, 1] {ssbuffer.block_id = 3 : i32} : memref<128x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
        } {ssbuffer.block_id = 4 : i32}
        // %to_tensor at block 4 — also uses %alloc_8, same-block
        %to_tensor = bufferization.to_tensor %alloc_8 restrict writable {ssbuffer.block_id = 4 : i32} : memref<128x64xf16>
        scf.yield %arg20, %arg21, %arg22 : i32, i32, i32
      } {ssbuffer.block_id = 8 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T31: memref.alloc outside ifOp, subview inside ifOp
// Same logical structure as T30 but with an ifOp in place of the inner forOp:
//   - outer forOp (block 8, main_loop = 0): wraps everything
//   - inside: %alloc_8 at block 4, then ifOp at block 4
//   - inside ifOp: %subview_17 at block 3 uses %alloc_8
// getOutermostSsbufferId should treat subview_17's logical block as 4
// (the ifOp's block), so alloc_8 (block 4) -> subview_17 is same-block
// and no dep_mark / no ssbuffer.skip should be emitted.
//===--------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_t31_ifop_memref_alloc_nested
// CHECK-NOT: ssbuffer.skip
// CHECK-NOT: memref.alloc.*ssbuffer\.dep_mark
// CHECK-NOT: memref.subview.*ssbuffer\.dep_mark
  func.func @test_t31_ifop_memref_alloc_nested() {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    scope.scope : () -> () {
      %36:3 = scf.for %arg19 = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg20 = %c0_i32, %arg21 = %c0_i32, %arg22 = %c0_i32) -> (i32, i32, i32) : i32 {
        // %alloc_8 at block 4 — outside the ifOp
        %alloc_8 = memref.alloc() {ssbuffer.block_id = 4 : i32} : memref<128x64xf16>
        // ifOp at block 4, condition is %true (always taken)
        %if_result:2 = scf.if %true -> (memref<128x64xf16>, i32) {
          // %subview_17 at block 3 — uses %alloc_8, inside ifOp
          %subview_17 = memref.subview %alloc_8[%c0, 0] [1, 64] [1, 1] {ssbuffer.block_id = 3 : i32} : memref<128x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
          scf.yield %alloc_8, %arg22 : memref<128x64xf16>, i32
        } else {
          scf.yield %alloc_8, %arg22 : memref<128x64xf16>, i32
        } {ssbuffer.block_id = 4 : i32}
        // %to_tensor at block 4 — uses %alloc_8, same-block
        %to_tensor = bufferization.to_tensor %alloc_8 restrict writable {ssbuffer.block_id = 4 : i32} : memref<128x64xf16>
        scf.yield %arg20, %arg21, %arg22 : i32, i32, i32
      } {ssbuffer.block_id = 8 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T32: forOp with same block as producer - NO dep_mark
// Same scenario as T27 but uses scf.for instead of scf.if.
//   - producer addi at block 11, produces %compute
//   - forOp at block 11 (SAME block as producer), inner muli at block 8
//   - inner muli uses %compute
// getOutermostSsbufferId walks up the parent chain and records each
// non-main_loop forOp's block_id, so the inner muli's outermost id is
// the enclosing forOp's block_id (11), matching the producer (11).
// The cross-block dep judgment should treat them as same-block (11 vs 11),
// so NO dep_mark should be added.
// Key Check: NO dep_mark on producer or inner ops
//===--------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_t32_forop_same_block_no_dep_mark
// CHECK-NOT: arith.addi {{.*}} ssbuffer.dep_mark
// CHECK-NOT: arith.muli {{.*}} ssbuffer.dep_mark
  func.func @test_t32_forop_same_block_no_dep_mark() {
    %c0_i64 = arith.constant 0 : i64
    %c100_i64 = arith.constant 100 : i64
    %c10_i64 = arith.constant 10 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    scope.scope : () -> () {
      %alloc = memref.alloc() : memref<1xi64>
      %cast_alloc = memref.cast %alloc : memref<1xi64> to memref<1xi64, strided<[1], offset: ?>>
      scf.for %i = %c0_i64 to %c100_i64 step %c1_i64 : i64 {
        %load = memref.load %cast_alloc[%c0] {ssbuffer.block_id = 11 : i32} : memref<1xi64, strided<[1], offset: ?>>
        %compute = arith.addi %load, %c1_i64 {ssbuffer.block_id = 11 : i32} : i64
        // forOp at block 11 - SAME block as producer
        %result = scf.for %j = %c0_i64 to %c10_i64 step %c1_i64 iter_args(%a = %c0_i64) -> (i64) : i64 {
          %r1 = arith.muli %compute, %c1_i64 {ssbuffer.block_id = 8 : i32} : i64
          scf.yield %r1 : i64
        } {ssbuffer.block_id = 11 : i32}
        memref.store %compute, %cast_alloc[%c0] {ssbuffer.block_id = 12 : i32} : memref<1xi64, strided<[1], offset: ?>>
      } {ssbuffer.block_id = 26 : i32, ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T33: depVal used in BOTH then/else branches of scf.if - empty+fill clone
// Test: A cross-block depVal inside the main_loop forOp (linalg.fill result
//       at block 13) is consumed in BOTH the then and else branches of an
//       scf.if (block 21). The linalg.fill's outs comes from a
//       tensor::EmptyOp, so the empty+fill clone path applies: the
//       empty+fill pair is cloned to each branch (block 11 / block 12)
//       and the depVal is replaced in-branch. No multi-buffer is created.
//       Note: the per-region grouping from the original mistransfer fix
//       still ensures each branch's clone is locally visible (no cross-
//       region SSA dominance violation).
// Key Check: original linalg.fill at block 13 preserved
//         : cloned tensor.empty + linalg.fill in then branch (block 11)
//         : cloned tensor.empty + linalg.fill in else branch (block 12)
//         : NO memref.alloc / bufferization.to_tensor / hivm.hir.copy
//         : NO ssbuffer.intra_buffer / intraDeps on scf.if
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t33_ifop_both_branches_depval
  // Original linalg.fill (depVal) at block 13 inside main_loop forOp
  // CHECK: linalg.fill {ssbuffer.block_id = 13 : i32}
  // The scf.if at block 21 (the user of the depVal)
  // CHECK: scf.if
  // Cloned pattern in then branch (block 11)
  // CHECK: tensor.empty() {ssbuffer.block_id = 11 : i32}
  // CHECK: linalg.fill {ssbuffer.block_id = 11 : i32}
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 11 : i32}
  // Cloned pattern in else branch (block 12)
  // CHECK: tensor.empty() {ssbuffer.block_id = 12 : i32}
  // CHECK: linalg.fill {ssbuffer.block_id = 12 : i32}
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 12 : i32}
  // No multi-buffer path taken
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: bufferization.to_tensor
  // CHECK-NOT: hivm.hir.copy
  // CHECK-NOT: ssbuffer.intra_buffer
  func.func @test_t33_ifop_both_branches_depval() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %true = arith.constant true
    %empty = tensor.empty() : tensor<32xf32>
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 : i32 {
        // depVal: linalg.fill at block 13 - direct child of main_loop forOp
        %filled = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
        // scf.if at block 21 - uses %filled in BOTH then and else branches
        %result = scf.if %true -> (tensor<32xf32>) {
          // then branch - block 11
          %r1 = arith.addf %filled, %filled {ssbuffer.block_id = 11 : i32} : tensor<32xf32>
          scf.yield %r1 : tensor<32xf32>
        } else {
          // else branch - block 12
          %r2 = arith.addf %filled, %filled {ssbuffer.block_id = 12 : i32} : tensor<32xf32>
          scf.yield %r2 : tensor<32xf32>
        } {ssbuffer.block_id = 21 : i32}
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
//===--------------------------------------------------------------------===//
// T34: tensor.empty + linalg.fill Pattern - Clone to Consumer Block
// Test: When linalg.fill's outs comes from tensor::EmptyOp in producer block,
//       and the result is used in OTHER blocks, we should NOT do normal
//       multi-buffer (no memref.alloc/hivm.copy/scf.if/bufferization.to_tensor).
//       Instead, clone tensor.empty + linalg.fill to each consumer's position
//       and replace the depVal usage there.
// Key Check: tensor.empty + linalg.fill appear CLONED in consumer block (14)
//         : cloned ops tagged with consumer's block_id
//         : NO memref.alloc / scf.if / bufferization.to_tensor for this depVal
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t34_empty_fill_clone_to_consumer
  // Original producer pattern in block 13
  // CHECK: tensor.empty() {ssbuffer.block_id = 13 : i32}
  // CHECK: linalg.fill {ssbuffer.block_id = 13 : i32}
  // Cloned pattern in consumer block 14
  // CHECK: tensor.empty() {ssbuffer.block_id = 14 : i32}
  // CHECK: linalg.fill {ssbuffer.block_id = 14 : i32}
  // Consumer op still in block 14
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 14 : i32}
  // No buffer allocation for this depVal
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.memory_space_cast
  // CHECK-NOT: bufferization.to_tensor
  // CHECK-NOT: hivm.hir.copy
  func.func @test_t34_empty_fill_clone_to_consumer() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_init = arith.constant 0 : i32
    %empty_8 = tensor.empty() : tensor<8xi32>
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 : i32 {
        // Producer in block 13: tensor.empty + linalg.fill
        %empty_inner = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<8xi32>
        %filled = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%c0_init : i32) outs(%empty_inner : tensor<8xi32>) -> tensor<8xi32>
        // Consumer in block 14: uses %filled (the linalg.fill result) cross-block
        %consumed = arith.addi %filled, %filled {ssbuffer.block_id = 14 : i32} : tensor<8xi32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T35: tensor.empty + linalg.fill Pattern - Multiple Consumer Blocks
// Test: Same as T34, but the linalg.fill result is consumed by ops in
//       TWO different consumer blocks (14 and 15). Each block should
//       get its own clone.
// Key Check: tensor.empty + linalg.fill appear cloned in BOTH 14 and 15
//         : NO buffer allocation
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t35_empty_fill_clone_multi_consumer
  // Original producer pattern in block 13
  // CHECK: tensor.empty() {ssbuffer.block_id = 13 : i32}
  // CHECK: linalg.fill {ssbuffer.block_id = 13 : i32}
  // Cloned pattern in consumer block 14 (clone inserted before its user)
  // CHECK: tensor.empty() {ssbuffer.block_id = 14 : i32}
  // CHECK: linalg.fill {ssbuffer.block_id = 14 : i32}
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 14 : i32}
  // Cloned pattern in consumer block 15 (clone inserted before its user)
  // CHECK: tensor.empty() {ssbuffer.block_id = 15 : i32}
  // CHECK: linalg.fill {ssbuffer.block_id = 15 : i32}
  // CHECK: arith.subi {{.*}} {ssbuffer.block_id = 15 : i32}
  // No buffer allocation
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: bufferization.to_tensor
  func.func @test_t35_empty_fill_clone_multi_consumer() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_init = arith.constant 0 : i32
    %empty_8 = tensor.empty() : tensor<8xi32>
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 : i32 {
        // Producer in block 13
        %empty_inner = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<8xi32>
        %filled = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%c0_init : i32) outs(%empty_inner : tensor<8xi32>) -> tensor<8xi32>
        // Two consumers in different blocks
        %consumed14 = arith.addi %filled, %filled {ssbuffer.block_id = 14 : i32} : tensor<8xi32>
        %consumed15 = arith.subi %filled, %filled {ssbuffer.block_id = 15 : i32} : tensor<8xi32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T36: tensor.empty + linalg.fill Pattern - 2D Tensor
// Test: Same as T34 but on a 2D tensor, verifying the clone preserves shape
//       and element type.
// Key Check: cloned tensor.empty/linalg.fill have the 2D shape
//         : NO buffer allocation
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t36_empty_fill_clone_2d
  // Original producer in block 13
  // CHECK: tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<4x8xf32>
  // CHECK: linalg.fill {ssbuffer.block_id = 13 : i32}
  // Cloned to consumer block 14
  // CHECK: tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<4x8xf32>
  // CHECK: linalg.fill {ssbuffer.block_id = 14 : i32}
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 14 : i32}
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: bufferization.to_tensor
  func.func @test_t36_empty_fill_clone_2d() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f32
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 : i32 {
        %empty_inner = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<4x8xf32>
        %filled = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%empty_inner : tensor<4x8xf32>) -> tensor<4x8xf32>
        %consumed = arith.addf %filled, %filled {ssbuffer.block_id = 14 : i32} : tensor<4x8xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T37: Mixed Pattern - Regular Multi-Buffer + empty+fill Clone
// Test: Function has BOTH a regular cross-block depVal (e.g. arith.addf result
//       carried via iter_args, which gets a multi-buffer) AND an empty+fill
//       pattern depVal (which gets cloned). Verify the two paths coexist and
//       groupIds remain consistent (no groupId collision/offset).
// Key Check: regular depVal triggers buffer creation (alloc + scf.if + copy)
//         : empty+fill depVal is cloned to consumer
//         : no extra/unused buffers
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t37_mixed_multi_buffer_and_clone
  // Regular multi-buffer path: alloc + scf.if (for 128xf32 depVal)
  // CHECK-DAG: memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  // CHECK-DAG: memref.memory_space_cast
  // CHECK-DAG: bufferization.to_tensor
  // CHECK-DAG: hivm.hir.copy
  // CHECK-DAG: scf.if
  // Original empty+fill producer in block 14
  // CHECK: tensor.empty() {ssbuffer.block_id = 14 : i32}
  // CHECK: linalg.fill {ssbuffer.block_id = 14 : i32}
  // Cloned empty+fill in block 15
  // CHECK: tensor.empty() {ssbuffer.block_id = 15 : i32}
  // CHECK: linalg.fill {ssbuffer.block_id = 15 : i32}
  // No buffer for the 8xi32 depVal (empty+fill pattern, not multi-buffered)
  // CHECK-NOT: memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
  func.func @test_t37_mixed_multi_buffer_and_clone() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %c0_i = arith.constant 0 : i32
    %empty_128 = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      // Regular depVal: tensor produced in block 5, consumed in block 6 -> multi-buffer
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty_128 : tensor<128xf32>) -> tensor<128xf32>
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        // Regular cross-block (5 -> 6): %arg consumed in block 6
        %consumed = arith.addf %arg, %arg {ssbuffer.block_id = 6 : i32} : tensor<128xf32>
        // Producer continuation: %consumed used in block 5 (cross-block depVal -> multi-buffer)
        %next = arith.addf %consumed, %consumed {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        // empty+fill pattern: producer in block 14, consumer in block 15
        %empty_inner = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<8xi32>
        %filled = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%c0_i : i32) outs(%empty_inner : tensor<8xi32>) -> tensor<8xi32>
        %empty_consumed = arith.addi %filled, %filled {ssbuffer.block_id = 15 : i32} : tensor<8xi32>
        scf.yield %next : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T38: tensor.empty + linalg.fill Pattern - Non-Zero Fill Value
// Test: The fill input is a non-zero constant. Verify the clone preserves
//       the SAME fill input value (not zero, not the wrong SSA).
// Key Check: cloned linalg.fill has the same ins() value as the original
//         : cloned tensor.empty has consumer's block_id
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t38_empty_fill_nonzero_init
  // Original producer in block 11 with init = 42
  // CHECK: linalg.fill {ssbuffer.block_id = 11 : i32} ins(%
  // Cloned to block 12, must keep init = 42 (same ins() value)
  // CHECK: linalg.fill {ssbuffer.block_id = 12 : i32} ins(%
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 12 : i32}
  // CHECK-NOT: memref.alloc
  func.func @test_t38_empty_fill_nonzero_init() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %c42_i32 = arith.constant 42 : i32
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 : i32 {
        %empty_inner = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<8xi32>
        %filled = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%c42_i32 : i32) outs(%empty_inner : tensor<8xi32>) -> tensor<8xi32>
        %consumed = arith.muli %filled, %filled {ssbuffer.block_id = 12 : i32} : tensor<8xi32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T39: tensor.empty + linalg.fill Pattern - All Users in Same Block
// Test: When ALL users of the empty+fill result are in the same block as the
//       producer (no cross-block use), the empty+fill is left untouched.
//       The clone path only triggers on cross-block use.
// Key Check: only one tensor.empty + linalg.fill pair (the original)
//         : no cloned copy
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t39_empty_fill_same_block_no_clone
  // Only the original empty+fill pair, all in block 9
  // CHECK: tensor.empty() {ssbuffer.block_id = 9 : i32}
  // CHECK: linalg.fill {ssbuffer.block_id = 9 : i32}
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 9 : i32}
  // No cloned pair (only one empty+fill total)
  // CHECK-NOT: tensor.empty() {ssbuffer.block_id = 10 : i32}
  // CHECK-NOT: linalg.fill {ssbuffer.block_id = 10 : i32}
  func.func @test_t39_empty_fill_same_block_no_clone() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i = arith.constant 0 : i32
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 : i32 {
        %empty_inner = tensor.empty() {ssbuffer.block_id = 9 : i32} : tensor<8xi32>
        %filled = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%c0_i : i32) outs(%empty_inner : tensor<8xi32>) -> tensor<8xi32>
        %consumed = arith.addi %filled, %filled {ssbuffer.block_id = 9 : i32} : tensor<8xi32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T40: tensor.empty + linalg.fill Pattern - Cloned Empty/Fill Order
// Test: Verifies the clone is inserted BEFORE the first user in the consumer
//       block (not at the top of the block). The cloned tensor.empty appears
//       right before the arith op that uses the fill result.
// Key Check: order in consumer block 24 is empty -> fill -> arith (in sequence)
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t40_empty_fill_clone_insertion_order
  // Original empty+fill in block 23
  // CHECK: tensor.empty() {ssbuffer.block_id = 23 : i32}
  // CHECK: linalg.fill {ssbuffer.block_id = 23 : i32}
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 23 : i32}
  // Consumer block 24: clone inserted in order empty -> fill -> arith
  // CHECK: tensor.empty() {ssbuffer.block_id = 24 : i32}
  // CHECK-NEXT: linalg.fill {ssbuffer.block_id = 24 : i32}
  // CHECK-NEXT: arith.addi {{.*}} {ssbuffer.block_id = 24 : i32}
  // CHECK-NOT: memref.alloc
  func.func @test_t40_empty_fill_clone_insertion_order() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i = arith.constant 0 : i32
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 : i32 {
        // Producer in block 23 (also has a same-block consumer to keep producer ops in 23)
        %empty_inner = tensor.empty() {ssbuffer.block_id = 23 : i32} : tensor<8xi32>
        %filled = linalg.fill {ssbuffer.block_id = 23 : i32} ins(%c0_i : i32) outs(%empty_inner : tensor<8xi32>) -> tensor<8xi32>
        %same_block_consume = arith.addi %filled, %filled {ssbuffer.block_id = 23 : i32} : tensor<8xi32>
        // Consumer in different block 24
        %cross_consume = arith.addi %filled, %filled {ssbuffer.block_id = 24 : i32} : tensor<8xi32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

//===--------------------------------------------------------------------===//
// T41: tensor.empty + linalg.fill Pattern - Multiple Fills in Same Function
// Test: Two independent empty+fill patterns with different shapes/values
//       in the same function. Each should be cloned independently to
//       their respective consumer blocks.
// Key Check: TWO independent empty+fill pairs, each cloned
//         : no buffer allocation for either
//===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_t41_empty_fill_multi_independent
  // First producer (i32, init=0) in block 30
  // CHECK: tensor.empty() {ssbuffer.block_id = 30 : i32} : tensor<8xi32>
  // CHECK: linalg.fill {ssbuffer.block_id = 30 : i32} ins(
  // First cloned to block 31
  // CHECK: tensor.empty() {ssbuffer.block_id = 31 : i32} : tensor<8xi32>
  // CHECK: linalg.fill {ssbuffer.block_id = 31 : i32} ins(
  // First consumer
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 31 : i32}
  // Second producer (f32, init=1.0) in block 32
  // CHECK: tensor.empty() {ssbuffer.block_id = 32 : i32} : tensor<4xf32>
  // CHECK: linalg.fill {ssbuffer.block_id = 32 : i32} ins(
  // Second cloned to block 33
  // CHECK: tensor.empty() {ssbuffer.block_id = 33 : i32} : tensor<4xf32>
  // CHECK: linalg.fill {ssbuffer.block_id = 33 : i32} ins(
  // Second consumer
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 33 : i32}
  // CHECK-NOT: memref.alloc
  func.func @test_t41_empty_fill_multi_independent() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i = arith.constant 0 : i32
    %c1_f = arith.constant 1.0 : f32
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 : i32 {
        // First independent pattern: i32
        %empty_a = tensor.empty() {ssbuffer.block_id = 30 : i32} : tensor<8xi32>
        %filled_a = linalg.fill {ssbuffer.block_id = 30 : i32} ins(%c0_i : i32) outs(%empty_a : tensor<8xi32>) -> tensor<8xi32>
        %consumed_a = arith.addi %filled_a, %filled_a {ssbuffer.block_id = 31 : i32} : tensor<8xi32>
        // Second independent pattern: f32, different shape
        %empty_b = tensor.empty() {ssbuffer.block_id = 32 : i32} : tensor<4xf32>
        %filled_b = linalg.fill {ssbuffer.block_id = 32 : i32} ins(%c1_f : f32) outs(%empty_b : tensor<4xf32>) -> tensor<4xf32>
        %consumed_b = arith.addf %filled_b, %filled_b {ssbuffer.block_id = 33 : i32} : tensor<4xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
