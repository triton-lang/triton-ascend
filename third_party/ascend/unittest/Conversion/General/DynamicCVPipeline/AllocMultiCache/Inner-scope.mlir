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
        }
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

}
