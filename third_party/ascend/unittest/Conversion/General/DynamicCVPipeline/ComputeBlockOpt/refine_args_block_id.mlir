// RUN: triton-opt --refine-args-block-id %s | FileCheck %s

module {
  // Test1: Basic scenario - yield op can be moved to first user's block
  // The yield def op only depends on the iter_arg itself, should be moved to first user's block_id

  // CHECK-LABEL: func.func @test_basic_refine
  // CHECK: scf.for
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK-NOT: arith.addi {{.*}} {ssbuffer.block_id = 2 : i32
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: scf.yield
  // CHECK: } {ssbuffer.main_loop = 1 : i64}

  func.func @test_basic_refine(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %init = arith.addi %arg0, %c1_i32 {ssbuffer.block_id = 0 : i32} : i32

    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%iter = %init) -> (i32) {
      %use1 = arith.addi %iter, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      %use2 = arith.muli %use1, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      %update = arith.addi %iter, %c1_i32 {ssbuffer.block_id = 2 : i32} : i32
      scf.yield %update : i32
    } {ssbuffer.main_loop = 1 : i64}

    return %result : i32
  }

  // Test2: Dependency scenario - yield def op depends on other ops, should NOT be moved

  // CHECK-LABEL: func.func @test_depends_other
  // CHECK: scf.for
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 3 : i32
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 2 : i32
  // CHECK: scf.yield
  // CHECK: } {ssbuffer.main_loop = 1 : i64}

  func.func @test_depends_other(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %init = arith.addi %arg0, %c1_i32 {ssbuffer.block_id = 0 : i32} : i32

    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%iter = %init) -> (i32) {
      %use1 = arith.addi %iter, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      %use2 = arith.muli %use1, %c1_i32 {ssbuffer.block_id = 3 : i32} : i32
      // This yield def op depends on use2 (another op in loop), should NOT be moved
      %update = arith.addi %use2, %c1_i32 {ssbuffer.block_id = 2 : i32} : i32
      scf.yield %update : i32
    } {ssbuffer.main_loop = 1 : i64}

    return %result : i32
  }

  // Test3: Same block scenario - yield def op and first user in same block, no change needed

  // CHECK-LABEL: func.func @test_same_block
  // CHECK: scf.for
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: scf.yield
  // CHECK: } {ssbuffer.main_loop = 1 : i64}

  func.func @test_same_block(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %init = arith.addi %arg0, %c1_i32 {ssbuffer.block_id = 0 : i32} : i32

    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%iter = %init) -> (i32) {
      // First user in block 1
      %use1 = arith.addi %iter, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      // Yield def op also in block 1 - same as first user, no change needed
      %update = arith.addi %iter, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      scf.yield %update : i32
    } {ssbuffer.main_loop = 1 : i64}

    return %result : i32
  }

  // Test4: Multiple iter_args - different refinement for each

  // CHECK-LABEL: func.func @test_multiple_iter_args
  // CHECK: scf.for
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK-NOT: arith.addi {{.*}} {ssbuffer.block_id = 3 : i32
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 2 : i32
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 3 : i32
  // CHECK: scf.yield
  // CHECK: } {ssbuffer.main_loop = 1 : i64}

  func.func @test_multiple_iter_args(%arg0: i32) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %init1 = arith.addi %arg0, %c1_i32 {ssbuffer.block_id = 0 : i32} : i32
    %init2 = arith.muli %arg0, %c1_i32 {ssbuffer.block_id = 0 : i32} : i32

    %result:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%iter1 = %init1, %iter2 = %init2) -> (i32, i32) {
      %use1 = arith.addi %iter1, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      %use1b = arith.muli %use1, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      %update1 = arith.addi %iter1, %c1_i32 {ssbuffer.block_id = 3 : i32} : i32
      %use2 = arith.muli %iter2, %c1_i32 {ssbuffer.block_id = 2 : i32} : i32
      %update2 = arith.muli %use2, %c1_i32 {ssbuffer.block_id = 3 : i32} : i32

      scf.yield %update1, %update2 : i32, i32
    } {ssbuffer.main_loop = 1 : i64}

    return %result#0, %result#1 : i32, i32
  }

  // Test5: Not main_loop - should not be processed

  // CHECK-LABEL: func.func @test_not_main_loop
  // CHECK: scf.for
  // CHECK-NOT: {ssbuffer.main_loop}
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 2 : i32
  // CHECK: scf.yield

  func.func @test_not_main_loop(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %init = arith.addi %arg0, %c1_i32 {ssbuffer.block_id = 0 : i32} : i32

    // No ssbuffer.main_loop attribute, should not be processed
    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%iter = %init) -> (i32) {
      %use1 = arith.addi %iter, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      %update = arith.addi %iter, %c1_i32 {ssbuffer.block_id = 2 : i32} : i32
      scf.yield %update : i32
    }

    return %result : i32
  }

  // Test6: Yield directly yields iter_arg (identity case)

  // CHECK-LABEL: func.func @test_yield_iter_arg
  // CHECK: scf.for
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: scf.yield
  // CHECK: } {ssbuffer.main_loop = 1 : i64}

  func.func @test_yield_iter_arg(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %init = arith.addi %arg0, %c1_i32 {ssbuffer.block_id = 0 : i32} : i32

    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%iter = %init) -> (i32) {
      %use1 = arith.addi %iter, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      // Directly yield iter_arg, no refinement needed
      scf.yield %iter : i32
    } {ssbuffer.main_loop = 1 : i64}

    return %result : i32
  }

  // Test7: Multiple users of iter_arg - refine to first user

  // CHECK-LABEL: func.func @test_multiple_users
  // CHECK: scf.for
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: arith.muli {{.*}} {ssbuffer.block_id = 2 : i32
  // CHECK-NOT: arith.addi {{.*}} {ssbuffer.block_id = 3 : i32
  // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: scf.yield
  // CHECK: } {ssbuffer.main_loop = 1 : i64}

  func.func @test_multiple_users(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %init = arith.addi %arg0, %c1_i32 {ssbuffer.block_id = 0 : i32} : i32

    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%iter = %init) -> (i32) {
      // First user (appears first in the block)
      %use1 = arith.addi %iter, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      // Second user
      %use2 = arith.muli %iter, %c1_i32 {ssbuffer.block_id = 2 : i32} : i32
      // Update in block 3, should be refined to block 1 (first user)
      %update = arith.addi %iter, %c1_i32 {ssbuffer.block_id = 3 : i32} : i32
      scf.yield %update : i32
    } {ssbuffer.main_loop = 1 : i64}

    return %result : i32
  }

  // Test8: Use other iteration args - yield def op depends on different iter_arg
  // arg11's yield op (new11) only depends on arg11 itself, should be moved to block 1 (first user of arg11)
  // arg10's yield op (new10) depends on use10 (op result), should NOT be moved
  // arg12's yield op (new12) depends on arg10 (different iter_arg), should NOT be moved

  // CHECK-LABEL: func.func @use_other_args_test
  // CHECK: scf.for
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 2 : i32
  // CHECK: arith.mulf {{.*}} {ssbuffer.block_id = 3 : i32
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 4 : i32
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 4 : i32
  // CHECK-NOT: arith.addf {{.*}} {ssbuffer.block_id = 5 : i32
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 1 : i32
  // CHECK: arith.addf {{.*}} {ssbuffer.block_id = 6 : i32
  // CHECK: scf.yield
  // CHECK: } {ssbuffer.main_loop = 0 : i32}

  func.func @use_other_args_test(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<16xf32>, %arg4: memref<16xf32>, %arg5: memref<16xf32>, %init10: f32, %init11: f32, %init12: f32) {
    scope.scope : () -> ()  {
      %c0 = arith.constant {ssbuffer.block_id = 0 : i32}  0 : index
      %results:3 = scf.for %iv = %arg0 to %arg1 step %arg2 iter_args(%arg10 = %init10, %arg11 = %init11, %arg12 = %init12) -> (f32, f32, f32) {
        %add11_op1 = arith.addf %arg11, %arg11 {ssbuffer.block_id = 1 : i32} : f32
        %mul12_op1 = arith.mulf %arg12, %arg12 {ssbuffer.block_id = 2 : i32} : f32
        %add11_op2 = arith.mulf %arg11, %add11_op1 {ssbuffer.block_id = 3 : i32} : f32
        %use10 = arith.addf %arg10, %mul12_op1 {ssbuffer.block_id = 4 : i32} : f32
        %new10 = arith.addf %arg10, %use10 {ssbuffer.block_id = 4 : i32} : f32
        %new11 = arith.addf %arg11, %arg11 {ssbuffer.block_id = 5 : i32} : f32
        %new12 = arith.addf %arg10, %arg10 {ssbuffer.block_id = 6 : i32} : f32
        scf.yield %new10, %new11, %new12 : f32, f32, f32
      } {ssbuffer.main_loop = 0 : i32}

    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    func.return
  }
}
