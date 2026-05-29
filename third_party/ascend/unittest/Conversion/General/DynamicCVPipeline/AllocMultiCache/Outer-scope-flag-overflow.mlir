// RUN: ! triton-opt --add_multi_buffer_outer_scope %s 2>&1 | FileCheck %s
// TC-FB02: flagCount > 15 causes signalPassFailure

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @tc_fb02_flag_gt_15() {
  %c0_i32 = arith.constant 0 : i32
  %c100_i32 = arith.constant 100 : i32
  %c1_i32 = arith.constant 1 : i32

  // Consume flags 0-15 (16 flags) to trigger pass failure
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 7
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 7
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 8
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 8
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 10
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 10
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 11
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 11
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 12
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 12
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 13
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 13
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 14
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 14
  hivm.hir.sync_block_set [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 15
  hivm.hir.sync_block_wait [<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 15

  return
}
}

// Verify the exact error message from flag budget check
// CHECK: flag count {{[0-9]+}} > 15, halting pass
