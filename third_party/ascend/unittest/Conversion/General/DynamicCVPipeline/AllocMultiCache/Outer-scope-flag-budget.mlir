// RUN: triton-opt --add_multi_buffer_outer_scope %s | FileCheck %s
// TC-FB01: flagCount > 7 forces single-buffer (crossDeps only, no double-buffering)

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @tc_fb01_flag_gt_7() {
  %c0_i32 = arith.constant 0 : i32
  %c100_i32 = arith.constant 100 : i32
  %c1_i32 = arith.constant 1 : i32

  // Consume flags 0-7 to trigger single-buffer fallback
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

  scope.scope : () -> () {
    %buf_ub = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
      %buf = memref.memory_space_cast %buf_ub {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %buf restrict writable : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
  hivm.hir.sync_block_wait {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
  scope.scope : () -> () {
    %buf_cc = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cc>>
    scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
      %buf_cbuf = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
      hivm.hir.fixpipe {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} ins(%buf_cc : memref<128xf16, #hivm.address_space<cc>>) outs(%buf_cbuf : memref<128xf16, #hivm.address_space<cbuf>>)
      hivm.hir.sync_block_set {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
}

// flagCount=8 > 7: single-buffer — crossDeps present, no double-buffering
// CHECK-LABEL: func.func @tc_fb01_flag_gt_7
// CHECK: crossDeps
// CHECK-NOT: tightly_coupled_buffer
// CHECK-NOT: cross_buffer
// CHECK-NOT: scf.if
