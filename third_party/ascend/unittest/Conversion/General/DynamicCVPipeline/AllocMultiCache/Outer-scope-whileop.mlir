// RUN: triton-opt --add_multi_buffer_outer_scope %s | FileCheck %s

// Test: C→V transfer with scf.while (main_loop) in single-buffer mode.
// Pass must not crash; IR structure (whileOp, main_loop, TCB marks) preserved.

// CHECK-LABEL: func.func @tc_while_ctov_sender
// CHECK: scf.while
// CHECK: ssbuffer.main_loop
// CHECK: tightly_coupled_buffer

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @tc_while_ctov_sender() {
  %c0_i32 = arith.constant 0 : i32
  %c100_i32 = arith.constant 100 : i32
  %c1_i32 = arith.constant 1 : i32
  // --- VECTOR scope (receiver for C→V: memspace_cast reads from ub) ---
  scope.scope : () -> () {
    %buf_ub = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.while (%iter = %c0_i32) : (i32) -> i32 {
      %cond = arith.cmpi slt, %iter, %c100_i32 : i32
      scf.condition(%cond) %iter : i32
    } do {
    ^bb0(%iter: i32):
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
      %buf = memref.memory_space_cast %buf_ub {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %buf restrict writable : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      %next_iter = arith.addi %iter, %c1_i32 : i32
      scf.yield %next_iter : i32
    } attributes {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
  hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
  // --- CUBE scope (sender for C→V: fixpipe writes to ub) ---
  scope.scope : () -> () {
    %buf_cc = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<cc>>
    %buf_ub = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
      hivm.hir.fixpipe {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} ins(%buf_cc : memref<128xf16, #hivm.address_space<cc>>) outs(%buf_ub : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}


// Test: Both sender and receiver use scf.while with main_loop
// CHECK-LABEL: func.func @tc_while_both_sides
// CHECK: scf.while
// CHECK: ssbuffer.main_loop

func.func @tc_while_both_sides() {
  %c0_i32 = arith.constant 0 : i32
  %c100_i32 = arith.constant 100 : i32
  %c1_i32 = arith.constant 1 : i32
  // --- VECTOR scope ---
  scope.scope : () -> () {
    %buf_ub = memref.alloc() {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.while (%iter = %c0_i32) : (i32) -> i32 {
      %cond = arith.cmpi slt, %iter, %c100_i32 : i32
      scf.condition(%cond) %iter : i32
    } do {
    ^bb0(%iter: i32):
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
      %buf = memref.memory_space_cast %buf_ub {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %buf restrict writable : memref<128xf16> to tensor<128xf16>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
      %next_iter = arith.addi %iter, %c1_i32 : i32
      scf.yield %next_iter : i32
    } attributes {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
  hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
  // --- CUBE scope ---
  scope.scope : () -> () {
    %buf_cc = memref.alloc() {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<cc>>
    %buf_ub = memref.alloc() {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %buf_ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.while (%iter = %c0_i32) : (i32) -> i32 {
      %cond = arith.cmpi slt, %iter, %c100_i32 : i32
      scf.condition(%cond) %iter : i32
    } do {
    ^bb0(%iter: i32):
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
      hivm.hir.fixpipe {ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32} ins(%buf_cc : memref<128xf16, #hivm.address_space<cc>>) outs(%buf_ub : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 40 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
      %next_iter = arith.addi %iter, %c1_i32 : i32
      scf.yield %next_iter : i32
    } attributes {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
}
