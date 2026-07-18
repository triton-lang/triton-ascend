// RUN: triton-opt --add_multi_buffer_outer_scope %s | FileCheck %s

// Verify that in double-buffer mode (inter_core_buf_count = 2) the producer
// tag (crossDeps = [tid, 1]) lands on the cloned behavior op (hivm.hir.fixpipe)
// inside the scf.if branches, NOT on the outer scf.if wrapper.
// Consumer side (memory_space_cast + to_tensor chain) is unchanged: the
// consumer ifOp still carries crossDeps = [tid, 0] via wrapReceiverChainWithScfIf.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 2 : i32} {
func.func @tc_os_outer_producer_tag() {
  %c0_i32 = arith.constant 0 : i32
  %c100_i32 = arith.constant 100 : i32
  %c1_i32 = arith.constant 1 : i32
  scope.scope : () -> () {
    %buf_ub = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>>
    scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args() -> () : i32 {
      hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
      %buf = memref.memory_space_cast %buf_ub {ssbuffer.block_id = 10 : i32, ssbuffer.crossDeps = [1 : i32, 0 : i32], ssbuffer.transfer_id = 1 : i32} : memref<128xf16, #hivm.address_space<ub>> to memref<128xf16>
      %t = bufferization.to_tensor %buf restrict writable : memref<128xf16>
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
      hivm.hir.fixpipe {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32} ins(%buf_cc : memref<128xf16, #hivm.address_space<cc>>) outs(%buf_cbuf : memref<128xf16, #hivm.address_space<cbuf>>)
      hivm.hir.sync_block_set {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
      scf.yield
    } {ssbuffer.main_loop = 1 : i64}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
}

// ---------- Consumer side (VECTOR scope, comes first in output) ----------
// Consumer ifOp still carries [tid, 0] (unchanged behavior preserved):
// CHECK: } {ssbuffer.block_id = 10 : i32, ssbuffer.crossDeps = [1 : i32, 0 : i32], ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 1 : i32}

// ---------- Producer side (CUBE scope, comes second in output) ----------
// Both cloned fixpipe ops (then/else branches) carry [tid, 1]:
// CHECK: hivm.hir.fixpipe {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32} ins(%alloc
// CHECK: hivm.hir.fixpipe {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32} ins(%alloc

// ---------- Negative check ----------
// Producer scf.if wrapping the fixpipe must NOT carry crossDeps = [1 : i32, 1 : i32].
// The producer ifOp has cross_buffer = 1 + transfer_id but NO crossDeps:
// CHECK-NOT: } {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [1 : i32, 1 : i32], ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 1 : i32}