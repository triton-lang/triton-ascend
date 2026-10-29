// RUN: triton-opt %s --add-scalar-pipe-control | FileCheck %s

// `hivm.hir.try_wait` is only available on the register based architectures.
// On the memory based ones (A3) the LLVM export rejects the op, so
// AddScalarPipeControl must not build a guard there at all: the main loop is
// left exactly as it was.

// CHECK-LABEL: func.func @mem_based_arch
// CHECK-NOT: hivm.hir.try_wait
// CHECK-NOT: scf.if
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-NOT: scf.if
// CHECK-NOT: hivm.hir.try_wait
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @mem_based_arch() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> (index) {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      scf.yield %arg2 : index
    } {ssbuffer.block_id = 15 : i32, ssbuffer.main_loop = 0 : i32}
    return
  }
}
