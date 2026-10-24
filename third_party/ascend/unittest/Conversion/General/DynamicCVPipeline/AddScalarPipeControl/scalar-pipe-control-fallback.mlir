// RUN: triton-opt %s --add-scalar-pipe-control | FileCheck %s

// AddScalarPipeControl bails out on a module that carries the dynamic CV
// pipeline error code attribute: such a module already fell back to a
// different execution mode, so nothing is guarded.
//
// The attribute lives on the module, so this case needs its own file.

// CHECK: triton_ascend.dynamic_cv_pipeline.rc = 1
// CHECK-LABEL: func.func @fallback_module
// CHECK-NOT: scf.if
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-NOT: scf.if
module attributes {"triton_ascend.dynamic_cv_pipeline.rc" = 1 : i32} {
  func.func @fallback_module() {
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
