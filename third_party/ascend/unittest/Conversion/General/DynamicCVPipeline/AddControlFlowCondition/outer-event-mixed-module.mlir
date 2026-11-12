// RUN: triton-opt --add-control-flow-condition --canonicalize %s | FileCheck %s

// An ordinary domain in the same module keeps readiness dispatch, while the
// normalized domain still gets a proven static ordinal on both cores.
// CHECK-LABEL: func.func @outer_domain
// CHECK: arith.cmpi ult, {{.*}}, %arg0 : i64
// CHECK-NOT: memref_ext.volatile
// CHECK: scope.scope
// CHECK: arith.cmpi ult, {{.*}}, %arg0 : i64
// CHECK-NOT: memref_ext.volatile
// CHECK: scope.scope
// CHECK: memref_ext.volatile
// CHECK: arith.cmpi slt
// CHECK: scope.scope
// CHECK: memref_ext.volatile
// CHECK: arith.cmpi slt
module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">,
                   ssbuffer.intra_buf_count = 1 : i32,
                   ssbuffer.inter_core_buf_count = 1 : i32} {
  func.func @outer_domain(%total: i64, %seed: tensor<4xf32>,
                          %shared: memref<4xf32>, %sink: memref<4xf32>) {
    annotation.mark %sink {test.load_annotation_dialect} : memref<4xf32>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    scope.scope : () -> () {
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      %r = scf.for %iv = %zero to %total step %one iter_args(%event = %zero) -> i64 : i64 {
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        hivm.hir.copy ins(%shared : memref<4xf32>) outs(%sink : memref<4xf32>)
          {ssbuffer.block_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
        scf.yield %event : i64
      } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %r = scf.for %iv = %zero to %total step %one iter_args(%event = %zero) -> i64 : i64 {
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
        hivm.hir.copy ins(%seed : tensor<4xf32>) outs(%shared : memref<4xf32>)
          {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        scf.yield %event : i64
      } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      scf.for %iv = %zero to %total step %one : i64 {
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.copy ins(%shared : memref<4xf32>) outs(%sink : memref<4xf32>)
          {ssbuffer.block_id = 3 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32], ssbuffer.transfer_id = 1 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      } {ssbuffer.main_loop = 1 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      scf.for %iv = %zero to %total step %one : i64 {
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.copy ins(%seed : tensor<4xf32>) outs(%shared : memref<4xf32>)
          {ssbuffer.block_id = 4 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
      } {ssbuffer.main_loop = 1 : i32}
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
