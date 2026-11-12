// RUN: triton-opt --split-input-file --add-control-flow-condition --canonicalize %s | FileCheck %s --implicit-check-not=memref_ext.volatile

// A single slot cannot support two producer sweeps under one grant. Keep
// the scalar handshake in each sweep, along with the native data flags.
// CHECK-LABEL: func.func @native_credits
// CHECK: scf.for
// CHECK: sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>]
// CHECK: arith.cmpi ult, {{.*}}, %arg0 : i64
// CHECK: sync_block_wait {{.*}}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK: sync_block_set {{.*}}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK: sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>]
// CHECK: } {{.*}}ssbuffer.main_loop
// CHECK: scf.for
// CHECK: sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>]
// CHECK: arith.cmpi ult, {{.*}}, %arg0 : i64
// CHECK: sync_block_wait {{.*}}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK: sync_block_set {{.*}}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK: sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>]
// CHECK: } {{.*}}ssbuffer.main_loop

module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">,
                   ssbuffer.intra_buf_count = 1 : i32,
                   ssbuffer.inter_core_buf_count = 1 : i32} {
  func.func @native_credits(%total: i64, %seed: tensor<4xf32>,
                            %shared: memref<4xf32>, %sink: memref<4xf32>) {
    annotation.mark %sink {test.load_annotation_dialect} : memref<4xf32>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    scope.scope : () -> () {
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      %result = scf.for %iv = %zero to %total step %one
          iter_args(%event = %zero) -> i64 : i64 {
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        hivm.hir.copy ins(%shared : memref<4xf32>) outs(%sink : memref<4xf32>)
          {ssbuffer.block_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
        scf.yield %event : i64
      } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %result = scf.for %iv = %zero to %total step %one
          iter_args(%event = %zero) -> i64 : i64 {
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
        hivm.hir.copy ins(%seed : tensor<4xf32>) outs(%shared : memref<4xf32>)
          {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        scf.yield %event : i64
      } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}

// -----

// With two independent slots the two sweeps can share one scalar grant.
// Keep the per-slot flags and release the final grant even for an odd count.
// CHECK-LABEL: func.func @two_slots
// CHECK: scf.for %[[SWEEP:[a-zA-Z0-9_]+]] = {{.*}} to %[[BOUND:.*]] step
// CHECK: arith.andi %[[SWEEP]],
// CHECK: scf.if
// CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>]
// CHECK: sync_block_wait {{.*}}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK: sync_block_wait {{.*}}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: arith.cmpi eq, {{.*}}, %[[BOUND]] : i64
// CHECK: arith.ori
// CHECK: scf.if
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>]
// CHECK: scf.for
// CHECK: arith.andi
// CHECK: scf.if
// CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>]
// CHECK: sync_block_set {{.*}}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK: sync_block_set {{.*}}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: arith.ori
// CHECK: scf.if
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>]

module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">,
                   ssbuffer.intra_buf_count = 1 : i32,
                   ssbuffer.inter_core_buf_count = 2 : i32} {
  func.func @two_slots(%total: i64, %seed: tensor<4xf32>,
                       %slot0: memref<4xf32>, %slot1: memref<4xf32>, %sink: memref<4xf32>) {
    annotation.mark %sink {test.load_annotation_dialect} : memref<4xf32>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    scope.scope : () -> () {
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      %result = scf.for %iv = %zero to %total step %one
          iter_args(%event = %zero) -> i64 : i64 {
        %mod = arith.andi %iv, %one {ssbuffer.block_id = 1 : i32} : i64
        %even = arith.cmpi eq, %mod, %zero {ssbuffer.block_id = 1 : i32} : i64
        %src = scf.if %even -> memref<4xf32> {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
          scf.yield %slot0 : memref<4xf32>
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          scf.yield %slot1 : memref<4xf32>
        } {ssbuffer.block_id = 1 : i32}
        hivm.hir.copy ins(%src : memref<4xf32>) outs(%sink : memref<4xf32>)
          {ssbuffer.block_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 0 : i32}
        scf.if %even {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
        } else {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
        } {ssbuffer.block_id = 1 : i32}
        scf.yield %event : i64
      } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %result = scf.for %iv = %zero to %total step %one
          iter_args(%event = %zero) -> i64 : i64 {
        %mod = arith.andi %iv, %one {ssbuffer.block_id = 2 : i32} : i64
        %even = arith.cmpi eq, %mod, %zero {ssbuffer.block_id = 2 : i32} : i64
        scf.if %even {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
          hivm.hir.copy ins(%seed : tensor<4xf32>) outs(%slot0 : memref<4xf32>)
            {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32}
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
          hivm.hir.copy ins(%seed : tensor<4xf32>) outs(%slot1 : memref<4xf32>)
            {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32}
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        } {ssbuffer.block_id = 2 : i32}
        scf.yield %event : i64
      } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
