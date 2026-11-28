// RUN: triton-opt --split-input-file --add-control-flow-condition --canonicalize %s | FileCheck %s

// Reuse credit still gates Fixpipe after QK moves before the scalar turn gate.
// CHECK-LABEL: func.func @early_qk
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scf.for
// CHECK: sync_block_wait {{.*}}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK: hivm.hir.fixpipe
// CHECK: sync_block_set {{.*}}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK: sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>]
module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">,
                   ssbuffer.intra_buf_count = 1 : i32,
                   ssbuffer.inter_core_buf_count = 2 : i32} {
  func.func @early_qk(%total: i64, %seed: tensor<128x128xf32>,
                       %sink: memref<128x128xf32, #hivm.address_space<ub>>) {
    %slot0 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    %slot1 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    annotation.mark %sink {test.load_annotation_dialect} : memref<128x128xf32, #hivm.address_space<ub>>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    scope.scope : () -> () {
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      %result = scf.for %iv = %zero to %total step %one
          iter_args(%event = %zero) -> i64 : i64 {
        %mod = arith.andi %iv, %one {ssbuffer.block_id = 1 : i32} : i64
        %even = arith.cmpi eq, %mod, %zero {ssbuffer.block_id = 1 : i32} : i64
        %src = scf.if %even -> memref<128x128xf32, #hivm.address_space<ub>> {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
          scf.yield %slot0 : memref<128x128xf32, #hivm.address_space<ub>>
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          scf.yield %slot1 : memref<128x128xf32, #hivm.address_space<ub>>
        } {ssbuffer.block_id = 1 : i32}
        hivm.hir.copy ins(%src : memref<128x128xf32, #hivm.address_space<ub>>) outs(%sink : memref<128x128xf32, #hivm.address_space<ub>>)
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
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32}
            ins(%seed : tensor<128x128xf32>) outs(%slot0 : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32}
            ins(%seed : tensor<128x128xf32>) outs(%slot1 : memref<128x128xf32, #hivm.address_space<ub>>)
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

// -----
// An external ordinary write keeps the producer behind the scalar turn gate.
// CHECK-LABEL: func.func @external_write_keeps_wait
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scf.for
// CHECK: sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>]
// CHECK: memref.store
// CHECK: hivm.hir.fixpipe
module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">,
                   ssbuffer.intra_buf_count = 1 : i32,
                   ssbuffer.inter_core_buf_count = 2 : i32} {
  func.func @external_write_keeps_wait(%total: i64, %seed: tensor<128x128xf32>,
                       %sink: memref<128x128xf32, #hivm.address_space<ub>>) {
    %slot0 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    %slot1 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    annotation.mark %sink {test.load_annotation_dialect} : memref<128x128xf32, #hivm.address_space<ub>>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %index = arith.constant 0 : index
    %value = arith.constant 1.0 : f32
    scope.scope : () -> () {
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      %result = scf.for %iv = %zero to %total step %one
          iter_args(%event = %zero) -> i64 : i64 {
        %mod = arith.andi %iv, %one {ssbuffer.block_id = 1 : i32} : i64
        %even = arith.cmpi eq, %mod, %zero {ssbuffer.block_id = 1 : i32} : i64
        %src = scf.if %even -> memref<128x128xf32, #hivm.address_space<ub>> {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
          scf.yield %slot0 : memref<128x128xf32, #hivm.address_space<ub>>
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          scf.yield %slot1 : memref<128x128xf32, #hivm.address_space<ub>>
        } {ssbuffer.block_id = 1 : i32}
        hivm.hir.copy ins(%src : memref<128x128xf32, #hivm.address_space<ub>>) outs(%sink : memref<128x128xf32, #hivm.address_space<ub>>)
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
        memref.store %value, %sink[%index, %index] {ssbuffer.block_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        %even = arith.cmpi eq, %mod, %zero {ssbuffer.block_id = 2 : i32} : i64
        scf.if %even {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32}
            ins(%seed : tensor<128x128xf32>) outs(%slot0 : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32}
            ins(%seed : tensor<128x128xf32>) outs(%slot1 : memref<128x128xf32, #hivm.address_space<ub>>)
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
