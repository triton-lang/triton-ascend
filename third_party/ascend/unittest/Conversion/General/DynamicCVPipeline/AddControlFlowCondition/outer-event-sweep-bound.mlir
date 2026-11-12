// RUN: triton-opt --add-control-flow-condition --canonicalize %s | FileCheck %s --check-prefix=LEGACY
// RUN: sed 's/, ssbuffer.outer_event_control_count = 1 : i32//' %s | triton-opt --add-control-flow-condition --canonicalize | FileCheck %s --check-prefix=LEGACY
// RUN: sed 's/ssbuffer.intraDeps = \[1 : i32, 0 : i32\]/ssbuffer.intraDeps = [1 : i32, 0 : i32, 0 : i32, 0 : i32]/' %s | triton-opt --add-control-flow-condition --canonicalize | FileCheck %s --check-prefix=LEGACY
// RUN: sed -e 's/ssbuffer.block_id = 1 : i32,/ssbuffer.block_id = 1 : i32, ssbuffer.crossCoreDeps = [3 : i32, 1 : i32],/' -e 's/ssbuffer.block_id = 4 : i32,/ssbuffer.block_id = 4 : i32, ssbuffer.crossCoreDeps = [3 : i32, 0 : i32, 4 : i32, 1 : i32],/' -e 's/ssbuffer.block_id = 3 : i32,/ssbuffer.block_id = 3 : i32, ssbuffer.crossCoreDeps = [4 : i32, 0 : i32],/' -e 's|// PIPE: ||' %s | triton-opt --add-control-flow-condition --canonicalize | FileCheck %s --check-prefix=LEGACY

// Missing native cross-core credits, multiple consumers, or a round trip that
// cannot sustain one event per sweep all retain the existing readiness bound.
// LEGACY-LABEL: func.func @event_sweep
// LEGACY: scope.scope
// LEGACY: arith.muli %arg0,
// LEGACY: scope.scope
// LEGACY: arith.muli %arg0,

module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">,
                   ssbuffer.intra_buf_count = 1 : i32,
                   ssbuffer.inter_core_buf_count = 1 : i32} {
  func.func @event_sweep(%total: i64, %seed: tensor<4xf32>,
                        %sink: memref<4xf32>) {
    annotation.mark %sink {test.load_annotation_dialect} : memref<4xf32>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    scope.scope : () -> () {
      %buffer = memref.alloc() : memref<4xf32>
      %result:2 = scf.for %iv = %zero to %total step %one
          iter_args(%event = %zero, %carry = %seed)
          -> (i64, tensor<4xf32>) : i64 {
        %read = arith.mulf %carry, %seed {ssbuffer.block_id = 1 : i32} : tensor<4xf32>
        hivm.hir.copy ins(%read : tensor<4xf32>) outs(%buffer : memref<4xf32>)
          {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]}
        // PIPE: hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
        hivm.hir.copy ins(%seed : tensor<4xf32>) outs(%buffer : memref<4xf32>)
          {ssbuffer.block_id = 2 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]}
        hivm.hir.copy ins(%seed : tensor<4xf32>) outs(%buffer : memref<4xf32>)
          {ssbuffer.block_id = 2 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]}
        // PIPE: hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.copy ins(%seed : tensor<4xf32>) outs(%sink : memref<4xf32>)
          {ssbuffer.block_id = 3 : i32, ssbuffer.intraDeps = [1 : i32, 0 : i32]}
        %next = arith.addf %seed, %seed {ssbuffer.block_id = 3 : i32} : tensor<4xf32>
        scf.yield %event, %next : i64, tensor<4xf32>
      } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %buffer = memref.alloc() : memref<4xf32>
      %result = scf.for %iv = %zero to %total step %one
          iter_args(%event = %zero) -> i64 : i64 {
        // PIPE: hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
        hivm.hir.copy ins(%seed : tensor<4xf32>) outs(%buffer : memref<4xf32>)
          {ssbuffer.block_id = 4 : i32, ssbuffer.intraDeps = [2 : i32, 1 : i32]}
        // PIPE: hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.copy ins(%seed : tensor<4xf32>) outs(%sink : memref<4xf32>)
          {ssbuffer.block_id = 5 : i32, ssbuffer.intraDeps = [2 : i32, 0 : i32]}
        scf.yield %event : i64
      } {ssbuffer.main_loop = 0 : i32, ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
