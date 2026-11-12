// RUN: triton-opt --add-control-flow-condition --canonicalize %s | FileCheck %s --check-prefix=DIRECT
// RUN: sed 's|// ADD-EXTRA: ||' %s | triton-opt --add-control-flow-condition --canonicalize | FileCheck %s --check-prefix=EXTRA
// RUN: sed 's|// ADD-WRITE: ||' %s | triton-opt --add-control-flow-condition --canonicalize | FileCheck %s --check-prefix=WRITE

// Publish in each branch only when there is no extra user or intervening write.
// DIRECT-LABEL: func.func @publication
// DIRECT: scf.if
// DIRECT: sync_block_wait
// DIRECT-NEXT: hivm.hir.copy
// DIRECT-NEXT: hivm.hir.sync_block_set
// DIRECT: } else {
// DIRECT: sync_block_wait
// DIRECT-NEXT: hivm.hir.copy
// DIRECT-NEXT: hivm.hir.sync_block_set
// DIRECT-NOT: sync_block_wait
// DIRECT: return
// EXTRA-LABEL: func.func @publication
// EXTRA: arith.select
// EXTRA: sync_block_wait
// EXTRA-NEXT: hivm.hir.copy
// EXTRA-NEXT: hivm.hir.sync_block_set
// EXTRA: hivm.hir.copy
// EXTRA: return
// WRITE-LABEL: func.func @publication
// WRITE: arith.select
// WRITE: sync_block_wait
// WRITE-NEXT: memref.store
// WRITE-NEXT: hivm.hir.copy
// WRITE-NEXT: hivm.hir.sync_block_set
// WRITE: return
module attributes {ssbuffer.outer_main_loop, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @publication(%partial: i1, %p0: tensor<4x4xbf16>,
                          %p1: tensor<4x4xbf16>, %r0: tensor<4xf32>,
                          %r1: tensor<4xf32>, %rows: memref<4xf32>,
                          %l1: memref<4x4xbf16, #hivm.address_space<cbuf>>,
                          %n: index, %extra: memref<4x4xbf16>, %side: memref<1xindex>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scope.scope : () -> () {
      scf.for %i = %zero to %n step %one {
        %values:2 = scf.if %partial -> (tensor<4x4xbf16>, tensor<4xf32>) {
          scf.yield %p0, %r0 : tensor<4x4xbf16>, tensor<4xf32>
        } else {
          scf.yield %p1, %r1 : tensor<4x4xbf16>, tensor<4xf32>
        }
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
        // ADD-WRITE: memref.store %one, %side[%zero] : memref<1xindex>
        hivm.hir.copy ins(%values#0 : tensor<4x4xbf16>) outs(%l1 : memref<4x4xbf16, #hivm.address_space<cbuf>>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
        // ADD-EXTRA: hivm.hir.copy ins(%values#0 : tensor<4x4xbf16>) outs(%extra : memref<4x4xbf16>)
        hivm.hir.copy ins(%values#1 : tensor<4xf32>) outs(%rows : memref<4xf32>)
      } {ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
