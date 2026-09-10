// RUN: triton-opt --add-control-flow-condition --verify-each %s | FileCheck %s --check-prefix=SHARED --implicit-check-not=triton_ascend.dynamic_cv_pipeline.rc
// RUN: sed 's/ssbuffer.shared_page_write, //g' %s | triton-opt --add-control-flow-condition --verify-each | FileCheck %s --check-prefix=ORDINARY --implicit-check-not=triton_ascend.dynamic_cv_pipeline.rc

// RUN: sed 's/ssbuffer.shared_page_write, /ssbuffer.shared_page_slots = 2 : i32, ssbuffer.shared_page_write, /g' %s | triton-opt --add-control-flow-condition --verify-each | FileCheck %s --check-prefix=RING --implicit-check-not=triton_ascend.dynamic_cv_pipeline.rc

// RUN: sed 's/module attributes {/module attributes {ssbuffer.reserved_bytes = 1024 : i64,/' %s | triton-opt --add-control-flow-condition --verify-each | FileCheck %s --check-prefix=NO-SPACE --implicit-check-not=hivm.hir.pointer_cast
// NO-SPACE: triton_ascend.dynamic_cv_pipeline.rc
// NO-SPACE: func.func @shared_page_warmup

// RING-LABEL: func.func @shared_page_warmup
// RING: arith.cmpi ne,
// RING: ssbuffer.if = 1 : i32
// RING: arith.cmpi sge,
// RING: arith.cmpi sge,
// RING: ssbuffer.if = 3 : i32

// QK -> VECTOR -> PV has two cross-core slots and three ordinary intra-core
// slots, which normally delays PV until two QK iterations have completed.
// The additional single shared V slot must instead allow PV to consume the
// first iteration: QK cannot advance again until PV releases V.
// SHARED-LABEL: func.func @shared_page_warmup
// SHARED: arith.cmpi eq,
// SHARED: ssbuffer.if = 1 : i32
// SHARED-NOT: arith.cmpi sge,
// SHARED: ssbuffer.if = 3 : i32
// ORDINARY-LABEL: func.func @shared_page_warmup
// ORDINARY: ssbuffer.if = 1 : i32
// ORDINARY: arith.cmpi sge,
// ORDINARY: %[[TWO:.*]] = arith.constant 2 : i32
// ORDINARY: arith.muli %{{.*}}, %[[TWO]]
// ORDINARY: arith.cmpi sge,
// ORDINARY: ssbuffer.if = 3 : i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @shared_page_warmup(%shared_v: memref<1xi32>, %ordinary: memref<1xi32>, %qk: memref<1xi32>, %p: memref<1xi32>) {
    annotation.mark %shared_v {effects = ["read", "write"]} : memref<1xi32>
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %step = arith.constant 16 : i32
    %upper = arith.constant 48 : i32
    scope.scope : () -> () {
      scf.for %iv = %zero to %upper step %step : i32 {
        hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        memref.store %iv, %shared_v[%c0] {ssbuffer.shared_page_write, ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<1xi32>
        memref.store %iv, %ordinary[%c0] {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]} : memref<1xi32>
        memref.store %iv, %ordinary[%c0] {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]} : memref<1xi32>
        memref.store %iv, %ordinary[%c0] {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [1 : i32, 1 : i32]} : memref<1xi32>
        memref.store %iv, %qk[%c0] {ssbuffer.block_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} : memref<1xi32>
        memref.store %iv, %qk[%c0] {ssbuffer.block_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} : memref<1xi32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        %v = memref.load %shared_v[%c0] {ssbuffer.block_id = 3 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : memref<1xi32>
        %data = memref.load %ordinary[%c0] {ssbuffer.block_id = 3 : i32, ssbuffer.intraDeps = [1 : i32, 0 : i32]} : memref<1xi32>
        %prob = memref.load %p[%c0] {ssbuffer.block_id = 3 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32]} : memref<1xi32>
      } {ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      scf.for %iv = %zero to %upper step %step : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        %data = memref.load %qk[%c0] {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : memref<1xi32>
        memref.store %data, %p[%c0] {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} : memref<1xi32>
        hivm.hir.sync_block_set {ssbuffer.block_id = 2 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        memref.store %data, %p[%c0] {ssbuffer.block_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} : memref<1xi32>
      } {ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
