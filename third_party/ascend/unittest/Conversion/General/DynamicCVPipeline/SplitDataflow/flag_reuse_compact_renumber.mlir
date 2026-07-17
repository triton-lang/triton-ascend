// RUN: triton-opt --data-dependency-analysis --inter-core-transfer-and-sync %s | FileCheck %s --implicit-check-not="flag = -1"

module {
  func.func @flag_reuse_compact_renumber(%cond: i1) {
    scf.if %cond {
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
    } else {
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
    }
    hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 4
    hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
    hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 4
    hivm.hir.sync_block_wait {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
    return
  }
}

// CHECK-LABEL: func.func @flag_reuse_compact_renumber
// CHECK: {{flag = }}[[FIRST:[0-9]+]]{{$}}
// CHECK-COUNT-3: {{flag = }}[[FIRST]]{{$}}
// CHECK: {{flag = }}[[SECOND:[0-9]+]]{{$}}
// CHECK-NEXT: {{flag = }}[[SECOND]]{{$}}
// CHECK: hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
// CHECK-NEXT: hivm.hir.sync_block_wait {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
// CHECK-NEXT: hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 4
// CHECK-NEXT: hivm.hir.sync_block_wait {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
// CHECK: return
