// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=8" 2>/dev/null | FileCheck %s --check-prefix=LANES8
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=8" 2>/dev/null | FileCheck %s --check-prefix=FLAGS8
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>/dev/null | FileCheck %s --check-prefix=LANES2
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>/dev/null | FileCheck %s --check-prefix=FLAGS2
//
// Flag demand is set by the transfer phases and the inter-core buffer depth,
// not by the unroll factor.  Lanes that share a buffer slot are already ordered
// by that buffer, so they share its forward and release flags too.
//
// Before this allocation every lane took its own flag, so unroll factor eight
// asked for 24 IDs against the 15 the hardware provides and was rejected
// outright.  It now uses the same six as every other factor.

// Eight lanes: eight QK matmuls and eight PV matmuls.
// LANES8-COUNT-16: linalg.matmul
// LANES8-NOT: linalg.matmul

// Three phases at depth two: six IDs, and nothing above them.
// FLAGS8-DAG: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
// FLAGS8-DAG: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// FLAGS8-DAG: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
// FLAGS8-DAG: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
// FLAGS8-DAG: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
// FLAGS8-DAG: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5

// Slot reuse is ordered by the schedule, not by extra synchronization, so no
// flag runs consumer-to-producer in either direction.
// FLAGS8-NOT: sync_block{{.*}}<PIPE_V>, <PIPE_FIX>
// FLAGS8-NOT: sync_block{{.*}}<PIPE_MTE1>, <PIPE_MTE3>
// FLAGS8-NOT: flag = 6
// FLAGS8-NOT: flag = 7

// Two lanes fill the two slots exactly once each, so no slot is ever reused.
// LANES2-COUNT-4: linalg.matmul
// LANES2-NOT: linalg.matmul

// Six forward flags, plus one.  Demand follows the phases and the buffer depth,
// never the unroll factor, so the six do not move.  The extra one is the merged
// group's back-edge release: at two lanes one union slot per lane is cheaper
// than the two rotating pools it replaces, so the merge happens even on a zero
// budget, and a merged slot's last reader is the consumer of the *second* role
// -- past the point the producing core last waits -- so the loop back edge needs
// a flag of its own.  It runs consumer-to-producer on the canonical
// vector-to-cube channel, which is why the reverse-pipe checks below still hold.
// FLAGS2-DAG: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
// FLAGS2-DAG: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// FLAGS2-DAG: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
// FLAGS2-DAG: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
// FLAGS2-DAG: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 4
// FLAGS2-DAG: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 5
// FLAGS2-DAG: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 6
// FLAGS2-DAG: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 6
// FLAGS2-NOT: flag = 7
// FLAGS2-NOT: sync_block{{.*}}<PIPE_V>, <PIPE_FIX>
// FLAGS2-NOT: sync_block{{.*}}<PIPE_MTE1>, <PIPE_MTE3>
