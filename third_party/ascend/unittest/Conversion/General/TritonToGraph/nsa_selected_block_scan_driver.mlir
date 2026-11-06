// === #2686 SelectedBlockScanRule (bit 65536) — driver accept/reject ===
//
// STATUS: landed 2026-10-31 — all four RUN lines are green.  The REJECT line
// must STAY green: an out-of-range bit must keep being rejected so nobody
// "fixes" the whitelist by widening it.
//
// Deployed as a standalone file (the same role the 18-line
// graph-optimize-driver.mlir plays for the legacy bits: option plumbing, no
// rewrite assertions).
//
// Bit arithmetic: 65536 = 1 << 16 = SelectedBlockScan.  1048576 = 1 << 20 is
// out of range on purpose (bit index 20 is past the registry).
// 68607 = 3071 | 65536 = the legacy rule set plus the new bit.  The default stays
// 3071 (bits 16/17 are fixed default-off, RFC #2686): _graph_optimize_kwargs
// forwards rule_mask only when it differs from
// DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK (third_party/ascend/backend/compiler.py,
// _graph_optimize_kwargs).

// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=65536' -o - | FileCheck %s
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=68607' -o - | FileCheck %s --check-prefix=COMBINED
// RUN: not triton-opt %s -graph-optimize='rule-mask=1048576' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT

// The new bit must also be accepted on the real kernel IR (exit 0).  The scan
// rewrite has no separate row-lift gate -- it always runs the row lift's real
// implementation (the
// QueryRowBatching.h materialize entry) -- so bit 65536 alone must produce
// exactly the same IR as the joint 65536|131072 run below.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536' -o /dev/null
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536' -o %t.rulea
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608' -o %t.joint
// RUN: diff %t.rulea %t.joint

// Joint scan + row lift path: 65536 | 131072 = 196608.  The scan rewrite replaces the
// sparse loop and materializes the row lift's contract inside its own sandbox, so
// the output must carry the dynamic grid contract and the lifted dot shapes.
// The row count is now derived from the resource constraints alone: this run
// passes no L0 budget, so the per-value byte budget caps the 8 KiB f32
// intermediate at 128 KiB / 8 KiB = 16 rows (16 x 16 = 256 dot rows).
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608' -o - | FileCheck %s --check-prefix=JOINT

// CHECK-LABEL: tt.func @graph_optimize_selected_block_scan_accept()
tt.func @graph_optimize_selected_block_scan_accept() {
  // CHECK-NEXT: tt.return
  tt.return
  // CHECK-NEXT: }
}

// -----

// COMBINED-LABEL: tt.func @graph_optimize_selected_block_scan_with_defaults(
// COMBINED: tt.return %{{.*}} : i32
tt.func @graph_optimize_selected_block_scan_with_defaults(%value: i32) -> i32 {
  tt.return %value : i32
}

// -----

// The reject case needs a module to feed the option parser; it must reject
// before doing anything to it.
// REJECT: error: graph-optimize rule-mask contains unknown or out-of-range bits
// REJECT: 1048576
tt.func @graph_optimize_reject_unknown_bit() {
  tt.return
}

// -----

// Joint A+B contract checks (fed by the RUN line above from the real kernel).
// JOINT: hacc.program_grid_transforms
// JOINT-SAME: factor = 16
// JOINT: tensor<256x128xbf16>

// -----

// Static-grid joint commit: with hacc.grid_num_tiles on the module, the row lift takes
// its static path, so apply() must copy the three coalesce attributes and must
// NOT publish program_grid_transforms (that combination would divide the grid
// while the kernel still assumes the original one).
// RUN: triton-opt %S/Inputs/nsa_fwd_static_grid.mlir -graph-optimize='rule-mask=196608' -o - | FileCheck %s --check-prefix=STATICGRID
// the three attributes live on the module line, in name order
// STATICGRID: hacc.coalesce_axis = 0 : i32
// STATICGRID: hacc.coalesce_factor = 16 : i32
// STATICGRID: hacc.coalesce_grid_ceil_div = 1 : i32
// STATICGRID-NOT: hacc.program_grid_transforms
