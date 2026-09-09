// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=0' -o - | FileCheck %s
//
// The before kernel has a three-dimensional Q-tile/K-tile/provider-group grid.
// A later static-axis-fusion test may remove z only after it proves K reuse.

// CHECK-LABEL: tt.func @task0005_indexer_logits_baseline(
// CHECK: tt.get_program_id x
// CHECK: tt.get_program_id y
// CHECK: tt.get_program_id z
// CHECK: tt.make_range {end = 64 : i32, start = 0 : i32}
// CHECK: tt.make_range {end = 128 : i32, start = 0 : i32}
// CHECK: tt.return
tt.func @task0005_indexer_logits_baseline() {
  %q_tile = tt.get_program_id x : i32
  %k_tile = tt.get_program_id y : i32
  %group = tt.get_program_id z : i32
  %queries = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %keys = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  tt.return
}
