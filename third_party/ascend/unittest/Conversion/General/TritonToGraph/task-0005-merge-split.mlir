// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=0' -o - | FileCheck %s
//
// Semantic TTIR fixture for the frozen before merge baseline. It avoids SSA
// names and only freezes the two-axis program mapping plus split reduction
// range; a future head-tensorization rule must update this in its own test.

// CHECK-LABEL: tt.func @task0005_merge_split_baseline(
// CHECK: tt.get_program_id x
// CHECK: tt.get_program_id y
// CHECK: tt.make_range {end = 4 : i32, start = 0 : i32}
// CHECK: tt.make_range {end = 256 : i32, start = 0 : i32}
// CHECK: tt.return
tt.func @task0005_merge_split_baseline() {
  %token = tt.get_program_id x : i32
  %head = tt.get_program_id y : i32
  %splits = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %dims = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  tt.return
}
