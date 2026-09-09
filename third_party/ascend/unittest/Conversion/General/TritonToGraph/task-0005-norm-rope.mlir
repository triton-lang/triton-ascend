// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=0' -o - | FileCheck %s
//
// Semantic TTIR fixture for the before norm+RoPE path. The fixture captures
// token/head mapping and the 256-wide head / 16-wide rotary half, not unstable
// lowering-specific value names.

// CHECK-LABEL: tt.func @task0005_norm_rope_baseline(
// CHECK: tt.get_program_id x
// CHECK: tt.get_program_id y
// CHECK: tt.make_range {end = 256 : i32, start = 0 : i32}
// CHECK: tt.make_range {end = 16 : i32, start = 0 : i32}
// CHECK: tt.return
tt.func @task0005_norm_rope_baseline() {
  %token = tt.get_program_id x : i32
  %head = tt.get_program_id y : i32
  %dims = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  %rotary = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  tt.return
}
