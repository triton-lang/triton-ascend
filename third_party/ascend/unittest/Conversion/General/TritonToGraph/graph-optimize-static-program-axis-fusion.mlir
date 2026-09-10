// RUN: triton-opt --split-input-file %s --verify-each '-graph-optimize=rule-mask=1024 ub-capacity-bytes=2097152 device-core-count=1' -o - 2>&1 | FileCheck %s --check-prefix=FULL
// RUN: triton-opt --split-input-file %s --verify-each '-graph-optimize=rule-mask=1024 ub-capacity-bytes=2097152 device-core-count=2' -o - 2>&1 | FileCheck %s --check-prefix=PARTIAL
// RUN: triton-opt --split-input-file %s --verify-each '-graph-optimize=rule-mask=1024 ub-capacity-bytes=2097152 device-core-count=1 compile-mode=simt_only' -o - 2>&1 | FileCheck %s --check-prefix=SIMT

// The only positive fixture is deliberately logits-shaped.  K is loaded once
// before the first group use; Q, weight-like work, the f32 dot accumulator and
// store all live in the group suffix.  The group-major output rows and their
// zero-based mask are the proof that different loop iterations do not overlap.
// FULL-LABEL: module attributes
// FULL: hacc.program_grid_transforms = {{.*}}axis = 2 : i32{{.*}}factor = 4 : i64{{.*}}persistent_coverage = false{{.*}}
// FULL-LABEL: tt.func public @spaf_g4(
// FULL-NOT: tt.get_program_id z
// FULL: tt.load {{.*}} : tensor<16x16x!tt.ptr<bf16>>
// FULL: scf.for %[[FULL_GROUP:.*]] = {{.*}} to {{.*}} step {{.*}} : i32 {
// FULL: arith.muli %[[FULL_GROUP]], {{.*}} : i32
// FULL: tt.load {{.*}} : tensor<16x16x!tt.ptr<bf16>>
// FULL: tt.dot {{.*}} -> tensor<16x16xf32>
// FULL: tt.store {{.*}} : tensor<16x16x!tt.ptr<f32>>

// PARTIAL-LABEL: module attributes
// PARTIAL: hacc.program_grid_transforms = {{.*}}axis = 2 : i32{{.*}}factor = 2 : i64{{.*}}
// PARTIAL-LABEL: tt.func public @spaf_g4(
// PARTIAL: %[[PARTIAL_PID:.*]] = tt.get_program_id z
// PARTIAL: %[[PARTIAL_BASE:.*]] = arith.muli %[[PARTIAL_PID]], {{.*}} : i32
// PARTIAL: scf.for %[[PARTIAL_IV:.*]] = {{.*}} to {{.*}} step {{.*}} : i32 {
// PARTIAL: %[[PARTIAL_GROUP:.*]] = arith.addi %[[PARTIAL_BASE]], %[[PARTIAL_IV]] : i32
// PARTIAL: arith.muli %[[PARTIAL_GROUP]], {{.*}} : i32

// SIMT-LABEL: tt.func public @spaf_g4(
// SIMT: tt.get_program_id z
// SIMT-NOT: scf.for
// SIMT-NOT: hacc.program_grid_transforms

module attributes {
  hacc.grid_specialization = {version = 1 : i64, grid_0 = 1 : i64,
                              grid_1 = 1 : i64, grid_2 = 4 : i64,
                              rule_mask = 1024 : i64}
} {
  tt.func public @spaf_g4(%q: !tt.ptr<bf16>, %k: !tt.ptr<bf16>,
                          %out: !tt.ptr<f32>, %span: i32,
                          %out_stride: i32) attributes {
      hacc.grid_specialization = {version = 1 : i64, grid_0 = 1 : i64,
                                  grid_1 = 1 : i64, grid_2 = 4 : i64,
                                  rule_mask = 1024 : i64}
    } {
    %zero_bf16 = arith.constant dense<0.000000e+00> : tensor<16x16xbf16>
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %group = tt.get_program_id z : i32
    %queries = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %keys = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %query_span = tt.splat %span : i32 -> tensor<16xi32>
    %query_ok = arith.cmpi slt, %queries, %query_span : tensor<16xi32>
    %key_span = tt.splat %span : i32 -> tensor<16xi32>
    %key_ok = arith.cmpi slt, %keys, %key_span : tensor<16xi32>
    %query_e = tt.expand_dims %query_ok {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
    %key_e = tt.expand_dims %key_ok {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %query_mask = tt.broadcast %query_e : tensor<16x1xi1> -> tensor<16x16xi1>
    %key_mask = tt.broadcast %key_e : tensor<1x16xi1> -> tensor<16x16xi1>
    %mask = arith.andi %query_mask, %key_mask : tensor<16x16xi1>
    %key_base = tt.splat %k : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
    %key_tile = tt.load %key_base, %mask, %zero_bf16 : tensor<16x16x!tt.ptr<bf16>>
    %q_group = arith.muli %group, %span : i32
    %q_group_splat = tt.splat %q_group : i32 -> tensor<16x16xi32>
    %q_base = tt.splat %q : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
    %q_ptr = tt.addptr %q_base, %q_group_splat : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %q_tile = tt.load %q_ptr, %mask, %zero_bf16 : tensor<16x16x!tt.ptr<bf16>>
    %acc = tt.dot %q_tile, %key_tile, %zero_f32 : tensor<16x16xbf16> * tensor<16x16xbf16> -> tensor<16x16xf32>
    %rows = arith.muli %group, %span : i32
    %rows_splat = tt.splat %rows : i32 -> tensor<16xi32>
    %rows_with_query = arith.addi %rows_splat, %queries : tensor<16xi32>
    %rows_e = tt.expand_dims %rows_with_query {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %stride_e = tt.splat %out_stride : i32 -> tensor<16x1xi32>
    %row_offsets = arith.muli %rows_e, %stride_e : tensor<16x1xi32>
    %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %out_rows = tt.addptr %out_base, %row_offsets : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %out_rows_b = tt.broadcast %out_rows : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %keys_e = tt.expand_dims %keys {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %keys_b = tt.broadcast %keys_e : tensor<1x16xi32> -> tensor<16x16xi32>
    %out_ptr = tt.addptr %out_rows_b, %keys_b : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %out_ptr, %acc, %mask : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// G=1 is a required no-op even if the body superficially resembles the
// logits form: no loop and no launcher transform may be introduced.
// FULL-LABEL: tt.func public @spaf_g1_noop(
// FULL: tt.get_program_id z
// FULL-NOT: scf.for
// PARTIAL-LABEL: tt.func public @spaf_g1_noop(
// PARTIAL: tt.get_program_id z
// PARTIAL-NOT: scf.for
module attributes {
  hacc.grid_specialization = {version = 1 : i64, grid_0 = 1 : i64,
                              grid_1 = 1 : i64, grid_2 = 1 : i64,
                              rule_mask = 1024 : i64}
} {
  tt.func public @spaf_g1_noop() attributes {
      hacc.grid_specialization = {version = 1 : i64, grid_0 = 1 : i64,
                                  grid_1 = 1 : i64, grid_2 = 1 : i64,
                                  rule_mask = 1024 : i64}
    } {
    %group = tt.get_program_id z : i32
    tt.return
  }
}
