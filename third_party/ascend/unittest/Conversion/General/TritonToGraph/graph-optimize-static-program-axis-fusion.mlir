// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=1024 ub-capacity-bytes=2097152 device-core-count=1' -o - | FileCheck %s

// CHECK-LABEL: module attributes
// CHECK: hacc.program_grid_transforms = {{.*}}axis = 2 : i32{{.*}}factor = 4 : i64{{.*}}persistent_coverage = false
// CHECK-LABEL: tt.func public @spaf_g4(
// CHECK-NOT: tt.get_program_id z
// CHECK: tt.load {{.*}} : tensor<16x16x!tt.ptr<bf16>>
// CHECK: scf.for %[[GROUP:.*]] = {{.*}} to {{.*}} step {{.*}} : i32 {
// CHECK: arith.muli %[[GROUP]], {{.*}} : i32
// CHECK: tt.dot {{.*}} -> tensor<16x16xf32>
// CHECK: tt.store {{.*}} : tensor<16x16x!tt.ptr<f32>>
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
