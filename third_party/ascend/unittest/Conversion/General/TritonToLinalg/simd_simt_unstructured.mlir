// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=true compile-mode=simd_simt' --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=true compile-mode=simd_simt' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-structured '--triton-to-linalg=compile-on-910-95=true compile-mode=simd_simt global-kernel=false' --split-input-file %s | FileCheck %s

tt.func public @simd_simt_gather(%src: !tt.ptr<f32>, %indices: !tt.ptr<i64>, %out: !tt.ptr<f32>) {
  %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>

  %indices_ptrs = tt.splat %indices : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
  %indices_addrs = tt.addptr %indices_ptrs, %rows : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
  %loaded_indices = tt.load %indices_addrs : tensor<8x!tt.ptr<i64>>

  %indices_2d = tt.expand_dims %loaded_indices {axis = 1 : i32} : tensor<8xi64> -> tensor<8x1xi64>
  %row_stride = arith.constant dense<32> : tensor<8x1xi64>
  %row_offsets = arith.muli %indices_2d, %row_stride : tensor<8x1xi64>
  %row_offsets_2d = tt.broadcast %row_offsets : tensor<8x1xi64> -> tensor<8x32xi64>
  %cols_2d_i32 = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %cols_2d_i64 = arith.extsi %cols_2d_i32 : tensor<1x32xi32> to tensor<1x32xi64>
  %col_offsets = tt.broadcast %cols_2d_i64 : tensor<1x32xi64> -> tensor<8x32xi64>
  %src_offsets = arith.addi %row_offsets_2d, %col_offsets : tensor<8x32xi64>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>>
  %src_ptrs = tt.addptr %src_base, %src_offsets : tensor<8x32x!tt.ptr<f32>>, tensor<8x32xi64>
  %values = tt.load %src_ptrs : tensor<8x32x!tt.ptr<f32>>

  %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %out_stride = arith.constant dense<32> : tensor<8x1xi32>
  %out_rows = arith.muli %rows_2d, %out_stride : tensor<8x1xi32>
  %out_rows_2d = tt.broadcast %out_rows : tensor<8x1xi32> -> tensor<8x32xi32>
  %out_cols_2d = tt.broadcast %cols_2d_i32 : tensor<1x32xi32> -> tensor<8x32xi32>
  %out_offsets = arith.addi %out_rows_2d, %out_cols_2d : tensor<8x32xi32>
  %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>>
  %out_ptrs = tt.addptr %out_base, %out_offsets : tensor<8x32x!tt.ptr<f32>>, tensor<8x32xi32>
  tt.store %out_ptrs, %values : tensor<8x32x!tt.ptr<f32>>
  tt.return
}

// The row axis (8 gathered indices) is genuinely unstructured, but the
// column axis is a 32xf32 = 128-byte contiguous run (a "long axis", >= 64
// bytes). It must NOT be routed through the ascend.unstructured_load ->
// hfusion.gather_load indirect path: that path is burst-aware but still
// loses to a plain structured copy per row on this target. It should fall
// back to the legacy per-row loop, which the backend can still execute as a
// parallel structured copy (hivm.parallel_loop), not a real scalar/SIMT loop.
// CHECK-LABEL: func.func @simd_simt_gather
// CHECK-SAME: parallel_mode = "simd"
// CHECK-NOT: hfusion.gather_load
// CHECK: scf.for
// CHECK: memref.reinterpret_cast {{.*}} sizes: [1, 32]
// CHECK: memref.copy
// CHECK: {{.*}}ExtractedLoadOrStore{{.*}}hivm.parallel_loop

// -----

// Same row-gather shape as above, but the column axis is only 8xf32 = 32
// bytes (< 64), a genuinely short axis. This must still take the SimdSimt
// indirect fast path -> hfusion.gather_load, to pin the boundary of the
// long-axis exclusion and guard against the threshold regressing either way.
tt.func public @simd_simt_gather_short_axis(%src: !tt.ptr<f32>, %indices: !tt.ptr<i64>, %out: !tt.ptr<f32>) {
  %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>

  %indices_ptrs = tt.splat %indices : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
  %indices_addrs = tt.addptr %indices_ptrs, %rows : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
  %loaded_indices = tt.load %indices_addrs : tensor<8x!tt.ptr<i64>>

  %indices_2d = tt.expand_dims %loaded_indices {axis = 1 : i32} : tensor<8xi64> -> tensor<8x1xi64>
  %row_stride = arith.constant dense<8> : tensor<8x1xi64>
  %row_offsets = arith.muli %indices_2d, %row_stride : tensor<8x1xi64>
  %row_offsets_2d = tt.broadcast %row_offsets : tensor<8x1xi64> -> tensor<8x8xi64>
  %cols_2d_i32 = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %cols_2d_i64 = arith.extsi %cols_2d_i32 : tensor<1x8xi32> to tensor<1x8xi64>
  %col_offsets = tt.broadcast %cols_2d_i64 : tensor<1x8xi64> -> tensor<8x8xi64>
  %src_offsets = arith.addi %row_offsets_2d, %col_offsets : tensor<8x8xi64>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
  %src_ptrs = tt.addptr %src_base, %src_offsets : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi64>
  %values = tt.load %src_ptrs : tensor<8x8x!tt.ptr<f32>>

  %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %out_stride = arith.constant dense<8> : tensor<8x1xi32>
  %out_rows = arith.muli %rows_2d, %out_stride : tensor<8x1xi32>
  %out_rows_2d = tt.broadcast %out_rows : tensor<8x1xi32> -> tensor<8x8xi32>
  %out_cols_2d = tt.broadcast %cols_2d_i32 : tensor<1x8xi32> -> tensor<8x8xi32>
  %out_offsets = arith.addi %out_rows_2d, %out_cols_2d : tensor<8x8xi32>
  %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
  %out_ptrs = tt.addptr %out_base, %out_offsets : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
  tt.store %out_ptrs, %values : tensor<8x8x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: func.func @simd_simt_gather_short_axis
// CHECK-SAME: parallel_mode = "mix_simd_simt"
// CHECK: %[[BURST:.*]] = arith.constant 8 : i32
// CHECK: hfusion.gather_load ins({{.*}}, %[[BURST]] : i32)
