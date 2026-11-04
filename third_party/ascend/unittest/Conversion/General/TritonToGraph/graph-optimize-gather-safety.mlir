// RUN: triton-opt %s --split-input-file --verify-each -graph-optimize='target-arch=Ascend910B1 rule-mask=65536 ub-capacity-bytes=157286 compile-mode=simd' | FileCheck %s
// RUN: triton-opt %s --split-input-file --verify-each -graph-optimize='target-arch=Ascend910B1 rule-mask=65536 ub-capacity-bytes=704 compile-mode=simd' | FileCheck %s
// RUN: triton-opt %s --split-input-file --verify-each -graph-optimize='target-arch=Ascend910B1 rule-mask=65536 ub-capacity-bytes=703 compile-mode=simd' | FileCheck %s --check-prefix=SMALL --implicit-check-not=tt.gather

// The 2x16 source and 2x8 indices estimate to 704 bytes. The supplied
// budget is used directly: 704 accepts the candidate, while 703 rejects it.
// SMALL-LABEL: @unmasked_base_bias


// -----

// CHECK-LABEL: @unmasked_base_bias
// CHECK: %[[BASE:.*]] = arith.muli
// CHECK: scf.if
// CHECK: tt.broadcast %[[BASE]] : tensor<2x1xi32> -> tensor<2x16xi32>
// CHECK: tt.gather
// CHECK: else
// CHECK: gather.optimised.load = "fallback"
// CHECK: tt.return

tt.func @unmasked_base_bias(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %bias: i32) -> tensor<2x8xf32> {
  %rows = tt.make_range {start = 3 : i32, end = 5 : i32} : tensor<2xi32>
  %bias_tensor = tt.splat %bias : i32 -> tensor<2xi32>
  %biased = arith.addi %rows, %bias_tensor : tensor<2xi32>
  %expanded = tt.expand_dims %biased {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %width = arith.constant dense<16> : tensor<2x1xi32>
  %base = arith.muli %expanded, %width : tensor<2x1xi32>
  %base_grid = tt.broadcast %base : tensor<2x1xi32> -> tensor<2x8xi32>
  %idx_ptrs = tt.splat %idx : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
  %indices = tt.load %idx_ptrs : tensor<2x8x!tt.ptr<i32>>
  %offsets = arith.addi %base_grid, %indices : tensor<2x8xi32>
  %src_ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptrs = tt.addptr %src_ptrs, %offsets : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  %value = tt.load %ptrs : tensor<2x8x!tt.ptr<f32>>
  tt.return %value : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: @row_mask_other
// CHECK: %[[BASE:.*]] = arith.muli
// CHECK: scf.if
// CHECK: tt.broadcast %[[BASE]] : tensor<2x1xi32> -> tensor<2x16xi32>
// CHECK: %[[MASK:.*]] = tt.broadcast {{.*}} : tensor<2x1xi1> -> tensor<2x16xi1>
// CHECK: tt.load {{.*}}, %[[MASK]], {{.*}} {gather.optimised.load = "source"}
// CHECK: tt.gather
// CHECK: arith.select
// CHECK: scf.yield
// CHECK: else
// CHECK: gather.optimised.load = "fallback"
// CHECK: tt.return

tt.func @row_mask_other(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %bias: i32, %mask: tensor<2x1xi1>) -> tensor<2x8xf32> {
  %rows = tt.make_range {start = 3 : i32, end = 5 : i32} : tensor<2xi32>
  %bias_tensor = tt.splat %bias : i32 -> tensor<2xi32>
  %biased = arith.addi %rows, %bias_tensor : tensor<2xi32>
  %expanded = tt.expand_dims %biased {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %width = arith.constant dense<16> : tensor<2x1xi32>
  %base = arith.muli %expanded, %width : tensor<2x1xi32>
  %base_grid = tt.broadcast %base : tensor<2x1xi32> -> tensor<2x8xi32>
  %idx_ptrs = tt.splat %idx : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
  %indices = tt.load %idx_ptrs : tensor<2x8x!tt.ptr<i32>>
  %offsets = arith.addi %base_grid, %indices : tensor<2x8xi32>
  %src_ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptrs = tt.addptr %src_ptrs, %offsets : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  %mask_grid = tt.broadcast %mask : tensor<2x1xi1> -> tensor<2x8xi1>
  %other = arith.constant dense<-7.0> : tensor<2x8xf32>
  %value = tt.load %ptrs, %mask_grid, %other : tensor<2x8x!tt.ptr<f32>>
  tt.return %value : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: @lane_mask_rejected
// CHECK-NOT: tt.gather
// CHECK: tt.return

tt.func @lane_mask_rejected(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %bias: i32, %mask: tensor<2x8xi1>) -> tensor<2x8xf32> {
  %rows = tt.make_range {start = 3 : i32, end = 5 : i32} : tensor<2xi32>
  %bias_tensor = tt.splat %bias : i32 -> tensor<2xi32>
  %biased = arith.addi %rows, %bias_tensor : tensor<2xi32>
  %expanded = tt.expand_dims %biased {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %width = arith.constant dense<16> : tensor<2x1xi32>
  %base = arith.muli %expanded, %width : tensor<2x1xi32>
  %base_grid = tt.broadcast %base : tensor<2x1xi32> -> tensor<2x8xi32>
  %idx_ptrs = tt.splat %idx : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
  %indices = tt.load %idx_ptrs : tensor<2x8x!tt.ptr<i32>>
  %offsets = arith.addi %base_grid, %indices : tensor<2x8xi32>
  %src_ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptrs = tt.addptr %src_ptrs, %offsets : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  %mask_grid = arith.andi %mask, %mask : tensor<2x8xi1>
  %other = arith.constant dense<-7.0> : tensor<2x8xf32>
  %value = tt.load %ptrs, %mask_grid, %other : tensor<2x8x!tt.ptr<f32>>
  tt.return %value : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: @volatile_rejected
// CHECK-NOT: tt.gather
// CHECK: tt.return

tt.func @volatile_rejected(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %bias: i32) -> tensor<2x8xf32> {
  %rows = tt.make_range {start = 3 : i32, end = 5 : i32} : tensor<2xi32>
  %bias_tensor = tt.splat %bias : i32 -> tensor<2xi32>
  %biased = arith.addi %rows, %bias_tensor : tensor<2xi32>
  %expanded = tt.expand_dims %biased {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %width = arith.constant dense<16> : tensor<2x1xi32>
  %base = arith.muli %expanded, %width : tensor<2x1xi32>
  %base_grid = tt.broadcast %base : tensor<2x1xi32> -> tensor<2x8xi32>
  %idx_ptrs = tt.splat %idx : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
  %indices = tt.load %idx_ptrs : tensor<2x8x!tt.ptr<i32>>
  %offsets = arith.addi %base_grid, %indices : tensor<2x8xi32>
  %src_ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptrs = tt.addptr %src_ptrs, %offsets : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  %value = tt.load %ptrs {isVolatile = true} : tensor<2x8x!tt.ptr<f32>>
  tt.return %value : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: @index_i64_rejected
// CHECK-NOT: tt.gather
// CHECK: tt.return

tt.func @index_i64_rejected(%src: !tt.ptr<f32>, %idx: !tt.ptr<i64>, %bias: i32) -> tensor<2x8xf32> {
  %rows = tt.make_range {start = 3 : i32, end = 5 : i32} : tensor<2xi32>
  %bias_tensor = tt.splat %bias : i32 -> tensor<2xi32>
  %biased = arith.addi %rows, %bias_tensor : tensor<2xi32>
  %wide_rows = arith.extsi %biased : tensor<2xi32> to tensor<2xi64>
  %expanded = tt.expand_dims %wide_rows {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
  %width = arith.constant dense<16> : tensor<2x1xi64>
  %base = arith.muli %expanded, %width : tensor<2x1xi64>
  %base_grid = tt.broadcast %base : tensor<2x1xi64> -> tensor<2x8xi64>
  %idx_ptrs = tt.splat %idx : !tt.ptr<i64> -> tensor<2x8x!tt.ptr<i64>>
  %indices = tt.load %idx_ptrs : tensor<2x8x!tt.ptr<i64>>
  %offsets = arith.addi %base_grid, %indices : tensor<2x8xi64>
  %src_ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptrs = tt.addptr %src_ptrs, %offsets : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi64>
  %value = tt.load %ptrs : tensor<2x8x!tt.ptr<f32>>
  tt.return %value : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: @negative_extent_rejected
// CHECK-NOT: tt.gather
// CHECK: tt.return

tt.func @negative_extent_rejected(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %bias: i32) -> tensor<2x8xf32> {
  %rows = tt.make_range {start = 3 : i32, end = 5 : i32} : tensor<2xi32>
  %bias_tensor = tt.splat %bias : i32 -> tensor<2xi32>
  %biased = arith.addi %rows, %bias_tensor : tensor<2xi32>
  %expanded = tt.expand_dims %biased {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %width = arith.constant dense<-16> : tensor<2x1xi32>
  %base = arith.muli %expanded, %width : tensor<2x1xi32>
  %base_grid = tt.broadcast %base : tensor<2x1xi32> -> tensor<2x8xi32>
  %idx_ptrs = tt.splat %idx : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
  %indices = tt.load %idx_ptrs : tensor<2x8x!tt.ptr<i32>>
  %offsets = arith.addi %base_grid, %indices : tensor<2x8xi32>
  %src_ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptrs = tt.addptr %src_ptrs, %offsets : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  %value = tt.load %ptrs : tensor<2x8x!tt.ptr<f32>>
  tt.return %value : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: @deep_offset_cache
// CHECK: %[[BASE:.*]] = arith.muli
// CHECK: scf.if
// CHECK: tt.broadcast %[[BASE]] : tensor<2x1xi32> -> tensor<2x16xi32>
// CHECK: tt.gather
// CHECK: else
// CHECK: gather.optimised.load = "fallback"
// CHECK: tt.return

// More than 64 classified Values force recursive DenseMap growth.
// Run this case with an ASan-enabled triton-opt to catch dangling buckets.

tt.func @deep_offset_cache(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %bias: i32) -> tensor<2x8xf32> {
  %rows = tt.make_range {start = 3 : i32, end = 5 : i32} : tensor<2xi32>
  %bias_tensor = tt.splat %bias : i32 -> tensor<2xi32>
  %biased = arith.addi %rows, %bias_tensor : tensor<2xi32>
  %chain0 = arith.addi %biased, %bias_tensor : tensor<2xi32>
  %chain1 = arith.addi %chain0, %bias_tensor : tensor<2xi32>
  %chain2 = arith.addi %chain1, %bias_tensor : tensor<2xi32>
  %chain3 = arith.addi %chain2, %bias_tensor : tensor<2xi32>
  %chain4 = arith.addi %chain3, %bias_tensor : tensor<2xi32>
  %chain5 = arith.addi %chain4, %bias_tensor : tensor<2xi32>
  %chain6 = arith.addi %chain5, %bias_tensor : tensor<2xi32>
  %chain7 = arith.addi %chain6, %bias_tensor : tensor<2xi32>
  %chain8 = arith.addi %chain7, %bias_tensor : tensor<2xi32>
  %chain9 = arith.addi %chain8, %bias_tensor : tensor<2xi32>
  %chain10 = arith.addi %chain9, %bias_tensor : tensor<2xi32>
  %chain11 = arith.addi %chain10, %bias_tensor : tensor<2xi32>
  %chain12 = arith.addi %chain11, %bias_tensor : tensor<2xi32>
  %chain13 = arith.addi %chain12, %bias_tensor : tensor<2xi32>
  %chain14 = arith.addi %chain13, %bias_tensor : tensor<2xi32>
  %chain15 = arith.addi %chain14, %bias_tensor : tensor<2xi32>
  %chain16 = arith.addi %chain15, %bias_tensor : tensor<2xi32>
  %chain17 = arith.addi %chain16, %bias_tensor : tensor<2xi32>
  %chain18 = arith.addi %chain17, %bias_tensor : tensor<2xi32>
  %chain19 = arith.addi %chain18, %bias_tensor : tensor<2xi32>
  %chain20 = arith.addi %chain19, %bias_tensor : tensor<2xi32>
  %chain21 = arith.addi %chain20, %bias_tensor : tensor<2xi32>
  %chain22 = arith.addi %chain21, %bias_tensor : tensor<2xi32>
  %chain23 = arith.addi %chain22, %bias_tensor : tensor<2xi32>
  %chain24 = arith.addi %chain23, %bias_tensor : tensor<2xi32>
  %chain25 = arith.addi %chain24, %bias_tensor : tensor<2xi32>
  %chain26 = arith.addi %chain25, %bias_tensor : tensor<2xi32>
  %chain27 = arith.addi %chain26, %bias_tensor : tensor<2xi32>
  %chain28 = arith.addi %chain27, %bias_tensor : tensor<2xi32>
  %chain29 = arith.addi %chain28, %bias_tensor : tensor<2xi32>
  %chain30 = arith.addi %chain29, %bias_tensor : tensor<2xi32>
  %chain31 = arith.addi %chain30, %bias_tensor : tensor<2xi32>
  %chain32 = arith.addi %chain31, %bias_tensor : tensor<2xi32>
  %chain33 = arith.addi %chain32, %bias_tensor : tensor<2xi32>
  %chain34 = arith.addi %chain33, %bias_tensor : tensor<2xi32>
  %chain35 = arith.addi %chain34, %bias_tensor : tensor<2xi32>
  %chain36 = arith.addi %chain35, %bias_tensor : tensor<2xi32>
  %chain37 = arith.addi %chain36, %bias_tensor : tensor<2xi32>
  %chain38 = arith.addi %chain37, %bias_tensor : tensor<2xi32>
  %chain39 = arith.addi %chain38, %bias_tensor : tensor<2xi32>
  %chain40 = arith.addi %chain39, %bias_tensor : tensor<2xi32>
  %chain41 = arith.addi %chain40, %bias_tensor : tensor<2xi32>
  %chain42 = arith.addi %chain41, %bias_tensor : tensor<2xi32>
  %chain43 = arith.addi %chain42, %bias_tensor : tensor<2xi32>
  %chain44 = arith.addi %chain43, %bias_tensor : tensor<2xi32>
  %chain45 = arith.addi %chain44, %bias_tensor : tensor<2xi32>
  %chain46 = arith.addi %chain45, %bias_tensor : tensor<2xi32>
  %chain47 = arith.addi %chain46, %bias_tensor : tensor<2xi32>
  %chain48 = arith.addi %chain47, %bias_tensor : tensor<2xi32>
  %chain49 = arith.addi %chain48, %bias_tensor : tensor<2xi32>
  %chain50 = arith.addi %chain49, %bias_tensor : tensor<2xi32>
  %chain51 = arith.addi %chain50, %bias_tensor : tensor<2xi32>
  %chain52 = arith.addi %chain51, %bias_tensor : tensor<2xi32>
  %chain53 = arith.addi %chain52, %bias_tensor : tensor<2xi32>
  %chain54 = arith.addi %chain53, %bias_tensor : tensor<2xi32>
  %chain55 = arith.addi %chain54, %bias_tensor : tensor<2xi32>
  %chain56 = arith.addi %chain55, %bias_tensor : tensor<2xi32>
  %chain57 = arith.addi %chain56, %bias_tensor : tensor<2xi32>
  %chain58 = arith.addi %chain57, %bias_tensor : tensor<2xi32>
  %chain59 = arith.addi %chain58, %bias_tensor : tensor<2xi32>
  %chain60 = arith.addi %chain59, %bias_tensor : tensor<2xi32>
  %chain61 = arith.addi %chain60, %bias_tensor : tensor<2xi32>
  %chain62 = arith.addi %chain61, %bias_tensor : tensor<2xi32>
  %chain63 = arith.addi %chain62, %bias_tensor : tensor<2xi32>
  %chain64 = arith.addi %chain63, %bias_tensor : tensor<2xi32>
  %chain65 = arith.addi %chain64, %bias_tensor : tensor<2xi32>
  %chain66 = arith.addi %chain65, %bias_tensor : tensor<2xi32>
  %chain67 = arith.addi %chain66, %bias_tensor : tensor<2xi32>
  %chain68 = arith.addi %chain67, %bias_tensor : tensor<2xi32>
  %chain69 = arith.addi %chain68, %bias_tensor : tensor<2xi32>
  %chain70 = arith.addi %chain69, %bias_tensor : tensor<2xi32>
  %chain71 = arith.addi %chain70, %bias_tensor : tensor<2xi32>
  %chain72 = arith.addi %chain71, %bias_tensor : tensor<2xi32>
  %chain73 = arith.addi %chain72, %bias_tensor : tensor<2xi32>
  %chain74 = arith.addi %chain73, %bias_tensor : tensor<2xi32>
  %chain75 = arith.addi %chain74, %bias_tensor : tensor<2xi32>
  %chain76 = arith.addi %chain75, %bias_tensor : tensor<2xi32>
  %chain77 = arith.addi %chain76, %bias_tensor : tensor<2xi32>
  %chain78 = arith.addi %chain77, %bias_tensor : tensor<2xi32>
  %chain79 = arith.addi %chain78, %bias_tensor : tensor<2xi32>
  %chain80 = arith.addi %chain79, %bias_tensor : tensor<2xi32>
  %chain81 = arith.addi %chain80, %bias_tensor : tensor<2xi32>
  %chain82 = arith.addi %chain81, %bias_tensor : tensor<2xi32>
  %chain83 = arith.addi %chain82, %bias_tensor : tensor<2xi32>
  %chain84 = arith.addi %chain83, %bias_tensor : tensor<2xi32>
  %chain85 = arith.addi %chain84, %bias_tensor : tensor<2xi32>
  %chain86 = arith.addi %chain85, %bias_tensor : tensor<2xi32>
  %chain87 = arith.addi %chain86, %bias_tensor : tensor<2xi32>
  %chain88 = arith.addi %chain87, %bias_tensor : tensor<2xi32>
  %chain89 = arith.addi %chain88, %bias_tensor : tensor<2xi32>
  %chain90 = arith.addi %chain89, %bias_tensor : tensor<2xi32>
  %chain91 = arith.addi %chain90, %bias_tensor : tensor<2xi32>
  %chain92 = arith.addi %chain91, %bias_tensor : tensor<2xi32>
  %chain93 = arith.addi %chain92, %bias_tensor : tensor<2xi32>
  %chain94 = arith.addi %chain93, %bias_tensor : tensor<2xi32>
  %chain95 = arith.addi %chain94, %bias_tensor : tensor<2xi32>
  %expanded = tt.expand_dims %chain95 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %width = arith.constant dense<16> : tensor<2x1xi32>
  %base = arith.muli %expanded, %width : tensor<2x1xi32>
  %base_grid = tt.broadcast %base : tensor<2x1xi32> -> tensor<2x8xi32>
  %idx_ptrs = tt.splat %idx : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
  %indices = tt.load %idx_ptrs : tensor<2x8x!tt.ptr<i32>>
  %offsets = arith.addi %base_grid, %indices : tensor<2x8xi32>
  %src_ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %ptrs = tt.addptr %src_ptrs, %offsets : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  %value = tt.load %ptrs : tensor<2x8x!tt.ptr<f32>>
  tt.return %value : tensor<2x8xf32>
}

// -----

// Opaque tensor arguments still carry their rank. Mixing one with a known
// splat must not assert inside combineInfo during candidate discovery.
// CHECK-LABEL: @opaque_tensor_offsets
// CHECK-NOT: tt.gather
// CHECK: tt.return
tt.func @opaque_tensor_offsets(%src: !tt.ptr<f32>, %offsets: tensor<2x8xi32>, %bias: i32) -> tensor<2x8xf32> {
  %bias_tensor = tt.splat %bias : i32 -> tensor<2x8xi32>
  %sum = arith.addi %offsets, %bias_tensor : tensor<2x8xi32>
  %ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
  %addresses = tt.addptr %ptrs, %sum : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
  %result = tt.load %addresses : tensor<2x8x!tt.ptr<f32>>
  tt.return %result : tensor<2x8xf32>
}

// -----

// A runtime row stride leaves the source width unknown, so the axis multiplier
// must not fall back to a unit dimension: sizing the source tile at width 1
// would also wrap negative indices against the wrong width.
// The 16 indices are deliberate. Folding the stride to a unit dimension leaves
// a width-1 f32 source, which the benefit gate admits only from 16 indices up,
// so a narrower tile would be declined for that reason instead and the case
// would pass whether or not the multiplier is rejected.
// CHECK-LABEL: @runtime_row_stride
// CHECK-NOT: tt.gather
// CHECK: tt.return
tt.func @runtime_row_stride(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %stride: i32) -> tensor<2x16xf32> {
  %rows = tt.make_range {start = 3 : i32, end = 5 : i32} : tensor<2xi32>
  %expanded = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %width = tt.splat %stride : i32 -> tensor<2x1xi32>
  %base = arith.muli %expanded, %width : tensor<2x1xi32>
  %base_grid = tt.broadcast %base : tensor<2x1xi32> -> tensor<2x16xi32>
  %idx_ptrs = tt.splat %idx : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
  %indices = tt.load %idx_ptrs : tensor<2x16x!tt.ptr<i32>>
  %offsets = arith.addi %base_grid, %indices : tensor<2x16xi32>
  %src_ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<2x16x!tt.ptr<f32>>
  %ptrs = tt.addptr %src_ptrs, %offsets : tensor<2x16x!tt.ptr<f32>>, tensor<2x16xi32>
  %value = tt.load %ptrs : tensor<2x16x!tt.ptr<f32>>
  tt.return %value : tensor<2x16xf32>
}

// -----

// The scalar-row path has no tt.expand_dims tying its constant to the axis it
// sizes, so the multiplicand has to carry that evidence instead: a provable
// per-row advance. A row offset loaded from memory proves nothing, so the
// constant beside it need not be the row stride at all.
// CHECK-LABEL: @scalar_row_loaded_carrier
// CHECK-NOT: tt.gather
// CHECK: tt.return
tt.func @scalar_row_loaded_carrier(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>,
                                   %off: !tt.ptr<i32>, %out: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %c16_i32 = arith.constant 16 : i32
  %cols = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %idx_base = tt.splat %idx : !tt.ptr<i32> -> tensor<1x8x!tt.ptr<i32>>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
  %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
  scf.for %i = %c0_i32 to %c64_i32 step %c1_i32 : i32 {
    %off_ptr = tt.addptr %off, %i : !tt.ptr<i32>, i32
    %row = tt.load %off_ptr : !tt.ptr<i32>
    %row_base = arith.muli %row, %c16_i32 : i32
    %idx_ptrs = tt.addptr %idx_base, %cols_2d : tensor<1x8x!tt.ptr<i32>>, tensor<1x8xi32>
    %indices = tt.load %idx_ptrs : tensor<1x8x!tt.ptr<i32>>
    %row_splat = tt.splat %row_base : i32 -> tensor<1x8xi32>
    %offsets = arith.addi %row_splat, %indices : tensor<1x8xi32>
    %ptrs = tt.addptr %src_base, %offsets : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
    %value = tt.load %ptrs : tensor<1x8x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %cols_2d : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
    tt.store %out_ptrs, %value : tensor<1x8x!tt.ptr<f32>>
  }
  tt.return
}

// -----

// Two row-driven multiplies in one source offset: the offset advances by
// 1024 + 16 per iteration, so neither constant is the row stride and a
// 1024-wide read would cross into the next row's data.
// CHECK-LABEL: @scalar_row_two_strides
// CHECK-NOT: tt.gather
// CHECK: tt.return
tt.func @scalar_row_two_strides(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>,
                                %out: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %cols = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %idx_base = tt.splat %idx : !tt.ptr<i32> -> tensor<1x8x!tt.ptr<i32>>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
  %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
  scf.for %i = %c0_i32 to %c64_i32 step %c1_i32 : i32 {
    %major = arith.muli %i, %c1024_i32 : i32
    %minor = arith.muli %i, %c16_i32 : i32
    %row_base = arith.addi %major, %minor : i32
    %idx_ptrs = tt.addptr %idx_base, %cols_2d : tensor<1x8x!tt.ptr<i32>>, tensor<1x8xi32>
    %indices = tt.load %idx_ptrs : tensor<1x8x!tt.ptr<i32>>
    %row_splat = tt.splat %row_base : i32 -> tensor<1x8xi32>
    %offsets = arith.addi %row_splat, %indices : tensor<1x8xi32>
    %ptrs = tt.addptr %src_base, %offsets : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
    %value = tt.load %ptrs : tensor<1x8x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %cols_2d : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
    tt.store %out_ptrs, %value : tensor<1x8x!tt.ptr<f32>>
  }
  tt.return
}

// -----

// The scalar row path measures the row advance against an enclosing scf.for.
// With no loop there is no per-row counter to measure, so the width stays
// unknown and the load is left alone.
// CHECK-LABEL: @scalar_row_without_loop
// CHECK-NOT: tt.gather
// CHECK: tt.return
tt.func @scalar_row_without_loop(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>,
                                 %row: i32) -> tensor<1x8xf32> {
  %c16_i32 = arith.constant 16 : i32
  %cols = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %row_base = arith.muli %row, %c16_i32 : i32
  %idx_base = tt.splat %idx : !tt.ptr<i32> -> tensor<1x8x!tt.ptr<i32>>
  %idx_ptrs = tt.addptr %idx_base, %cols_2d : tensor<1x8x!tt.ptr<i32>>, tensor<1x8xi32>
  %indices = tt.load %idx_ptrs : tensor<1x8x!tt.ptr<i32>>
  %row_splat = tt.splat %row_base : i32 -> tensor<1x8xi32>
  %offsets = arith.addi %row_splat, %indices : tensor<1x8xi32>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
  %ptrs = tt.addptr %src_base, %offsets : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
  %value = tt.load %ptrs : tensor<1x8x!tt.ptr<f32>>
  tt.return %value : tensor<1x8xf32>
}
