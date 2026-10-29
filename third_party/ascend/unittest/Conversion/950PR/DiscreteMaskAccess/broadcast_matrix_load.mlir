// RUN: triton-opt %s --discrete-mask-access-conversion='compile-on-910-95=true compile-mode=simd_simt_template' | FileCheck %s --check-prefix=MASK
// RUN: triton-opt %s --discrete-mask-access-conversion='compile-on-910-95=true compile-mode=simd_simt_template' --triton-to-unstructure='compile-on-910-95=true compile-mode=simd_simt_template' --bubble-up-operation='compile-on-910-95=true' --triton-to-structured | FileCheck %s --check-prefix=LOWER

// A row predicate must guard the load itself, including when an invalid row's
// base is out of bounds. The false branch preserves the original 'other'.
// MASK-LABEL: tt.func @gather_rows(
// MASK-SAME: %[[OTHER:arg[0-9]+]]: tensor<4x64xi8>
// MASK-NOT: tt.load
// MASK: %[[BYTE_MASK:.*]] = arith.extui {{.*}} : tensor<4x1xi1> to tensor<4x1xi8>
// MASK: scf.for %[[I:[^ ]+]] = {{.*}} iter_args(%[[ACC:.*]] = %[[OTHER]])
// MASK-NOT: tt.load
// MASK: %[[PRED:.*]] = tensor.extract %[[BYTE_MASK]][%[[I]]{{.*}}] : tensor<4x1xi8>
// MASK-NEXT: %[[VALID:.*]] = arith.cmpi ne, %[[PRED]], {{.*}} : i8
// MASK-NEXT: %[[IF:.*]] = scf.if %[[VALID]] -> (tensor<64xi8>) {
// MASK: %[[PTR:.*]] = tensor.extract_slice {{.*}}[%[[I]], 0] [1, 64] [1, 1]
// MASK: %[[LOAD:.*]] = tt.load %[[PTR]] : tensor<64x!tt.ptr<i8>>
// MASK-NEXT: scf.yield %[[LOAD]] : tensor<64xi8>
// MASK-NEXT: } else {
// MASK-NEXT: %[[OTHER_ROW:.*]] = tensor.extract_slice %[[ACC]][%[[I]], 0] [1, 64] [1, 1]
// MASK-NEXT: scf.yield %[[OTHER_ROW]] : tensor<64xi8>
// MASK-NEXT: }
// MASK-NEXT: %[[INSERT:.*]] = tensor.insert_slice %[[IF]] into %[[ACC]][%[[I]], 0] [1, 64] [1, 1]
// MASK-NEXT: scf.yield %[[INSERT]] : tensor<4x64xi8>
// MASK-NOT: tt.load
// MASK: tt.return
// LOWER-LABEL: tt.func @gather_rows(
// LOWER-NOT: ascend.indirect_load
// LOWER: scf.for
// LOWER: scf.if
// LOWER: tt.load {{.*}} : tensor<64x!tt.ptr<i8>>
// LOWER: } else {
// LOWER-NOT: tt.load
// LOWER: tt.return
tt.func @gather_rows(%base: !tt.ptr<i8>, %offsets: tensor<4xi32>, %valid: tensor<4xi1>, %other: tensor<4x64xi8>) -> tensor<4x64xi8> {
  %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %rows = tt.expand_dims %offsets {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %a = tt.broadcast %rows : tensor<4x1xi32> -> tensor<4x64xi32>
  %b = tt.broadcast %cols : tensor<1x64xi32> -> tensor<4x64xi32>
  %off = arith.addi %a, %b : tensor<4x64xi32>
  %base_tensor = tt.splat %base : !tt.ptr<i8> -> tensor<4x64x!tt.ptr<i8>>
  %ptr = tt.addptr %base_tensor, %off : tensor<4x64x!tt.ptr<i8>>, tensor<4x64xi32>
  %m = tt.expand_dims %valid {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %mask = tt.broadcast %m : tensor<4x1xi1> -> tensor<4x64xi1>
  %v = tt.load %ptr, %mask, %other : tensor<4x64x!tt.ptr<i8>>
  tt.return %v : tensor<4x64xi8>
}

// The RoPE byte plane has stride two within each row. Preserve that stride and
// volatility, and allow omitted 'other' without speculating the load.
// MASK-LABEL: tt.func @strided_rows_without_other(
// MASK: %[[ZERO:.*]] = arith.constant dense<0> : tensor<4x64xi8>
// MASK: scf.for {{.*}} iter_args({{.*}} = %[[ZERO]])
// MASK: scf.if
// MASK: tt.load {{.*}} {isVolatile = true} : tensor<64x!tt.ptr<i8>>
// LOWER-LABEL: tt.func @strided_rows_without_other(
// LOWER-NOT: ascend.indirect_load
// LOWER: scf.if
// LOWER: tt.load {{.*}} {isVolatile = true} : tensor<64x!tt.ptr<i8>>
// LOWER: tt.return
tt.func @strided_rows_without_other(%base: !tt.ptr<i8>, %offsets: tensor<4xi32>, %valid: tensor<4xi1>) -> tensor<4x64xi8> {
  %two = arith.constant dense<2> : tensor<64xi32>
  %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %r2 = arith.muli %r, %two : tensor<64xi32>
  %rows = tt.expand_dims %offsets {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %cols = tt.expand_dims %r2 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %a = tt.broadcast %rows : tensor<4x1xi32> -> tensor<4x64xi32>
  %b = tt.broadcast %cols : tensor<1x64xi32> -> tensor<4x64xi32>
  %off = arith.addi %a, %b : tensor<4x64xi32>
  %bt = tt.splat %base : !tt.ptr<i8> -> tensor<4x64x!tt.ptr<i8>>
  %ptr = tt.addptr %bt, %off : tensor<4x64x!tt.ptr<i8>>, tensor<4x64xi32>
  %m = tt.expand_dims %valid {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %mask = tt.broadcast %m : tensor<4x1xi1> -> tensor<4x64xi1>
  %v = tt.load %ptr, %mask {isVolatile = true} : tensor<4x64x!tt.ptr<i8>>
  tt.return %v : tensor<4x64xi8>
}

// A column predicate can similarly guard a structured column.
// MASK-LABEL: tt.func @gather_columns(
// MASK: scf.for %[[J:[^ ]+]] =
// MASK: scf.if
// MASK: tensor.extract_slice {{.*}}[0, %[[J]]] [64, 1] [1, 1]
// MASK: tt.load {{.*}} : tensor<64x!tt.ptr<i8>>
// LOWER-LABEL: tt.func @gather_columns(
// LOWER-NOT: ascend.indirect_load
// LOWER: scf.if
// LOWER: tt.load {{.*}} : tensor<64x!tt.ptr<i8>>
// LOWER: tt.return
tt.func @gather_columns(%base: !tt.ptr<i8>, %offsets: tensor<4xi32>, %valid: tensor<4xi1>) -> tensor<64x4xi8> {
  %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %cols = tt.expand_dims %offsets {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %a = tt.broadcast %rows : tensor<64x1xi32> -> tensor<64x4xi32>
  %b = tt.broadcast %cols : tensor<1x4xi32> -> tensor<64x4xi32>
  %off = arith.addi %a, %b : tensor<64x4xi32>
  %bt = tt.splat %base : !tt.ptr<i8> -> tensor<64x4x!tt.ptr<i8>>
  %ptr = tt.addptr %bt, %off : tensor<64x4x!tt.ptr<i8>>, tensor<64x4xi32>
  %m = tt.expand_dims %valid {axis = 0 : i32} : tensor<4xi1> -> tensor<1x4xi1>
  %mask = tt.broadcast %m : tensor<1x4xi1> -> tensor<64x4xi1>
  %v = tt.load %ptr, %mask : tensor<64x4x!tt.ptr<i8>>
  tt.return %v : tensor<64x4xi8>
}

// A row mask alone is insufficient: columns with unknown offsets stay indirect.
// MASK-LABEL: tt.func @unstructured_columns(
// MASK-NOT: scf.if
// MASK: tt.load {{.*}} {MixCompileDiscreteMask}
// LOWER-LABEL: tt.func @unstructured_columns(
// LOWER: ascend.indirect_load
tt.func @unstructured_columns(%base: !tt.ptr<i8>, %offsets: tensor<4x64xi32>, %valid: tensor<4xi1>) -> tensor<4x64xi8> {
  %bt = tt.splat %base : !tt.ptr<i8> -> tensor<4x64x!tt.ptr<i8>>
  %ptr = tt.addptr %bt, %offsets : tensor<4x64x!tt.ptr<i8>>, tensor<4x64xi32>
  %m = tt.expand_dims %valid {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %mask = tt.broadcast %m : tensor<4x1xi1> -> tensor<4x64xi1>
  %v = tt.load %ptr, %mask : tensor<4x64x!tt.ptr<i8>>
  tt.return %v : tensor<4x64xi8>
}

// Small blocks retain the existing indirect policy.
// MASK-LABEL: tt.func @small_block(
// MASK-NOT: scf.if
// MASK: tt.load {{.*}} {MixCompileDiscreteMask}
// LOWER-LABEL: tt.func @small_block(
// LOWER: ascend.indirect_load
tt.func @small_block(%base: !tt.ptr<i8>, %offsets: tensor<4xi32>, %valid: tensor<4xi1>) -> tensor<4x32xi8> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %rows = tt.expand_dims %offsets {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %a = tt.broadcast %rows : tensor<4x1xi32> -> tensor<4x32xi32>
  %b = tt.broadcast %cols : tensor<1x32xi32> -> tensor<4x32xi32>
  %off = arith.addi %a, %b : tensor<4x32xi32>
  %bt = tt.splat %base : !tt.ptr<i8> -> tensor<4x32x!tt.ptr<i8>>
  %ptr = tt.addptr %bt, %off : tensor<4x32x!tt.ptr<i8>>, tensor<4x32xi32>
  %m = tt.expand_dims %valid {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %mask = tt.broadcast %m : tensor<4x1xi1> -> tensor<4x32xi1>
  %v = tt.load %ptr, %mask : tensor<4x32x!tt.ptr<i8>>
  tt.return %v : tensor<4x32xi8>
}

// A genuinely per-element matrix mask must remain masked indirect.
// MASK-LABEL: tt.func @element_mask(
// MASK-NOT: scf.if
// MASK: tt.load {{.*}} {MixCompileDiscreteMask}
// LOWER-LABEL: tt.func @element_mask(
// LOWER: ascend.indirect_load
tt.func @element_mask(%base: !tt.ptr<i8>, %offsets: tensor<4x64xi32>, %mask: tensor<4x64xi1>) -> tensor<4x64xi8> {
  %bt = tt.splat %base : !tt.ptr<i8> -> tensor<4x64x!tt.ptr<i8>>
  %ptr = tt.addptr %bt, %offsets : tensor<4x64x!tt.ptr<i8>>, tensor<4x64xi32>
  %v = tt.load %ptr, %mask : tensor<4x64x!tt.ptr<i8>>
  tt.return %v : tensor<4x64xi8>
}

// A row/column conjunction is not a uniform row mask. In particular, reading
// an entire row would ignore the column predicate's memory-safety boundary.
// MASK-LABEL: tt.func @row_and_column_mask(
// MASK-NOT: scf.if
// MASK: tt.load {{.*}} {MixCompileDiscreteMask}
// LOWER-LABEL: tt.func @row_and_column_mask(
// LOWER: ascend.indirect_load
tt.func @row_and_column_mask(%base: !tt.ptr<i8>, %offsets: tensor<4xi32>, %valid_rows: tensor<4xi1>, %valid_cols: tensor<64xi1>) -> tensor<4x64xi8> {
  %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %rows = tt.expand_dims %offsets {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %a = tt.broadcast %rows : tensor<4x1xi32> -> tensor<4x64xi32>
  %b = tt.broadcast %cols : tensor<1x64xi32> -> tensor<4x64xi32>
  %off = arith.addi %a, %b : tensor<4x64xi32>
  %bt = tt.splat %base : !tt.ptr<i8> -> tensor<4x64x!tt.ptr<i8>>
  %ptr = tt.addptr %bt, %off : tensor<4x64x!tt.ptr<i8>>, tensor<4x64xi32>
  %mr = tt.expand_dims %valid_rows {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %mc = tt.expand_dims %valid_cols {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
  %row_mask = tt.broadcast %mr : tensor<4x1xi1> -> tensor<4x64xi1>
  %col_mask = tt.broadcast %mc : tensor<1x64xi1> -> tensor<4x64xi1>
  %mask = arith.andi %row_mask, %col_mask : tensor<4x64xi1>
  %v = tt.load %ptr, %mask : tensor<4x64x!tt.ptr<i8>>
  tt.return %v : tensor<4x64xi8>
}
