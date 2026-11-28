// RUN: triton-opt %s --discrete-mask-access-conversion --triton-to-unstructure --verify-each | FileCheck %s --check-prefixes=MASK,STORE
// RUN: triton-opt %s --discrete-mask-access-conversion --triton-to-unstructure --bubble-up-operation --triton-to-linalg --verify-each | FileCheck %s
// RUN: sed 's/tt.broadcast %cpred /tt.broadcast %expanded_pred /' %s | triton-opt --discrete-mask-access-conversion --triton-to-unstructure --bubble-up-operation --triton-to-linalg --verify-each | FileCheck %s
// RUN: sed -e 's/tt.broadcast %cpred /tt.broadcast %expanded_pred /' -e 's/, %expanded_width : tensor<1x8xi32>/, %lane_widths : tensor<1x8xi32>/' %s | triton-opt --discrete-mask-access-conversion --triton-to-unstructure --bubble-up-operation --triton-to-linalg --verify-each | FileCheck %s --check-prefix=SCALAR
// RUN: sed 's/, %active, %other/, %rowmask, %other/' %s | triton-opt --discrete-mask-access-conversion --triton-to-unstructure --bubble-up-operation --triton-to-linalg --verify-each | FileCheck %s --check-prefix=FULL
// RUN: sed 's/cmpi slt, %columns/cmpi ne, %columns/' %s | triton-opt --discrete-mask-access-conversion --triton-to-unstructure --bubble-up-operation --triton-to-linalg --verify-each | FileCheck %s --check-prefix=SCALAR
// RUN: sed 's/%blockWidth = arith.constant 8/%blockWidth = arith.constant 1/' %s | triton-opt --discrete-mask-access-conversion --triton-to-unstructure --bubble-up-operation --triton-to-linalg --verify-each | FileCheck %s --check-prefix=SCALAR
// RUN: sed 's/, %mask, %constant_other :/, %mask, %inactive :/' %s | triton-opt --discrete-mask-access-conversion --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=PADDING

// Guard each contiguous row before the copy and clip it at the column bound.
// MASK-LABEL: tt.func public @masked_indirect_gather(
// MASK: %[[OTHER:.*]] = arith.constant dense<-7.000000e+00> : tensor<4x8xf32>
// MASK: %[[COL:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : tensor<8xi32>
// MASK: %[[PREFIX:.*]] = tt.expand_dims %[[COL]]
// MASK: scf.for %{{.*}} = {{.*}} iter_args(%[[ACC:[a-zA-Z0-9_]+]] = %[[OTHER]])
// MASK-NOT: scf.for
// MASK: scf.if
// MASK: tt.load %{{[^,]+}}, %[[PREFIX]], %{{.*}} : tensor<1x8x!tt.ptr<f32>>
// MASK: } else {
// MASK-NEXT: scf.yield %[[ACC]] : tensor<4x8xf32>
// CHECK-LABEL: func.func @masked_indirect_gather(
// CHECK: scf.for
// CHECK: %[[ROW:.*]] = arith.cmpi ne, %{{.*}}, %{{.*}} : i32
// CHECK: scf.if %[[ROW]]
// CHECK: %[[END:.*]] = arith.addi %[[START:[a-zA-Z0-9_]+]], %{{.*}} : index
// CHECK: %[[LOW:.*]] = arith.maxsi %[[START]], %{{.*}} : index
// CHECK: %[[HIGH:.*]] = arith.minsi %[[END]], %[[LOW]] : index
// CHECK: %[[WIDTH:.*]] = arith.subi %[[HIGH]], %[[START]] : index
// CHECK: %[[SRC:.*]] = memref.subview %{{.*}}[0, 0] [1, %[[WIDTH]]] [1, 1]
// CHECK: %[[DST:.*]] = memref.subview %{{.*}}[0, 0] [1, %[[WIDTH]]] [1, 1]
// CHECK: memref.copy %[[SRC]], %[[DST]] : memref<1x?xf32,
// CHECK: scf.yield
// FULL-LABEL: func.func @masked_indirect_gather(
// FULL: scf.for
// FULL: %[[ROW:.*]] = arith.cmpi ne, %{{.*}}, %{{.*}} : i32
// FULL: scf.if %[[ROW]]
// FULL: memref.copy %{{.*}}, %{{.*}} : memref<1x8xf32,
// FULL: scf.yield
// A hole in the mask or a column start that can wrap within this tile needs
// elementwise checks. Only the mask uses the start; addresses stay in bounds.
// SCALAR-LABEL: func.func @masked_indirect_gather(
// SCALAR: scf.for
// SCALAR: scf.for
// SCALAR: %[[ACTIVE:.*]] = arith.andi %{{.*}}, %{{.*}} : i1
// SCALAR: scf.if %[[ACTIVE]] -> (f32) {
// SCALAR-NEXT: %[[VALUE:.*]] = memref.load
// SCALAR-NEXT: scf.yield %[[VALUE]] : f32
tt.func public @masked_indirect_gather(%src: !tt.ptr<f32>, %indices: !tt.ptr<i32>, %width: i32, %tile: i32, %lane_widths: tensor<1x8xi32>) -> tensor<4x8xf32> {
  %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %ibase = tt.splat %indices : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %iptr = tt.addptr %ibase, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %ids = tt.load %iptr : tensor<4x!tt.ptr<i32>>
  %three = arith.constant dense<3> : tensor<4xi32>
  %rows = arith.andi %ids, %three : tensor<4xi32>
  %stride = arith.constant dense<8> : tensor<4xi32>
  %offset = arith.muli %rows, %stride : tensor<4xi32>
  %r2 = tt.expand_dims %offset {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %c2 = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %rb = tt.broadcast %r2 : tensor<4x1xi32> -> tensor<4x8xi32>
  %cb = tt.broadcast %c2 : tensor<1x8xi32> -> tensor<4x8xi32>
  %address = arith.addi %rb, %cb : tensor<4x8xi32>
  %base = tt.splat %src : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %address : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
  %validrow = arith.cmpi ne, %rows, %three : tensor<4xi32>
  %rpred = tt.expand_dims %validrow {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %rowmask = tt.broadcast %rpred : tensor<4x1xi1> -> tensor<4x8xi1>
  %widths = tt.splat %width : i32 -> tensor<8xi32>
  %blockWidth = arith.constant 8 : i32
  %start = arith.muli %tile, %blockWidth : i32
  %starts = tt.splat %start : i32 -> tensor<8xi32>
  %columns = arith.addi %starts, %c : tensor<8xi32>
  %validcol = arith.cmpi slt, %columns, %widths : tensor<8xi32>
  %cpred = tt.expand_dims %validcol {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  // Issue #2648 uses columns[None, :] < width, expanding before the comparison.
  %expanded_columns = tt.expand_dims %columns {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %expanded_width = tt.splat %width : i32 -> tensor<1x8xi32>
  %expanded_pred = arith.cmpi slt, %expanded_columns, %expanded_width : tensor<1x8xi32>
  %colmask = tt.broadcast %cpred : tensor<1x8xi1> -> tensor<4x8xi1>
  %active = arith.andi %rowmask, %colmask : tensor<4x8xi1>
  %other = arith.constant dense<-7.0> : tensor<4x8xf32>
  %value = tt.load %ptr, %active, %other : tensor<4x8x!tt.ptr<f32>>
  tt.return %value : tensor<4x8xf32>
}

// A uniform row predicate alone does not make per-lane addresses contiguous.
// CHECK-LABEL: func.func @opaque_row_offsets(
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.if %{{.*}} -> (f32) {
// CHECK-NEXT: %[[DATA:.*]] = memref.load
// CHECK-NEXT: scf.yield %[[DATA]] : f32
tt.func public @opaque_row_offsets(%src: !tt.ptr<f32>, %offsets: tensor<4x8xi32>, %rows: tensor<4x1xi32>) -> tensor<4x8xf32> {
  %bound = arith.constant dense<31> : tensor<4x8xi32>
  %index = arith.andi %offsets, %bound : tensor<4x8xi32>
  %base = tt.splat %src : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %index : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
  %zero = arith.constant dense<0> : tensor<4x1xi32>
  %valid = arith.cmpi ne, %rows, %zero : tensor<4x1xi32>
  %mask = tt.broadcast %valid : tensor<4x1xi1> -> tensor<4x8xi1>
  %value = tt.load %ptr, %mask : tensor<4x8x!tt.ptr<f32>>
  tt.return %value : tensor<4x8xf32>
}

// A branch can produce a materialized vector of predicates. Its lane value
// must survive conversion to the scalar guard on the indirect read.
// CHECK-LABEL: func.func @branch_selected_mask(
// CHECK: %[[SELECTED:.*]] = scf.if %{{.*}} -> (tensor<16xi1>)
// CHECK: %[[BYTES:.*]] = arith.extui %[[SELECTED]] : tensor<16xi1> to tensor<16xi8>
// CHECK: scf.for
// CHECK: %[[LANE:.*]] = tensor.extract %[[BYTES]][%{{.*}}] : tensor<16xi8>
// CHECK: %[[PRED:.*]] = arith.cmpi ne, %[[LANE]], %{{.*}} : i8
// CHECK: scf.if %[[PRED]] -> (i32) {
// CHECK-NEXT: %[[DATA:.*]] = memref.load
// CHECK-NEXT: scf.yield %[[DATA]] : i32
tt.func public @branch_selected_mask(%src: !tt.ptr<i32>, %offsets: tensor<16xi32>, %first: i1) -> tensor<16xi32> {
  %seven = arith.constant dense<7> : tensor<16xi32>
  %two = arith.constant dense<2> : tensor<16xi32>
  %rows = arith.andi %offsets, %seven : tensor<16xi32>
  %base = tt.splat %src : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %ptr = tt.addptr %base, %rows : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %mask = scf.if %first -> (tensor<16xi1>) {
    %low = arith.cmpi slt, %rows, %two : tensor<16xi32>
    scf.yield %low : tensor<16xi1>
  } else {
    %high = arith.cmpi sge, %rows, %two : tensor<16xi32>
    scf.yield %high : tensor<16xi1>
  }
  %value = tt.load %ptr, %mask : tensor<16x!tt.ptr<i32>>
  tt.return %value : tensor<16xi32>
}

// The store shares the column limit, so only the valid columns of the gather
// can reach the destination. A row that fails its predicate still writes other.
// STORE-LABEL: tt.func public @masked_gather_store(
// STORE: %[[COLS:.*]] = tt.broadcast %{{.*}} : tensor<1x8xi1> -> tensor<4x8xi1>
// STORE: %[[ACTIVE:.*]] = arith.andi %{{.*}}, %[[COLS]] : tensor<4x8xi1>
// STORE: %[[RESULT:.*]] = scf.for
// STORE-NOT: scf.if
// STORE: %[[ROWMASK:.*]] = tensor.extract_slice %[[ACTIVE]][%{{.*}}, 0] [1, 8] [1, 1]
// STORE-NEXT: %[[LOADED:.*]] = tt.load %{{[^,]+}}, %[[ROWMASK]], %{{.*}} : tensor<1x8x!tt.ptr<f32>>
// STORE-NEXT: tensor.insert_slice %[[LOADED]] into
// STORE-NOT: arith.select
// STORE: tt.store %{{[^,]+}}, %[[RESULT]], %[[COLS]]
// PADDING-LABEL: tt.func public @masked_gather_store(
// PADDING-SAME: %[[INACTIVE:[a-zA-Z0-9_]+]]: tensor<4x8xf32>
// PADDING: scf.for %{{.*}} = {{.*}} iter_args(%[[ACC:[a-zA-Z0-9_]+]] = %[[INACTIVE]])
// PADDING: scf.if %{{.*}} -> (tensor<4x8xf32>)
// PADDING: %[[ROWPAD:.*]] = tensor.extract_slice %[[INACTIVE]]
// PADDING: tt.load %{{[^,]+}}, %{{[^,]+}}, %[[ROWPAD]]
// PADDING: } else {
// PADDING-NEXT: scf.yield %[[ACC]] : tensor<4x8xf32>
tt.func public @masked_gather_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %indices: !tt.ptr<i32>, %width: i32, %inactive: tensor<4x8xf32>) {
  %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %idxbase = tt.splat %indices : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %idxptr = tt.addptr %idxbase, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %row_ids = tt.load %idxptr : tensor<4x!tt.ptr<i32>>
  %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %c2 = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %columns = tt.broadcast %c2 : tensor<1x8xi32> -> tensor<4x8xi32>
  %three = arith.constant dense<3> : tensor<4xi32>
  %eight = arith.constant dense<8> : tensor<4xi32>
  %rows = arith.andi %row_ids, %three : tensor<4xi32>
  %offset = arith.muli %rows, %eight : tensor<4xi32>
  %offset2 = tt.expand_dims %offset {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %rb = tt.broadcast %offset2 : tensor<4x1xi32> -> tensor<4x8xi32>
  %address = arith.addi %rb, %columns : tensor<4x8xi32>
  %base = tt.splat %src : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %address : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
  %validrow = arith.cmpi ne, %rows, %three : tensor<4xi32>
  %validrow2 = tt.expand_dims %validrow {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
  %rowmask = tt.broadcast %validrow2 : tensor<4x1xi1> -> tensor<4x8xi1>
  %widths = tt.splat %width : i32 -> tensor<1x8xi32>
  %validcol = arith.cmpi slt, %c2, %widths : tensor<1x8xi32>
  %colmask = tt.broadcast %validcol : tensor<1x8xi1> -> tensor<4x8xi1>
  %mask = arith.andi %rowmask, %colmask : tensor<4x8xi1>
  %constant_other = arith.constant dense<-7.0> : tensor<4x8xf32>
  %value = tt.load %ptr, %mask, %constant_other : tensor<4x8x!tt.ptr<f32>>
  %rowoffsets = arith.muli %r, %eight : tensor<4xi32>
  %r2 = tt.expand_dims %rowoffsets {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %outrows = tt.broadcast %r2 : tensor<4x1xi32> -> tensor<4x8xi32>
  %outoffset = arith.addi %outrows, %columns : tensor<4x8xi32>
  %outbase = tt.splat %dst : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
  %outptr = tt.addptr %outbase, %outoffset : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
  tt.store %outptr, %value, %colmask : tensor<4x8x!tt.ptr<f32>>
  tt.return
}
