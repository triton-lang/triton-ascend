// RUN: triton-opt %s --triton-to-linalg --canonicalize --cse --verify-each --split-input-file | FileCheck %s --implicit-check-not=tt.load

// These inputs are the extracted load loops produced by Unstructure. The
// consumer replaces their tensor accumulator with a shared DMA allocation.
// Padding may be reused only when that allocation is initialized and each
// iteration writes a distinct slice.

// The fill precedes the dynamic loop, including rows=0 or cols=0.
// No per-row fill may be introduced for the same padding.
// CHECK-LABEL: func.func @disjoint_initialized_rows(
// CHECK: %[[FULL:.*]] = memref.alloc() : memref<2x8xf32>
// CHECK: linalg.fill {{.*}} outs(%[[FULL]] : memref<2x8xf32>)
// CHECK: scf.for
// CHECK-NOT: linalg.fill
// CHECK: %[[ROW:.*]] = memref.subview %[[FULL]]
// CHECK-NOT: linalg.fill
// CHECK: memref.copy {{.*}} : memref<1x?xf32{{.*}}> to memref<1x?xf32{{.*}}>
// CHECK-NOT: linalg.fill
// CHECK: return
module {
  tt.func public @disjoint_initialized_rows(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %audit: !tt.ptr<f32>, %rows: index, %cols: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %width = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %other = arith.constant dense<0.0> : tensor<1x8xf32>
    %empty = tensor.empty() : tensor<2x8xf32>
    %initial = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x8xf32>) -> tensor<2x8xf32>
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %col = tt.expand_dims %r {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %audit_base = tt.splat %audit : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
    %audit_ptr = tt.addptr %audit_base, %col : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
    %result = scf.for %i = %c0 to %rows step %c1 iter_args(%tile = %initial) -> tensor<2x8xf32> {
      %offset = arith.muli %i, %width : index
      %offset64 = arith.index_cast %offset : index to i64
      %row = tt.addptr %src, %offset64 : !tt.ptr<f32>, i64
      %base = tt.splat %row : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ptr = tt.addptr %base, %col : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %limit = tt.splat %cols : i32 -> tensor<1x8xi32>
      %mask = arith.cmpi slt, %col, %limit : tensor<1x8xi32>
      %value = tt.load %ptr, %mask, %other {DiscreteMemAccess} : tensor<1x8x!tt.ptr<f32>>
      %next = tensor.insert_slice %value into %tile[%i, 0] [1, 8] [1, 1] : tensor<1x8xf32> into tensor<2x8xf32>
      scf.yield %next : tensor<2x8xf32>
    } {ExtractedLoadOrStore}
    %out_rows = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %out_width = arith.constant dense<8> : tensor<2xi32>
    %out_offsets = arith.muli %out_rows, %out_width : tensor<2xi32>
    %out_re = tt.expand_dims %out_offsets {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %out_rb = tt.broadcast %out_re : tensor<2x1xi32> -> tensor<2x8xi32>
    %out_cb = tt.broadcast %col : tensor<1x8xi32> -> tensor<2x8xi32>
    %out_offset = arith.addi %out_rb, %out_cb : tensor<2x8xi32>
    %out_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_base, %out_offset : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
    tt.store %out_ptr, %result : tensor<2x8x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// The second iteration writes the same row with only three valid columns.
// Its inactive tail must overwrite the five values from the first iteration.
// CHECK-LABEL: func.func @overlapping_rows(
// CHECK: %[[FULL:.*]] = memref.alloc() : memref<2x8xf32>
// CHECK: linalg.fill {{.*}} outs(%[[FULL]] : memref<2x8xf32>)
// CHECK: scf.for
// CHECK: %[[ROW:.*]] = memref.subview %[[FULL]][0, 0] [1, 8] [1, 1]
// CHECK: linalg.fill {{.*}} outs(%[[ROW]] : memref<1x8xf32
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @overlapping_rows(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %audit: !tt.ptr<f32>, %rows: index, %cols: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %width = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %other = arith.constant dense<0.0> : tensor<1x8xf32>
    %empty = tensor.empty() : tensor<2x8xf32>
    %initial = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x8xf32>) -> tensor<2x8xf32>
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %col = tt.expand_dims %r {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %audit_base = tt.splat %audit : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
    %audit_ptr = tt.addptr %audit_base, %col : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
    %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%tile = %initial) -> tensor<2x8xf32> {
      %offset = arith.muli %i, %width : index
      %offset64 = arith.index_cast %offset : index to i64
      %row = tt.addptr %src, %offset64 : !tt.ptr<f32>, i64
      %base = tt.splat %row : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ptr = tt.addptr %base, %col : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %iv = arith.index_cast %i : index to i32
      %five = arith.constant 5 : i32
      %eight = arith.constant 8 : i32
      %drop = arith.muli %iv, %five : i32
      %tail_cols = arith.subi %eight, %drop : i32
      %limit = tt.splat %tail_cols : i32 -> tensor<1x8xi32>
      %mask = arith.cmpi slt, %col, %limit : tensor<1x8xi32>
      %value = tt.load %ptr, %mask, %other {DiscreteMemAccess} : tensor<1x8x!tt.ptr<f32>>
      %next = tensor.insert_slice %value into %tile[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> into tensor<2x8xf32>
      scf.yield %next : tensor<2x8xf32>
    } {ExtractedLoadOrStore}
    %out_rows = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %out_width = arith.constant dense<8> : tensor<2xi32>
    %out_offsets = arith.muli %out_rows, %out_width : tensor<2xi32>
    %out_re = tt.expand_dims %out_offsets {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %out_rb = tt.broadcast %out_re : tensor<2x1xi32> -> tensor<2x8xi32>
    %out_cb = tt.broadcast %col : tensor<1x8xi32> -> tensor<2x8xi32>
    %out_offset = arith.addi %out_rb, %out_cb : tensor<2x8xi32>
    %out_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_base, %out_offset : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
    tt.store %out_ptr, %result : tensor<2x8x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A second load user prevents reuse of the tensor accumulator.
// The standalone row allocation must receive padding before either use.
// CHECK-LABEL: func.func @load_with_second_use(
// CHECK: scf.for
// CHECK: %[[ROW:.*]] = memref.alloc() : memref<1x8xf32>
// CHECK: linalg.fill {{.*}} outs(%[[ROW]] : memref<1x8xf32>)
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @load_with_second_use(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %audit: !tt.ptr<f32>, %rows: index, %cols: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %width = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %other = arith.constant dense<0.0> : tensor<1x8xf32>
    %empty = tensor.empty() : tensor<2x8xf32>
    %initial = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x8xf32>) -> tensor<2x8xf32>
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %col = tt.expand_dims %r {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %audit_base = tt.splat %audit : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
    %audit_ptr = tt.addptr %audit_base, %col : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
    %result = scf.for %i = %c0 to %rows step %c1 iter_args(%tile = %initial) -> tensor<2x8xf32> {
      %offset = arith.muli %i, %width : index
      %offset64 = arith.index_cast %offset : index to i64
      %row = tt.addptr %src, %offset64 : !tt.ptr<f32>, i64
      %base = tt.splat %row : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ptr = tt.addptr %base, %col : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %limit = tt.splat %cols : i32 -> tensor<1x8xi32>
      %mask = arith.cmpi slt, %col, %limit : tensor<1x8xi32>
      %value = tt.load %ptr, %mask, %other {DiscreteMemAccess} : tensor<1x8x!tt.ptr<f32>>
      tt.store %audit_ptr, %value : tensor<1x8x!tt.ptr<f32>>
      %next = tensor.insert_slice %value into %tile[%i, 0] [1, 8] [1, 1] : tensor<1x8xf32> into tensor<2x8xf32>
      scf.yield %next : tensor<2x8xf32>
    } {ExtractedLoadOrStore}
    %out_rows = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %out_width = arith.constant dense<8> : tensor<2xi32>
    %out_offsets = arith.muli %out_rows, %out_width : tensor<2xi32>
    %out_re = tt.expand_dims %out_offsets {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %out_rb = tt.broadcast %out_re : tensor<2x1xi32> -> tensor<2x8xi32>
    %out_cb = tt.broadcast %col : tensor<1x8xi32> -> tensor<2x8xi32>
    %out_offset = arith.addi %out_rb, %out_cb : tensor<2x8xi32>
    %out_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_base, %out_offset : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
    tt.store %out_ptr, %result : tensor<2x8x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Positive zero in the destination does not replace negative-zero other.
// CHECK-LABEL: func.func @negative_zero_padding(
// CHECK: %[[NEGZERO:.*]] = arith.constant -0.000000e+00 : f32
// CHECK: %[[FULL:.*]] = memref.alloc() : memref<2x8xf32>
// CHECK: linalg.fill {{.*}} outs(%[[FULL]] : memref<2x8xf32>)
// CHECK: scf.for
// CHECK: %[[ROW:.*]] = memref.subview %[[FULL]]
// CHECK: linalg.fill ins(%[[NEGZERO]] : f32) outs(%[[ROW]] : memref<1x8xf32
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @negative_zero_padding(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %audit: !tt.ptr<f32>, %rows: index, %cols: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %width = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %other = arith.constant dense<-0.0> : tensor<1x8xf32>
    %empty = tensor.empty() : tensor<2x8xf32>
    %initial = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x8xf32>) -> tensor<2x8xf32>
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %col = tt.expand_dims %r {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %audit_base = tt.splat %audit : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
    %audit_ptr = tt.addptr %audit_base, %col : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
    %result = scf.for %i = %c0 to %rows step %c1 iter_args(%tile = %initial) -> tensor<2x8xf32> {
      %offset = arith.muli %i, %width : index
      %offset64 = arith.index_cast %offset : index to i64
      %row = tt.addptr %src, %offset64 : !tt.ptr<f32>, i64
      %base = tt.splat %row : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ptr = tt.addptr %base, %col : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %limit = tt.splat %cols : i32 -> tensor<1x8xi32>
      %mask = arith.cmpi slt, %col, %limit : tensor<1x8xi32>
      %value = tt.load %ptr, %mask, %other {DiscreteMemAccess} : tensor<1x8x!tt.ptr<f32>>
      %next = tensor.insert_slice %value into %tile[%i, 0] [1, 8] [1, 1] : tensor<1x8xf32> into tensor<2x8xf32>
      scf.yield %next : tensor<2x8xf32>
    } {ExtractedLoadOrStore}
    %out_rows = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %out_width = arith.constant dense<8> : tensor<2xi32>
    %out_offsets = arith.muli %out_rows, %out_width : tensor<2xi32>
    %out_re = tt.expand_dims %out_offsets {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %out_rb = tt.broadcast %out_re : tensor<2x1xi32> -> tensor<2x8xi32>
    %out_cb = tt.broadcast %col : tensor<1x8xi32> -> tensor<2x8xi32>
    %out_offset = arith.addi %out_rb, %out_cb : tensor<2x8xi32>
    %out_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_base, %out_offset : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
    tt.store %out_ptr, %result : tensor<2x8x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// The extracted axis is distinct from the retained, masked row axis. A
// singleton retained row would fold its range to a scalar predicate, which
// Unstructure must handle before this consumer. Keep a two-row prefix here.
// The stride-two consumer must initialize its wide scratch and write the
// extracted lanes back to the original destination slice.
// CHECK-LABEL: func.func @masked_stride_two_rows(
// CHECK: %[[FULL:.*]] = memref.alloc() : memref<2x2x8xf32>
// CHECK: linalg.fill {{.*}} outs(%[[FULL]] : memref<2x2x8xf32>)
// CHECK: scf.for
// CHECK: %[[ROW:.*]] = memref.subview %[[FULL]]
// CHECK: %[[WIDE:.*]] = memref.alloc() : memref<1x2x16xf32>
// CHECK: linalg.fill {{.*}} outs(%[[WIDE]] : memref<1x2x16xf32>)
// CHECK: memref.copy
// CHECK: %[[LANES:.*]] = tensor.extract_slice {{.*}}[0, 0, 0] [1, 2, 8] [1, 1, 2]
// CHECK: bufferization.materialize_in_destination %[[LANES]] in writable %[[ROW]]
// CHECK: return
module {
  tt.func public @masked_stride_two_rows(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %tiles: index, %rows: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %width = arith.constant 32 : index
    %zero = arith.constant 0.0 : f32
    %other = arith.constant dense<0.0> : tensor<1x2x8xf32>
    %empty = tensor.empty() : tensor<2x2x8xf32>
    %initial = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %sixteen = arith.constant dense<16> : tensor<2xi32>
    %two = arith.constant dense<2> : tensor<8xi32>
    %row_offset = arith.muli %r, %sixteen : tensor<2xi32>
    %col_offset = arith.muli %c, %two : tensor<8xi32>
    %re = tt.expand_dims %row_offset {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %ree = tt.expand_dims %re {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32>
    %ce = tt.expand_dims %col_offset {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %cee = tt.expand_dims %ce {axis = 0 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32>
    %rb = tt.broadcast %ree : tensor<1x2x1xi32> -> tensor<1x2x8xi32>
    %cb = tt.broadcast %cee : tensor<1x1x8xi32> -> tensor<1x2x8xi32>
    %offsets = arith.addi %rb, %cb : tensor<1x2x8xi32>
    %limit = tt.splat %rows : i32 -> tensor<2xi32>
    %valid = arith.cmpi slt, %r, %limit : tensor<2xi32>
    %me = tt.expand_dims %valid {axis = 0 : i32} : tensor<2xi1> -> tensor<1x2xi1>
    %mee = tt.expand_dims %me {axis = 2 : i32} : tensor<1x2xi1> -> tensor<1x2x1xi1>
    %mask = tt.broadcast %mee : tensor<1x2x1xi1> -> tensor<1x2x8xi1>
    %result = scf.for %i = %c0 to %tiles step %c1 iter_args(%tile = %initial) -> tensor<2x2x8xf32> {
      %offset = arith.muli %i, %width : index
      %offset64 = arith.index_cast %offset : index to i64
      %origin = tt.addptr %src, %offset64 : !tt.ptr<f32>, i64
      %base = tt.splat %origin : !tt.ptr<f32> -> tensor<1x2x8x!tt.ptr<f32>>
      %ptr = tt.addptr %base, %offsets : tensor<1x2x8x!tt.ptr<f32>>, tensor<1x2x8xi32>
      %value = tt.load %ptr, %mask, %other {DiscreteMemAccess} : tensor<1x2x8x!tt.ptr<f32>>
      %next = tensor.insert_slice %value into %tile[%i, 0, 0] [1, 2, 8] [1, 1, 1] : tensor<1x2x8xf32> into tensor<2x2x8xf32>
      scf.yield %next : tensor<2x2x8xf32>
    } {ExtractedLoadOrStore}
    %flat = tt.reshape %result : tensor<2x2x8xf32> -> tensor<32xf32>
    %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %out_base = tt.splat %dst : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_base, %range : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %out_ptr, %flat : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
