// RUN: triton-opt --triton-to-linalg --verify-each --split-input-file %s | FileCheck %s

// An i32 parent and coordinate can be subtracted safely in index64. Both
// accesses must reuse the GM view's clamped origin and agree on tile padding.
// CHECK-LABEL: func.func @signed_i32_parent(
// CHECK-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[DST:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[ROW:[^ ,]+]]: i32
// CHECK-SAME: %[[ROWS:[^ ,]+]]: i32
// CHECK-DAG: %[[ZERO32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[TILE:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[COORD:.*]] = arith.index_cast %[[ROW]] : i32 to index
// CHECK-DAG: %[[PARENT:.*]] = arith.index_cast %[[ROWS]] : i32 to index
// CHECK-DAG: %[[CLAMP:.*]] = arith.maxsi %[[ROW]], %[[ZERO32]] : i32
// CHECK-DAG: %[[GM:.*]] = arith.index_cast %[[CLAMP]] : i32 to index
// CHECK: %[[INPUT:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [%[[GM]]], sizes: [32], strides: [1]
// CHECK: %[[OUTPUT:.*]] = memref.reinterpret_cast %[[DST]] to offset: [%[[GM]]], sizes: [32], strides: [1]
// CHECK: %[[NEG:.*]] = arith.subi %[[ZERO]], %[[COORD]] : index
// CHECK: %[[LEFT:.*]] = arith.maxsi %[[NEG]], %[[ZERO]] : index
// CHECK: %[[START:.*]] = arith.minsi %[[LEFT]], %[[TILE]] : index
// CHECK: %[[REMAINING:.*]] = arith.subi %[[PARENT]], %[[GM]] : index
// CHECK: %[[AVAILABLE:.*]] = arith.maxsi %[[REMAINING]], %[[ZERO]] : index
// CHECK: %[[CAPACITY:.*]] = arith.subi %[[TILE]], %[[START]] : index
// CHECK: %[[COUNT:.*]] = arith.minsi %[[AVAILABLE]], %[[CAPACITY]] : index
// CHECK: %[[GM_READ:.*]] = memref.subview %[[INPUT]][0] [%[[COUNT]]] [1]
// CHECK: %[[UB_WRITE:.*]] = memref.subview %{{.*}}[%[[START]]] [%[[COUNT]]] [1]
// CHECK: memref.copy %[[GM_READ]], %[[UB_WRITE]]
// CHECK: %[[UB_READ:.*]] = tensor.extract_slice %{{.*}}[%[[START]]] [%[[COUNT]]] [1]
// CHECK: %[[GM_WRITE:.*]] = memref.subview %[[OUTPUT]][0] [%[[COUNT]]] [1]
// CHECK: bufferization.materialize_in_destination %[[UB_READ]] in writable %[[GM_WRITE]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @signed_i32_parent(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %row: i32, %rows: i32) {
    %parent = arith.extsi %rows : i32 to i64
    %one = arith.constant 1 : i64
    %p = tt.make_tensor_ptr %src, [%parent], [%one], [%row] {order = array<i32: 0>} : <tensor<32xf32>>
    %q = tt.make_tensor_ptr %dst, [%parent], [%one], [%row] {order = array<i32: 0>} : <tensor<32xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32xf32>>
    tt.store %q, %v {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<32xf32>>
    tt.return
  }
}

// -----

// A constant parent need not fit in i32: 2^32 minus any i32 coordinate fits
// in index64. The same clamped coordinate must feed the view and the length.
// CHECK-LABEL: func.func @safe_constant_parent(
// CHECK-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// CHECK-DAG: %[[PARENT:.*]] = arith.constant 4294967296 : index
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [%[[GM:.*]]], sizes: [32], strides: [1]
// CHECK: %[[REMAINING:.*]] = arith.subi %[[PARENT]], %[[GM]] : index
// CHECK: %[[AVAILABLE:.*]] = arith.maxsi %[[REMAINING]], %{{.*}} : index
// CHECK: %[[CAPACITY:.*]] = arith.subi %{{.*}}, %{{.*}} : index
// CHECK: %[[COUNT:.*]] = arith.minsi %[[AVAILABLE]], %[[CAPACITY]] : index
// CHECK: memref.subview %[[VIEW]][0] [%[[COUNT]]] [1]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @safe_constant_parent(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %row: i32) {
    %parent = arith.constant 4294967296 : i64
    %tile = arith.constant 32 : i64
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    %p = tt.make_tensor_ptr %src, [%parent], [%one], [%row] {order = array<i32: 0>} : <tensor<32xf32>>
    %q = tt.make_tensor_ptr %dst, [%tile], [%one], [%zero] {order = array<i32: 0>} : <tensor<32xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32xf32>>
    tt.store %q, %v : !tt.ptr<tensor<32xf32>>
    tt.return
  }
}

// -----

// Unknown i64 parents retain the original endpoint formula. The checked GM
// view still has offset max(row,0); the fallback subtracts the *raw* row.
// CHECK-LABEL: func.func @dynamic_i64_parent_fallback(
// CHECK-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[DST:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[ROW:[^ ,]+]]: i32
// CHECK-SAME: %[[ROWS:[^ ,]+]]: i64
// CHECK-DAG: %[[COORD:.*]] = arith.index_cast %[[ROW]] : i32 to index
// CHECK-DAG: %[[PARENT:.*]] = arith.index_cast %[[ROWS]] : i64 to index
// CHECK-DAG: %[[TILE:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[INPUT:.*]] = memref.reinterpret_cast %[[SRC]]
// CHECK: %[[OUTPUT:.*]] = memref.reinterpret_cast %[[DST]]
// CHECK: %[[NEG:.*]] = arith.subi %[[ZERO]], %[[COORD]] : index
// CHECK: %[[LEFT:.*]] = arith.maxsi %[[NEG]], %[[ZERO]] : index
// CHECK: %[[START:.*]] = arith.minsi %[[LEFT]], %[[TILE]] : index
// CHECK: %[[REMAINING:.*]] = arith.subi %[[PARENT]], %[[COORD]] : index
// CHECK: %[[AVAILABLE:.*]] = arith.maxsi %[[REMAINING]], %[[ZERO]] : index
// CHECK: %[[END:.*]] = arith.minsi %[[AVAILABLE]], %[[TILE]] : index
// CHECK: %[[LENGTH:.*]] = arith.subi %[[END]], %[[START]] : index
// CHECK: %[[COUNT:.*]] = arith.maxsi %[[LENGTH]], %[[ZERO]] : index
// CHECK: %[[GM_READ:.*]] = memref.subview %[[INPUT]][0] [%[[COUNT]]] [1]
// CHECK: %[[UB_WRITE:.*]] = memref.subview %{{.*}}[%[[START]]] [%[[COUNT]]] [1]
// CHECK: memref.copy %[[GM_READ]], %[[UB_WRITE]]
// CHECK: %[[UB_READ:.*]] = tensor.extract_slice %{{.*}}[%[[START]]] [%[[COUNT]]] [1]
// CHECK: %[[GM_WRITE:.*]] = memref.subview %[[OUTPUT]][0] [%[[COUNT]]] [1]
// CHECK: bufferization.materialize_in_destination %[[UB_READ]] in writable %[[GM_WRITE]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @dynamic_i64_parent_fallback(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %row: i32, %rows: i64) {
    %one = arith.constant 1 : i64
    %p = tt.make_tensor_ptr %src, [%rows], [%one], [%row] {order = array<i32: 0>} : <tensor<32xf32>>
    %q = tt.make_tensor_ptr %dst, [%rows], [%one], [%row] {order = array<i32: 0>} : <tensor<32xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32xf32>>
    tt.store %q, %v {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<32xf32>>
    tt.return
  }
}

// -----

// INT64_MAX minus a negative coordinate may overflow. Being a constant is
// insufficient to use the new formula; this must retain the endpoint path.
// CHECK-LABEL: func.func @large_constant_parent_fallback(
// CHECK-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[DST:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[ROW:[^ ,]+]]: i32
// CHECK-DAG: %[[COORD:.*]] = arith.index_cast %[[ROW]] : i32 to index
// CHECK-DAG: %[[PARENT:.*]] = arith.constant 9223372036854775807 : index
// CHECK-DAG: %[[TILE:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: memref.reinterpret_cast %[[SRC]]
// CHECK: %[[REMAINING:.*]] = arith.subi %[[PARENT]], %[[COORD]] : index
// CHECK: %[[AVAILABLE:.*]] = arith.maxsi %[[REMAINING]], %[[ZERO]] : index
// CHECK: %[[END:.*]] = arith.minsi %[[AVAILABLE]], %[[TILE]] : index
// CHECK: %[[LENGTH:.*]] = arith.subi %[[END]], %{{.*}} : index
// CHECK: %[[COUNT:.*]] = arith.maxsi %[[LENGTH]], %[[ZERO]] : index
// CHECK: memref.subview %{{.*}}[0] [%[[COUNT]]] [1]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @large_constant_parent_fallback(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %row: i32) {
    %parent = arith.constant 9223372036854775807 : i64
    %tile = arith.constant 32 : i64
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    %p = tt.make_tensor_ptr %src, [%parent], [%one], [%row] {order = array<i32: 0>} : <tensor<32xf32>>
    %q = tt.make_tensor_ptr %dst, [%tile], [%one], [%zero] {order = array<i32: 0>} : <tensor<32xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32xf32>>
    tt.store %q, %v : !tt.ptr<tensor<32xf32>>
    tt.return
  }
}

// -----

// P=5, B=32, x=-2 has five valid elements at tile lanes [2,7). Padding is
// deducted from tile capacity, not from the five remaining parent elements.
// CHECK-LABEL: func.func @small_parent_negative_prefix
// CHECK: %[[INPUT:.*]] = memref.reinterpret_cast {{.*}}sizes: [32], strides: [1]
// CHECK: %[[OUTPUT:.*]] = memref.reinterpret_cast {{.*}}sizes: [32], strides: [1]
// CHECK: %[[GM_READ:.*]] = memref.subview %[[INPUT]][0] [5] [1]
// CHECK: %[[UB_WRITE:.*]] = memref.subview %{{.*}}[2] [5] [1]
// CHECK: memref.copy %[[GM_READ]], %[[UB_WRITE]]
// CHECK: %[[UB_READ:.*]] = tensor.extract_slice %{{.*}}[2] [5] [1]
// CHECK: %[[GM_WRITE:.*]] = memref.subview %[[OUTPUT]][0] [5] [1]
// CHECK: bufferization.materialize_in_destination %[[UB_READ]] in writable %[[GM_WRITE]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @small_parent_negative_prefix(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %parent = arith.constant 5 : i64
    %one = arith.constant 1 : i64
    %row = arith.constant -2 : i32
    %p = tt.make_tensor_ptr %src, [%parent], [%one], [%row] {order = array<i32: 0>} : <tensor<32xf32>>
    %q = tt.make_tensor_ptr %dst, [%parent], [%one], [%row] {order = array<i32: 0>} : <tensor<32xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32xf32>>
    tt.store %q, %v {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<32xf32>>
    tt.return
  }
}

// -----

// A 32-aligned row can have a partial right edge. The 128-aligned column
// covers the whole parent column axis or none of it. Negative starts on
// either axis give an empty transfer. Both local offsets are static zero.
// CHECK-LABEL: func.func @tile_aligned_boundary(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[TILE:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[COLS:.*]] = arith.constant 128 : index
// CHECK: %[[ROW_OK:.*]] = arith.cmpi sge, %{{.*}}, %[[ZERO]] : index
// CHECK: %[[ROW_COUNT:.*]] = arith.minsi {{.*}}, %[[TILE]] : index
// CHECK: %[[ROWS_VALID:.*]] = arith.select %[[ROW_OK]], %[[ROW_COUNT]], %[[ZERO]] : index
// CHECK: %[[COL_OK:.*]] = arith.cmpi ult, %{{.*}}, %[[COLS]] : index
// CHECK: %[[COLS_VALID:.*]] = arith.select %[[COL_OK]], %[[COLS]], %[[ZERO]] : index
// CHECK: memref.subview %{{.*}}[0, 0] [%[[ROWS_VALID]], %[[COLS_VALID]]]
// CHECK: memref.subview %{{.*}}[0, 0] [%[[ROWS_VALID]], %[[COLS_VALID]]]
// CHECK: memref.copy
// CHECK: tensor.extract_slice %{{.*}}[0, 0] [%[[ROWS_VALID]], %[[COLS_VALID]]]
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tile_aligned_boundary(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %r: i32, %c: i32, %rows: i32) {
    %rows64 = arith.extsi %rows : i32 to i64
    %tile = arith.constant 32 : i32
    %cols32 = arith.constant 128 : i32
    %cols = arith.constant 128 : i64
    %one = arith.constant 1 : i64
    %row = arith.muli %r, %tile : i32
    %col = arith.muli %c, %cols32 : i32
    %p = tt.make_tensor_ptr %src, [%rows64, %cols], [%cols, %one], [%row, %col] {order = array<i32: 1, 0>} : <tensor<32x128xf32>>
    %q = tt.make_tensor_ptr %dst, [%rows64, %cols], [%cols, %one], [%row, %col] {order = array<i32: 1, 0>} : <tensor<32x128xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x128xf32>>
    tt.store %q, %v {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x128xf32>>
    tt.return
  }
}
