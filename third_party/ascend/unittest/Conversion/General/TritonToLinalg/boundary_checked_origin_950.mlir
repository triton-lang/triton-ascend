// RUN: triton-opt --triton-to-linalg --verify-each --split-input-file %s | FileCheck %s

// Keep the address origin separate from the tile intersection: only the GM
// view uses the nonnegative coordinate, while UB lanes retain the original
// leading padding. Check the i32 clamp before its signed cast to index.
// CHECK-LABEL: func.func @dynamic_checked_origin(
// CHECK-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[DST:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[ROW:[^ ,]+]]: i32
// CHECK-DAG: %[[ZERO32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[STRIDE:.*]] = arith.constant 16 : index
// CHECK: %[[CLAMPED32:.*]] = arith.maxsi %[[ROW]], %[[ZERO32]] : i32
// CHECK: %[[CLAMPED:.*]] = arith.index_cast %[[CLAMPED32]] : i32 to index
// CHECK: %[[OFFSET:.*]] = arith.muli %[[CLAMPED]], %[[STRIDE]] : index
// CHECK: %[[INPUT:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [%[[OFFSET]]], sizes: [8, 16], strides: [16, 1]
// CHECK: %[[OUTPUT:.*]] = memref.reinterpret_cast %[[DST]] to offset: [%[[OFFSET]]], sizes: [8, 16], strides: [16, 1]
// CHECK: %[[GM_READ:.*]] = memref.subview %[[INPUT]][0, 0] [%[[COUNT:.*]], 16] [1, 1]
// CHECK: %[[UB_WRITE:.*]] = memref.subview %{{.*}}[%[[START:.*]], 0] [%[[COUNT]], 16] [1, 1]
// CHECK: memref.copy %[[GM_READ]], %[[UB_WRITE]]
// CHECK: %[[UB_READ:.*]] = tensor.extract_slice %{{.*}}[%[[START]], 0] [%[[COUNT]], 16] [1, 1]
// CHECK: %[[GM_WRITE:.*]] = memref.subview %[[OUTPUT]][0, 0] [%[[COUNT]], 16] [1, 1]
// CHECK: bufferization.materialize_in_destination %[[UB_READ]] in writable %[[GM_WRITE]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @dynamic_checked_origin(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %row: i32, %rows: i64) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %cols = arith.constant 16 : i64
    %p = tt.make_tensor_ptr %src, [%rows, %cols], [%cols, %one], [%row, %zero] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    %q = tt.make_tensor_ptr %dst, [%rows, %cols], [%cols, %one], [%row, %zero] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<8x16xf32>>
    tt.store %q, %v {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
    tt.return
  }
}

// -----

// r=-2, P=4, B=8 selects tile lanes [2,6), exactly four elements.
// The valid source addresses are src+[17,18,19,20], and the destinations
// are dst+[29,30,31,32]. Neither scalar base displacement may be added twice.
// CHECK-LABEL: func.func @negative_prefix_preserves_base
// CHECK: %[[INPUT:.*]] = memref.reinterpret_cast {{.*}} to offset: [17], sizes: [8], strides: [1]
// CHECK: %[[OUTPUT:.*]] = memref.reinterpret_cast {{.*}} to offset: [29], sizes: [8], strides: [1]
// CHECK: linalg.fill
// CHECK: %[[GM_READ:.*]] = memref.subview %[[INPUT]][0] [4] [1]
// CHECK: %[[UB_WRITE:.*]] = memref.subview %{{.*}}[2] [4] [1]
// CHECK: memref.copy %[[GM_READ]], %[[UB_WRITE]]
// CHECK: %[[UB_READ:.*]] = tensor.extract_slice %{{.*}}[2] [4] [1]
// CHECK: %[[GM_WRITE:.*]] = memref.subview %[[OUTPUT]][0] [4] [1]
// CHECK: bufferization.materialize_in_destination %[[UB_READ]] in writable %[[GM_WRITE]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @negative_prefix_preserves_base(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %parent = arith.constant 4 : i64
    %stride = arith.constant 1 : i64
    %row = arith.constant -2 : i32
    %src_shift = arith.constant 17 : i32
    %dst_shift = arith.constant 29 : i32
    %src_base = tt.addptr %src, %src_shift : !tt.ptr<f32>, i32
    %dst_base = tt.addptr %dst, %dst_shift : !tt.ptr<f32>, i32
    %p = tt.make_tensor_ptr %src_base, [%parent], [%stride], [%row] {order = array<i32: 0>} : <tensor<8xf32>>
    %q = tt.make_tensor_ptr %dst_base, [%parent], [%stride], [%row] {order = array<i32: 0>} : <tensor<8xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
    tt.store %q, %v {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<8xf32>>
    tt.return
  }
}

// -----

// Only columns are checked. Row -1 must remain in the origin:
// 20 + (-1)*16 + max(-2,0) = 4. The two valid columns occupy UB [0,2].
// Clamping both coordinates would incorrectly change the origin to 20.
// CHECK-LABEL: func.func @unchecked_negative_axis
// CHECK: %[[INPUT:.*]] = memref.reinterpret_cast {{.*}} to offset: [4], sizes: [2, 4], strides: [16, 1]
// CHECK: %[[GM:.*]] = memref.subview %[[INPUT]][0, 0] [2, 2] [1, 1]
// CHECK: %[[UB:.*]] = memref.subview %{{.*}}[0, 2] [2, 2] [1, 1]
// CHECK: memref.copy %[[GM]], %[[UB]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @unchecked_negative_axis(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %one = arith.constant 1 : i64
    %rows = arith.constant 2 : i64
    %parent_cols = arith.constant 3 : i64
    %cols = arith.constant 4 : i64
    %stride = arith.constant 16 : i64
    %base_shift = arith.constant 20 : i32
    %row = arith.constant -1 : i32
    %col = arith.constant -2 : i32
    %zero = arith.constant 0 : i32
    %base = tt.addptr %src, %base_shift : !tt.ptr<f32>, i32
    %p = tt.make_tensor_ptr %base, [%rows, %parent_cols], [%stride, %one], [%row, %col] {order = array<i32: 1, 0>} : <tensor<2x4xf32>>
    %q = tt.make_tensor_ptr %dst, [%rows, %cols], [%cols, %one], [%zero, %zero] {order = array<i32: 1, 0>} : <tensor<2x4xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<2x4xf32>>
    tt.store %q, %v : !tt.ptr<tensor<2x4xf32>>
    tt.return
  }
}

// -----

// The negative-stride load reads src+[31,29,27,25], into UB lanes [2,6).
// The zero-stride load reads its one address into the same four logical lanes.
// Bounds depend on coordinates even when the physical stride is zero.
// CHECK-LABEL: func.func @nonpositive_strides
// CHECK: %[[NEG:.*]] = memref.reinterpret_cast {{.*}} to offset: [31], sizes: [8], strides: [-2]
// CHECK: %[[ZERO:.*]] = memref.reinterpret_cast {{.*}} to offset: [0], sizes: [8], strides: [0]
// CHECK: %[[NEG_GM:.*]] = memref.subview %[[NEG]][0] [4] [1]
// CHECK: %[[NEG_UB:.*]] = memref.subview %{{.*}}[2] [4] [1]
// CHECK: memref.copy %[[NEG_GM]], %[[NEG_UB]]
// CHECK: %[[ZERO_GM:.*]] = memref.subview %[[ZERO]][0] [4] [1]
// CHECK: %[[ZERO_UB:.*]] = memref.subview %{{.*}}[2] [4] [1]
// CHECK: memref.copy %[[ZERO_GM]], %[[ZERO_UB]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @nonpositive_strides(%src: !tt.ptr<f32>, %broadcast: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %parent = arith.constant 4 : i64
    %out_size = arith.constant 8 : i64
    %neg_stride = arith.constant -2 : i64
    %zero_stride = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %row = arith.constant -2 : i32
    %zero = arith.constant 0 : i32
    %shift = arith.constant 31 : i32
    %base = tt.addptr %src, %shift : !tt.ptr<f32>, i32
    %p = tt.make_tensor_ptr %base, [%parent], [%neg_stride], [%row] {order = array<i32: 0>} : <tensor<8xf32>>
    %b = tt.make_tensor_ptr %broadcast, [%parent], [%zero_stride], [%row] {order = array<i32: 0>} : <tensor<8xf32>>
    %q = tt.make_tensor_ptr %dst, [%out_size], [%one], [%zero] {order = array<i32: 0>} : <tensor<8xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
    %w = tt.load %b {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
    %sum = arith.addf %v, %w : tensor<8xf32>
    tt.store %q, %sum : !tt.ptr<tensor<8xf32>>
    tt.return
  }
}

// -----

// A one-row block view retains the descriptor stride of 4096 both in its
// absolute origin and in the memref layout consumed by DMA.
// The existing size==1 guard must still protect the complete static store.
// CHECK-LABEL: func.func @unit_row_keeps_descriptor_stride(
// CHECK-SAME: %[[BASE:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[ROW:[^ ,]+]]: i32
// CHECK-DAG: %[[ZERO32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[STRIDE:.*]] = arith.constant 4096 : index
// CHECK: %[[CLAMPED32:.*]] = arith.maxsi %[[ROW]], %[[ZERO32]] : i32
// CHECK: %[[CLAMPED:.*]] = arith.index_cast %[[CLAMPED32]] : i32 to index
// CHECK: %[[OFFSET:.*]] = arith.muli %[[CLAMPED]], %[[STRIDE]] : index
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[OFFSET]]], sizes: [1, 4096], strides: [4096, 1]
// CHECK: %[[ACTIVE:.*]] = arith.cmpi eq,
// CHECK: scf.if %[[ACTIVE]] {
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: memref.subview
// CHECK: bufferization.materialize_in_destination {{.*}} in writable %[[VIEW]] : (tensor<1x4096xf32>,
// CHECK: }
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @unit_row_keeps_descriptor_stride(%base: !tt.ptr<f32>, %row: i32, %rows: i64) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %cols = arith.constant 4096 : i64
    %data = arith.constant dense<7.0> : tensor<1x4096xf32>
    %p = tt.make_tensor_ptr %base, [%rows, %cols], [%cols, %one], [%row, %zero] {order = array<i32: 1, 0>} : <tensor<1x4096xf32>>
    tt.store %p, %data {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<1x4096xf32>>
    tt.return
  }
}

// -----

// The plane coordinate still scales by the original pitch. A positive
// memref stride describes the same single plane even for nonpositive pitches.
// CHECK-LABEL: func.func @unit_plane_dma_pitch(
// CHECK-SAME: %[[Q:[^ ,]+]]: memref<?xbf16>
// CHECK-SAME: %[[PITCH:[^ ,]+]]: i64
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[PITCH_IDX:.*]] = arith.index_cast %[[PITCH]] : i64 to index
// CHECK: %[[SAFE_PITCH:.*]] = arith.maxsi %[[PITCH_IDX]], %[[ONE]] : index
// CHECK: memref.reinterpret_cast %[[Q]] to offset: [{{.*}}], sizes: [1, 16, 128], strides: [%[[SAFE_PITCH]], 128, 1]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @unit_plane_dma_pitch(%base: !tt.ptr<bf16>, %pitch: i64, %row: i32, %rows: i64) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %heads = arith.constant 16 : i64
    %width = arith.constant 128 : i64
    %data = arith.constant dense<2.0> : tensor<1x16x128xbf16>
    %p = tt.make_tensor_ptr %base, [%rows, %heads, %width], [%pitch, %width, %one], [%row, %zero, %zero] {order = array<i32: 2, 1, 0>} : <tensor<1x16x128xbf16>>
    tt.store %p, %data {boundaryCheck = array<i32: 0, 1, 2>} : !tt.ptr<tensor<1x16x128xbf16>>
    tt.return
  }
}

// -----

// Clamping the address must not make an originally invalid unit row active.
// In particular, negating INT32_MIN must occur in index, not in i32.
// CHECK-LABEL: func.func @empty_and_extreme_unit_rows
// CHECK-NOT: bufferization.materialize_in_destination
// CHECK-NOT: memref.copy
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @empty_and_extreme_unit_rows(%base: !tt.ptr<f32>) {
    %min = arith.constant -2147483648 : i32
    %max = arith.constant 2147483647 : i32
    %zero = arith.constant 0 : i32
    %empty = arith.constant 0 : i64
    %parent = arith.constant 4 : i64
    %one = arith.constant 1 : i64
    %data = arith.constant dense<7.0> : tensor<1xf32>
    %below = tt.make_tensor_ptr %base, [%parent], [%one], [%min] {order = array<i32: 0>} : <tensor<1xf32>>
    %above = tt.make_tensor_ptr %base, [%parent], [%one], [%max] {order = array<i32: 0>} : <tensor<1xf32>>
    %no_parent = tt.make_tensor_ptr %base, [%empty], [%one], [%zero] {order = array<i32: 0>} : <tensor<1xf32>>
    tt.store %below, %data {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<1xf32>>
    tt.store %above, %data {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<1xf32>>
    tt.store %no_parent, %data {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<1xf32>>
    tt.return
  }
}

// -----

// Direct Advance remains on the original-coordinate path. A nonconstant
// increment keeps it from folding to its MakeTensorPtr input.
// CHECK-LABEL: func.func @advance_keeps_original_origin(
// CHECK-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[DST:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[ROW:[^ ,]+]]: i32
// CHECK: %[[ORIGIN:.*]] = arith.index_cast %[[ROW]] : i32 to index
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [%[[ORIGIN]]], sizes: [8], strides: [1]
// CHECK: %[[GM:.*]] = memref.subview %[[VIEW]][%[[START:.*]]] [%[[COUNT:.*]]] [1]
// CHECK: %[[UB:.*]] = memref.subview %{{.*}}[%[[START]]] [%[[COUNT]]] [1]
// CHECK: memref.copy %[[GM]], %[[UB]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @advance_keeps_original_origin(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %row: i32, %rows: i64) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %p = tt.make_tensor_ptr %src, [%rows], [%one], [%zero] {order = array<i32: 0>} : <tensor<8xf32>>
    %advanced = tt.advance %p, [%row] : <tensor<8xf32>>
    %q = tt.make_tensor_ptr %dst, [%eight], [%one], [%zero] {order = array<i32: 0>} : <tensor<8xf32>>
    %v = tt.load %advanced {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
    tt.store %q, %v : !tt.ptr<tensor<8xf32>>
    tt.return
  }
}

// -----

// Model the completed CFO handoff: a selected scalar coordinate feeds a
// marked descriptor rebuild. The marker keeps this SCF-derived view on the
// original path even though its immediate producer is MakeTensorPtr.
// CHECK-LABEL: func.func @scf_descriptor_rebuild_fallback
// CHECK: %[[COORD:.*]] = {{arith.select|scf.if}}
// CHECK: %[[ORIGIN:.*]] = arith.index_cast %[[COORD]] : i32 to index
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast {{.*}} to offset: [%[[ORIGIN]]], sizes: [8], strides: [1]
// CHECK: %[[GM:.*]] = memref.subview %[[VIEW]][%[[START:.*]]] [%[[COUNT:.*]]] [1]
// CHECK: %[[UB:.*]] = memref.subview %{{.*}}[%[[START]]] [%[[COUNT]]] [1]
// CHECK: memref.copy %[[GM]], %[[UB]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @scf_descriptor_rebuild_fallback(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %cond: i1, %left: i32, %right: i32, %rows: i64) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %row = scf.if %cond -> (i32) {
      scf.yield %left : i32
    } else {
      scf.yield %right : i32
    }
    %p = tt.make_tensor_ptr %src, [%rows], [%one], [%row] {order = array<i32: 0>, PointerDescriptorRebuild} : <tensor<8xf32>>
    %q = tt.make_tensor_ptr %dst, [%eight], [%one], [%zero] {order = array<i32: 0>} : <tensor<8xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
    tt.store %q, %v : !tt.ptr<tensor<8xf32>>
    tt.return
  }
}

// -----

// A2 retains the existing view origin and shifts both sides of the copy.
// CHECK-LABEL: func.func @non950_keeps_original_slices
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast {{.*}}sizes: [8], strides: [1]
// CHECK: %[[GM:.*]] = memref.subview %[[VIEW]][2] [4] [1]
// CHECK: %[[UB:.*]] = memref.subview %{{.*}}[2] [4] [1]
// CHECK: memref.copy %[[GM]], %[[UB]]
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @non950_keeps_original_slices(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %parent = arith.constant 4 : i64
    %eight = arith.constant 8 : i64
    %one = arith.constant 1 : i64
    %row = arith.constant -2 : i32
    %zero = arith.constant 0 : i32
    %p = tt.make_tensor_ptr %src, [%parent], [%one], [%row] {order = array<i32: 0>} : <tensor<8xf32>>
    %q = tt.make_tensor_ptr %dst, [%eight], [%one], [%zero] {order = array<i32: 0>} : <tensor<8xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
    tt.store %q, %v : !tt.ptr<tensor<8xf32>>
    tt.return
  }
}

// -----

// An unchecked block access may intentionally address before the descriptor
// base. It must use the raw coordinate and copy the complete tile.
// CHECK-LABEL: func.func @no_boundary_keeps_raw_coordinate(
// CHECK-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[DST:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[ROW:[^ ,]+]]: i32
// CHECK: %[[ORIGIN:.*]] = arith.index_cast %[[ROW]] : i32 to index
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [%[[ORIGIN]]], sizes: [8], strides: [1]
// CHECK-NOT: memref.subview
// CHECK: memref.copy %[[VIEW]],
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @no_boundary_keeps_raw_coordinate(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %row: i32, %rows: i64) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %p = tt.make_tensor_ptr %src, [%rows], [%one], [%row] {order = array<i32: 0>} : <tensor<8xf32>>
    %q = tt.make_tensor_ptr %dst, [%eight], [%one], [%zero] {order = array<i32: 0>} : <tensor<8xf32>>
    %v = tt.load %p : !tt.ptr<tensor<8xf32>>
    tt.store %q, %v : !tt.ptr<tensor<8xf32>>
    tt.return
  }
}
