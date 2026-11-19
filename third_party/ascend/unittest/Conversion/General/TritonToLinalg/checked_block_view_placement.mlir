// RUN: triton-opt --triton-to-linalg --verify-each --split-input-file %s | FileCheck %s --check-prefixes=CHECK,950
// RUN: sed 's/Ascend950PR_9579/Ascend910B2/g' %s | triton-opt --triton-to-linalg --verify-each --split-input-file | FileCheck %s --check-prefixes=CHECK,A2

// Checked 950 block views and their clamped address arithmetic follow their
// producers. Both descriptor inputs are ready before the observer. The memory
// operations still execute in observer, lhs, rhs order. A2 keeps use placement.
// CHECK-LABEL: func.func @checked_views_follow_producers(
// CHECK-SAME: %[[LHS:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[RHS:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[OBSERVER:[^ ,]+]]: memref<?xf32>
// 950: %[[LEFT:.*]] = memref.reinterpret_cast %[[LHS]] to offset: [{{[^]]+}}], sizes: [8], strides: [1]
// 950: %[[RIGHT:.*]] = memref.reinterpret_cast %[[RHS]] to offset: [{{[^]]+}}], sizes: [8], strides: [1]
// 950-NOT: memref.copy
// A2-NOT: memref.reinterpret_cast %[[LHS]]
// A2-NOT: memref.reinterpret_cast %[[RHS]]
// CHECK: %[[OBSERVED:.*]] = memref.load %[[OBSERVER]]
// A2: %[[LEFT:.*]] = memref.reinterpret_cast %[[LHS]] to offset: [{{[^]]+}}], sizes: [8], strides: [1]
// CHECK: %[[LEFT_GM:.*]] = memref.subview %[[LEFT]]
// CHECK: memref.copy %[[LEFT_GM]],
// A2: %[[RIGHT:.*]] = memref.reinterpret_cast %[[RHS]] to offset: [{{[^]]+}}], sizes: [8], strides: [1]
// CHECK: %[[RIGHT_GM:.*]] = memref.subview %[[RIGHT]]
// CHECK: memref.copy %[[RIGHT_GM]],
// CHECK: return %[[OBSERVED]] : f32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @checked_views_follow_producers(%lhs: !tt.ptr<f32>, %rhs: !tt.ptr<f32>, %observer: !tt.ptr<f32>, %out: !tt.ptr<f32>, %row: i32, %rows: i64) -> f32 {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    %eight = arith.constant 8 : i64
    %left = tt.make_tensor_ptr %lhs, [%rows], [%one], [%row] {order = array<i32: 0>} : <tensor<8xf32>>
    %right = tt.make_tensor_ptr %rhs, [%rows], [%one], [%row] {order = array<i32: 0>} : <tensor<8xf32>>
    %observed = tt.load %observer : !tt.ptr<f32>
    %a = tt.load %left {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
    %b = tt.load %right {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
    %sum = arith.addf %a, %b : tensor<8xf32>
    %dest = tt.make_tensor_ptr %out, [%eight], [%one], [%zero] {order = array<i32: 0>} : <tensor<8xf32>>
    tt.store %dest, %sum : !tt.ptr<tensor<8xf32>>
    tt.return %observed : f32
  }
}

// -----

// The pointer is defined outside this branch. Its checked view must remain
// inside the branch, after the observer, even though the address is available.
// CHECK-LABEL: func.func @outer_checked_view_stays_in_branch(
// CHECK-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// CHECK-SAME: %[[OBSERVER:[^ ,]+]]: memref<?xf32>
// CHECK-NOT: memref.reinterpret_cast %[[SRC]]
// CHECK: scf.if
// CHECK-NOT: memref.reinterpret_cast %[[SRC]]
// CHECK: %[[OBSERVED:.*]] = memref.load %[[OBSERVER]]
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[SRC]]
// CHECK: %[[GM:.*]] = memref.subview %[[VIEW]]
// CHECK: memref.copy %[[GM]],
// CHECK: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @outer_checked_view_stays_in_branch(%src: !tt.ptr<f32>, %observer: !tt.ptr<f32>, %out: !tt.ptr<f32>, %row: i32, %rows: i64, %cond: i1) {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    %eight = arith.constant 8 : i64
    %ptr = tt.make_tensor_ptr %src, [%rows], [%one], [%row] {order = array<i32: 0>} : <tensor<8xf32>>
    scf.if %cond {
      %observed = tt.load %observer : !tt.ptr<f32>
      %value = tt.load %ptr {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
      %observed_tile = tt.splat %observed : f32 -> tensor<8xf32>
      %sum = arith.addf %value, %observed_tile : tensor<8xf32>
      %dest = tt.make_tensor_ptr %out, [%eight], [%one], [%zero] {order = array<i32: 0>} : <tensor<8xf32>>
      tt.store %dest, %sum : !tt.ptr<tensor<8xf32>>
    }
    tt.return
  }
}

// -----

// Two accesses share a producer but check different axes, so their origins
// differ. Keep both view constructions in access order at the producer; do
// not reuse a view with another access's boundary-check coordinates.
// With base=32, row=-1 and col=-2, the origins are 30 and 16 respectively.
// 950-LABEL: func.func @shared_producer_distinct_checks(
// 950-SAME: %[[SRC:[^ ,]+]]: memref<?xf32>
// 950-SAME: %[[OBSERVER:[^ ,]+]]: memref<?xf32>
// 950: %[[ROWS_VIEW:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [30]
// 950: %[[COLS_VIEW:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [16]
// 950: %[[OBSERVED:.*]] = memref.load %[[OBSERVER]]
// 950: %[[ROWS_GM:.*]] = memref.subview %[[ROWS_VIEW]]
// 950: memref.copy %[[ROWS_GM]],
// 950: %[[COLS_GM:.*]] = memref.subview %[[COLS_VIEW]]
// 950: memref.copy %[[COLS_GM]],
// 950: return %[[OBSERVED]] : f32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @shared_producer_distinct_checks(%src: !tt.ptr<f32>, %observer: !tt.ptr<f32>, %out: !tt.ptr<f32>) -> f32 {
    %one = arith.constant 1 : i64
    %sixteen = arith.constant 16 : i64
    %eight = arith.constant 8 : i64
    %four = arith.constant 4 : i64
    %zero = arith.constant 0 : i32
    %row = arith.constant -1 : i32
    %col = arith.constant -2 : i32
    %shift = arith.constant 32 : i32
    %base = tt.addptr %src, %shift : !tt.ptr<f32>, i32
    %ptr = tt.make_tensor_ptr %base, [%eight, %sixteen], [%sixteen, %one], [%row, %col] {order = array<i32: 1, 0>} : <tensor<8x4xf32>>
    %observed = tt.load %observer : !tt.ptr<f32>
    %a = tt.load %ptr {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8x4xf32>>
    %b = tt.load %ptr {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<8x4xf32>>
    %sum = arith.addf %a, %b : tensor<8x4xf32>
    %dest = tt.make_tensor_ptr %out, [%eight, %four], [%four, %one], [%zero, %zero] {order = array<i32: 1, 0>} : <tensor<8x4xf32>>
    tt.store %dest, %sum : !tt.ptr<tensor<8x4xf32>>
    tt.return %observed : f32
  }
}
