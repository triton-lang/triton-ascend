// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @unit_row_guard
// CHECK: %[[ACTIVE:.*]] = arith.cmpi eq,
// CHECK: scf.if %[[ACTIVE]] {
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: memref.subview
// CHECK: bufferization.materialize_in_destination {{.*}} : (tensor<1x16xf32>, memref<1x16xf32, {{.*}}>)
// CHECK: }
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @unit_row_guard(%base: !tt.ptr<f32>, %rows: i64, %row: i32, %cols: i64, %col: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %data = arith.constant dense<7.0> : tensor<1x16xf32>
    %ptr = tt.make_tensor_ptr %base, [%rows, %c16_i64], [%c16_i64, %c1_i64], [%row, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x16xf32>>
    tt.store %ptr, %data {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<1x16xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @negative_unit_row
// CHECK-NOT: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @negative_unit_row(%base: !tt.ptr<f32>, %rows: i64, %row: i32, %cols: i64, %col: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %data = arith.constant dense<7.0> : tensor<1x16xf32>
    %ptr = tt.make_tensor_ptr %base, [%c16_i64, %c16_i64], [%c16_i64, %c1_i64], [%cm1_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x16xf32>>
    tt.store %ptr, %data {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<1x16xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @empty_parent
// CHECK-NOT: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @empty_parent(%base: !tt.ptr<f32>, %rows: i64, %row: i32, %cols: i64, %col: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %data = arith.constant dense<7.0> : tensor<1x16xf32>
    %ptr = tt.make_tensor_ptr %base, [%c0_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x16xf32>>
    tt.store %ptr, %data {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<1x16xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @row_at_end
// CHECK-NOT: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @row_at_end(%base: !tt.ptr<f32>, %rows: i64, %row: i32, %cols: i64, %col: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %data = arith.constant dense<7.0> : tensor<1x16xf32>
    %ptr = tt.make_tensor_ptr %base, [%c16_i64, %c16_i64], [%c16_i64, %c1_i64], [%c16_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x16xf32>>
    tt.store %ptr, %data {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<1x16xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @nonunit_row
// CHECK-NOT: scf.if
// CHECK: tensor.extract_slice
// CHECK: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @nonunit_row(%base: !tt.ptr<f32>, %rows: i64, %row: i32, %cols: i64, %col: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %data = arith.constant dense<7.0> : tensor<2x16xf32>
    %ptr = tt.make_tensor_ptr %base, [%rows, %c16_i64], [%c16_i64, %c1_i64], [%row, %c0_i32] {order = array<i32: 1, 0>} : <tensor<2x16xf32>>
    tt.store %ptr, %data {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<2x16xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @partial_columns
// CHECK-NOT: scf.if
// CHECK: tensor.extract_slice
// CHECK: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @partial_columns(%base: !tt.ptr<f32>, %rows: i64, %row: i32, %cols: i64, %col: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %data = arith.constant dense<7.0> : tensor<1x16xf32>
    %ptr = tt.make_tensor_ptr %base, [%rows, %cols], [%c16_i64, %c1_i64], [%row, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x16xf32>>
    tt.store %ptr, %data {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<1x16xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @unchanged_910
// CHECK-NOT: scf.if
// CHECK: tensor.extract_slice
// CHECK: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @unchanged_910(%base: !tt.ptr<f32>, %rows: i64, %row: i32, %cols: i64, %col: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %data = arith.constant dense<7.0> : tensor<1x16xf32>
    %ptr = tt.make_tensor_ptr %base, [%rows, %c16_i64], [%c16_i64, %c1_i64], [%row, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x16xf32>>
    tt.store %ptr, %data {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<1x16xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @two_unit_axes
// CHECK: %[[ACTIVE:.*]] = arith.andi
// CHECK: scf.if %[[ACTIVE]] {
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: memref.subview
// CHECK: bufferization.materialize_in_destination {{.*}} : (tensor<1x1xf32>, memref<1x1xf32, {{.*}}>)
// CHECK: }
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @two_unit_axes(%base: !tt.ptr<f32>, %rows: i64, %row: i32, %cols: i64, %col: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %data = arith.constant dense<7.0> : tensor<1x1xf32>
    %ptr = tt.make_tensor_ptr %base, [%rows, %cols], [%c16_i64, %c1_i64], [%row, %col] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
    tt.store %ptr, %data {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<1x1xf32>>
    tt.return
  }
}

// -----

// A pair of interleaved stores must retain their independent bounds checks.
// CHECK-LABEL: func.func @independent_interleaved_guards
// CHECK: scf.if %{{.*}} {
// CHECK-NEXT: bufferization.materialize_in_destination {{.*}} : (tensor<1x16xf32>, memref<1x16xf32, {{.*}}>)
// CHECK-NEXT: }
// CHECK: scf.if %{{.*}} {
// CHECK-NEXT: bufferization.materialize_in_destination {{.*}} : (tensor<1x16xf32>, memref<1x16xf32, {{.*}}>)
// CHECK-NEXT: }
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @independent_interleaved_guards(%base: !tt.ptr<f32>, %rows_even: i64, %rows_odd: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %c32_i64 = arith.constant 32 : i64
    %even = arith.constant dense<7.0> : tensor<1x16xf32>
    %odd = arith.constant dense<9.0> : tensor<1x16xf32>
    %odd_base = tt.addptr %base, %c1_i32 : !tt.ptr<f32>, i32
    %even_ptr = tt.make_tensor_ptr %base, [%rows_even, %c16_i64], [%c32_i64, %c2_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x16xf32>>
    %odd_ptr = tt.make_tensor_ptr %odd_base, [%rows_odd, %c16_i64], [%c32_i64, %c2_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x16xf32>>
    tt.store %even_ptr, %even {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<1x16xf32>>
    tt.store %odd_ptr, %odd {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<1x16xf32>>
    tt.return
  }
}
