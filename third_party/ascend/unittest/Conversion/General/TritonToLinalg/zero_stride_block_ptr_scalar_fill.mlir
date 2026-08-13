// RUN: triton-opt "--triton-to-linalg=global-kernel=false named-ops=True" --split-input-file %s | FileCheck %s

// A direct all-zero-stride MTP load is a scalar broadcast. Boundary sizes must
// be calculated from the original logical shape/offset, then the valid region
// is filled from base[0]. Do not materialize a zero-stride memref layout.
// CHECK-LABEL: func.func @all_zero_stride_dynamic
// CHECK-NOT: strided<[0
// CHECK: memref.alloc() : memref<4x4xf32>
// CHECK: arith.subi
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: memref.subview
// CHECK: arith.cmpi sgt
// CHECK: scf.if
// CHECK: memref.load
// CHECK: linalg.fill
// CHECK: bufferization.to_tensor
// CHECK-NOT: strided<[0
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @all_zero_stride_dynamic(
      %src: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %shape_m: i32, %shape_n: i32, %offset_m: i32, %offset_n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %shape_m_i64 = arith.extsi %shape_m : i32 to i64
    %shape_n_i64 = arith.extsi %shape_n : i32 to i64
    %src_block = tt.make_tensor_ptr %src, [%shape_m_i64, %shape_n_i64],
        [%c0_i64, %c0_i64], [%offset_m, %offset_n]
        {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
    %value = tt.load %src_block {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32}
        : !tt.ptr<tensor<4x4xf32>>
    %dst_block = tt.make_tensor_ptr %dst, [%c4_i64, %c4_i64],
        [%c4_i64, %c1_i64], [%c0_i32, %c0_i32]
        {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
    tt.store %dst_block, %value : !tt.ptr<tensor<4x4xf32>>
    tt.return
  }
}

// -----

// Negative logical offsets leave a padded prefix. The scalar source access
// remains guarded by a non-empty valid region.
// CHECK-LABEL: func.func @all_zero_stride_negative_offset
// CHECK-NOT: strided<[0
// CHECK: memref.subview
// CHECK: memref.load
// CHECK: linalg.fill
// CHECK-NOT: strided<[0
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @all_zero_stride_negative_offset(
      %src: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cneg2_i32 = arith.constant -2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %src_block = tt.make_tensor_ptr %src, [%c6_i64, %c5_i64],
        [%c0_i64, %c0_i64], [%cneg2_i32, %c0_i32]
        {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
    %value = tt.load %src_block {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32}
        : !tt.ptr<tensor<4x4xf32>>
    %dst_block = tt.make_tensor_ptr %dst, [%c4_i64, %c4_i64],
        [%c4_i64, %c1_i64], [%c0_i32, %c0_i32]
        {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
    tt.store %dst_block, %value : !tt.ptr<tensor<4x4xf32>>
    tt.return
  }
}

// -----

// A short logical shape with a negative offset has a valid suffix that starts
// inside the tile. The final valid subview must keep that suffix rather than
// subtracting the left padding from an already-clipped right extent.
// CHECK-LABEL: func.func @all_zero_stride_negative_offset_short_shape
// CHECK: memref.subview %alloc[2, 0] [1, 1] [1, 1]
// CHECK: linalg.fill ins(%cst : f32) outs(%alloc : memref<4x4xf32>)
// CHECK: memref.load
// CHECK: linalg.fill
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @all_zero_stride_negative_offset_short_shape(
      %src: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cneg2_i32 = arith.constant -2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %src_block = tt.make_tensor_ptr %src, [%c1_i64, %c1_i64],
        [%c0_i64, %c0_i64], [%cneg2_i32, %c0_i32]
        {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
    %value = tt.load %src_block {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32}
        : !tt.ptr<tensor<4x4xf32>>
    %dst_block = tt.make_tensor_ptr %dst, [%c4_i64, %c4_i64],
        [%c4_i64, %c1_i64], [%c0_i32, %c0_i32]
        {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
    tt.store %dst_block, %value : !tt.ptr<tensor<4x4xf32>>
    tt.return
  }
}

// -----

// All-nonzero MTP loads must retain the existing physical-offset
// reconstruction and copy lowering.
// CHECK-LABEL: func.func @nonzero_stride_dynamic
// CHECK: arith.divsi
// CHECK: arith.remsi
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @nonzero_stride_dynamic(
      %src: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %shape_m: i32, %shape_n: i32, %offset_m: i32, %offset_n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %shape_m_i64 = arith.extsi %shape_m : i32 to i64
    %shape_n_i64 = arith.extsi %shape_n : i32 to i64
    %src_block = tt.make_tensor_ptr %src, [%shape_m_i64, %shape_n_i64],
        [%c8_i64, %c1_i64], [%offset_m, %offset_n]
        {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
    %value = tt.load %src_block {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32}
        : !tt.ptr<tensor<4x4xf32>>
    %dst_block = tt.make_tensor_ptr %dst, [%c4_i64, %c4_i64],
        [%c4_i64, %c1_i64], [%c0_i32, %c0_i32]
        {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
    tt.store %dst_block, %value : !tt.ptr<tensor<4x4xf32>>
    tt.return
  }
}
