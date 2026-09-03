// RUN: triton-opt "--triton-to-linalg=global-kernel=false named-ops=True" --split-input-file %s | FileCheck %s

// A direct MTP load with zero strides must derive its boundary from the MTP
// logical shape/offset. The converted physical offset is always zero and
// cannot be divided by a zero physical stride to recover that information.
// CHECK-LABEL: func.func @zero_stride_dynamic
// CHECK: memref.reinterpret_cast {{.*}}strides: [0, 0]
// CHECK: arith.subi
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @zero_stride_dynamic(
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

// CHECK-LABEL: func.func @mixed_zero_stride_dynamic
// CHECK: memref.reinterpret_cast {{.*}}strides: [0, 1]
// CHECK: arith.subi
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @mixed_zero_stride_dynamic(
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
        [%c0_i64, %c1_i64], [%offset_m, %offset_n]
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

// CHECK-LABEL: func.func @zero_stride_negative_offset
// CHECK: memref.reinterpret_cast {{.*}}strides: [0, 0]
// CHECK: memref.subview {{.*}}[2, 0]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @zero_stride_negative_offset(
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

// A direct MTP with statically nonzero strides must keep the legacy physical
// offset reconstruction path.
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
