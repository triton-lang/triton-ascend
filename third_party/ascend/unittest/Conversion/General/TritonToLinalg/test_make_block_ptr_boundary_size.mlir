// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg --split-input-file %s 2>&1 | FileCheck %s --check-prefix=WARN

// Block-pointer boundaries use the original per-axis coordinates and parent
// shapes. They are independent of physical stride order and remain available
// for broadcast axes whose stride is zero.
// WARN-NOT: error:
// WARN-NOT: warning:
// WARN-LABEL: func.func @boundary_size_zero_stride_axis

// Primary case: axis 0 (stride 64) is *not* checked, axis 1 is.  The block
// starts at (16, 36) of a (64, 60) tensor: the flat offset 16*64 + 36 = 1060
// must be reduced by axis 0 first, leaving 36 for axis 1 -> boundary
// 60 - 36 = 24.  Before the fix the un-reduced offset 1060 was used,
// clamping the axis-1 boundary to 0 ([16, 0]) and dropping the whole load.

// CHECK-LABEL: func.func @boundary_size_unchecked_axis
// CHECK: memref.subview {{.*}}[0, 0] [16, 24] [1, 1]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @boundary_size_unchecked_axis(
      %base_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %c36_i32 = arith.constant 36 : i32
    %c1_i64 = arith.constant 1 : i64
    %c60_i64 = arith.constant 60 : i64
    %c64_i64 = arith.constant 64 : i64
    %load_ptr = tt.make_tensor_ptr %base_ptr, [%c64_i64, %c60_i64], [%c64_i64, %c1_i64], [%c16_i32, %c36_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>>
    %data = tt.load %load_ptr {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<16x32xf32>>
    %store_ptr = tt.make_tensor_ptr %out_ptr, [%c64_i64, %c60_i64], [%c64_i64, %c1_i64], [%c16_i32, %c36_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>>
    tt.store %store_ptr, %data {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<16x32xf32>>
    tt.return
  }
}

// -----
// Incidental case: axis 0 is a broadcast axis (stride 0) and is
// boundary-checked. Its coordinate is zero in a parent of size 64, so
// all 16 lanes are valid; the axis-1 boundary is 60 - 36 = 24.

// CHECK-LABEL: func.func @boundary_size_zero_stride_axis
// CHECK: memref.subview {{.*}}[0, 0] [16, 24] [1, 1]
// CHECK: memref.copy
// CHECK: memref.subview {{.*}}[0, 0] [16, 24] [1, 1]
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @boundary_size_zero_stride_axis(
      %base_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c36_i32 = arith.constant 36 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c60_i64 = arith.constant 60 : i64
    %c64_i64 = arith.constant 64 : i64
    %load_ptr = tt.make_tensor_ptr %base_ptr, [%c64_i64, %c60_i64], [%c0_i64, %c1_i64], [%c0_i32, %c36_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>>
    %data = tt.load %load_ptr {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x32xf32>>
    %store_ptr = tt.make_tensor_ptr %out_ptr, [%c64_i64, %c60_i64], [%c64_i64, %c1_i64], [%c0_i32, %c36_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>>
    tt.store %store_ptr, %data {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<16x32xf32>>
    tt.return
  }
}

// -----
// Permuted layout: logical axes [0, 1, 2] have physical strides [1, 800, 20].
// The block starts at logical offset [7, 18, 32] in shape [20, 30, 40].  Its
// flat offset is 7 + 18*800 + 32*20 = 15047.  Decomposing in physical order
// [1, 2, 0] recovers [18, 32, 7], which maps back to logical boundary sizes
// [1, min(16, 30-18), min(64, 40-32)] = [1, 12, 8].

// CHECK-LABEL: func.func @boundary_size_permuted_strides
// CHECK: memref.subview {{.*}}[0, 0, 0] [1, 12, 8] [1, 1, 1]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @boundary_size_permuted_strides(
      %base_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c7_i32 = arith.constant 7 : i32
    %c18_i32 = arith.constant 18 : i32
    %c32_i32 = arith.constant 32 : i32
    %c20_i64 = arith.constant 20 : i64
    %c30_i64 = arith.constant 30 : i64
    %c40_i64 = arith.constant 40 : i64
    %c800_i64 = arith.constant 800 : i64
    %c1200_i64 = arith.constant 1200 : i64
    %load_ptr = tt.make_tensor_ptr %base_ptr, [%c20_i64, %c30_i64, %c40_i64], [%c1_i64, %c800_i64, %c20_i64], [%c7_i32, %c18_i32, %c32_i32] {order = array<i32: 2, 1, 0>} : <tensor<1x16x64xf32>>
    %data = tt.load %load_ptr {boundaryCheck = array<i32: 0, 1, 2>} : !tt.ptr<tensor<1x16x64xf32>>
    %store_ptr = tt.make_tensor_ptr %out_ptr, [%c20_i64, %c30_i64, %c40_i64], [%c1200_i64, %c40_i64, %c1_i64], [%c7_i32, %c18_i32, %c32_i32] {order = array<i32: 2, 1, 0>} : <tensor<1x16x64xf32>>
    tt.store %store_ptr, %data {boundaryCheck = array<i32: 0, 1, 2>} : !tt.ptr<tensor<1x16x64xf32>>
    tt.return
  }
}

// -----

// A negative starting coordinate shifts both the source and destination slice.
// For parent size 5 and coordinate -2, lanes [2, 7) of the eight-lane tile are
// valid. Padding initializes the two leading and one trailing invalid lanes.
// CHECK-LABEL: func.func @negative_block_coordinate
// CHECK: linalg.fill
// CHECK: memref.subview {{.*}}[2] [5] [1]
// CHECK: memref.subview {{.*}}[2] [5] [1]
// CHECK: memref.copy
// CHECK: tensor.extract_slice {{.*}}[2] [5] [1]
// CHECK: memref.subview {{.*}}[2] [5] [1]
// CHECK: bufferization.materialize_in_destination
module {
  tt.func public @negative_block_coordinate(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %parent = arith.constant 5 : i64
    %stride = arith.constant 1 : i64
    %start = arith.constant -2 : i32
    %p = tt.make_tensor_ptr %src, [%parent], [%stride], [%start] {order = array<i32: 0>} : <tensor<8xf32>>
    %q = tt.make_tensor_ptr %dst, [%parent], [%stride], [%start] {order = array<i32: 0>} : <tensor<8xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
    tt.store %q, %v {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<8xf32>>
    tt.return
  }
}

// -----

// A zero physical stride does not remove the logical boundary: starting at
// coordinate one in a parent of size three leaves exactly two valid lanes.
// CHECK-LABEL: func.func @partial_zero_stride_boundary
// CHECK: memref.reinterpret_cast {{.*}}sizes: [8], strides: [0]
// CHECK: linalg.fill
// CHECK: memref.subview {{.*}}[0] [2] [1]
// CHECK: memref.copy
module {
  tt.func public @partial_zero_stride_boundary(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %parent = arith.constant 3 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i32
    %p = tt.make_tensor_ptr %src, [%parent], [%zero], [%one] {order = array<i32: 0>} : <tensor<8xf32>>
    %v = tt.load %p {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %base = tt.splat %dst : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %q = tt.addptr %base, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %q, %v : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
