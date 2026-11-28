// RUN: triton-opt '--triton-to-structured=compile-on-910-95=true' --verify-each --split-input-file %s | FileCheck %s

// Keep the input multiplication width, including narrow integer wraparound.
// CHECK-LABEL: tt.func public @i8_mul_before_extend(
// CHECK: %[[SCALED:.*]] = arith.muli {{.*}} : tensor<1x16xi8>
// CHECK: arith.extsi %[[SCALED]] : tensor<1x16xi8> to tensor<1x16xi32>
// CHECK: %[[LOADED:.*]] = tt.load {{.*}} {ImplicitPermuteHandled} : tensor<16x32x!tt.ptr<f32>>
// CHECK: tt.trans %[[LOADED]] {implicit_load_permute, order = array<i32: 1, 0>}
module {
  tt.func public @i8_mul_before_extend(%base: !tt.ptr<f32>, %idx: !tt.ptr<i8>) -> tensor<32x16xf32> {
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %d = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %idxvec = tt.splat %idx : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>>
    %idxptr = tt.addptr %idxvec, %n : tensor<16x!tt.ptr<i8>>, tensor<16xi32>
    %t = tt.load %idxptr : tensor<16x!tt.ptr<i8>>
    %trow = tt.expand_dims %t {axis = 0 : i32} : tensor<16xi8> -> tensor<1x16xi8>
    %c = arith.constant dense<64> : tensor<1x16xi8>
    %offset = arith.muli %trow, %c : tensor<1x16xi8>
    %basevec = tt.splat %base : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %rowptr = tt.addptr %basevec, %offset : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi8>
    %rowptrs = tt.broadcast %rowptr : tensor<1x16x!tt.ptr<f32>> -> tensor<32x16x!tt.ptr<f32>>
    %dcol = tt.expand_dims %d {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %ds = tt.broadcast %dcol : tensor<32x1xi32> -> tensor<32x16xi32>
    %p = tt.addptr %rowptrs, %ds : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
    %val = tt.load %p : tensor<32x16x!tt.ptr<f32>>
    tt.return %val : tensor<32x16xf32>
  }
}

// -----

// CHECK-LABEL: tt.func public @i16_combined_offset_mul(
// CHECK-DAG: %[[SCALED1:.*]] = arith.muli {{.*}} : tensor<1x16xi16>
// CHECK-DAG: %[[SCALED2:.*]] = arith.muli {{.*}} : tensor<1x16xi16>
// CHECK-DAG: arith.extsi %[[SCALED1]] : tensor<1x16xi16> to tensor<1x16xi32>
// CHECK-DAG: arith.extsi %[[SCALED2]] : tensor<1x16xi16> to tensor<1x16xi32>
// CHECK: %[[LOADED:.*]] = tt.load {{.*}} {ImplicitPermuteHandled} : tensor<16x32x!tt.ptr<f32>>
// CHECK: tt.trans %[[LOADED]] {implicit_load_permute, order = array<i32: 1, 0>}
module {
  tt.func public @i16_combined_offset_mul(%base: !tt.ptr<f32>, %idx: !tt.ptr<i16>, %idx2: !tt.ptr<i16>) -> tensor<32x16xf32> {
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %d = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %idxvec = tt.splat %idx : !tt.ptr<i16> -> tensor<16x!tt.ptr<i16>>
    %idxptr = tt.addptr %idxvec, %n : tensor<16x!tt.ptr<i16>>, tensor<16xi32>
    %t = tt.load %idxptr : tensor<16x!tt.ptr<i16>>
    %trow = tt.expand_dims %t {axis = 0 : i32} : tensor<16xi16> -> tensor<1x16xi16>
    %idxvec2 = tt.splat %idx2 : !tt.ptr<i16> -> tensor<16x!tt.ptr<i16>>
    %idxptr2 = tt.addptr %idxvec2, %n : tensor<16x!tt.ptr<i16>>, tensor<16xi32>
    %t2 = tt.load %idxptr2 : tensor<16x!tt.ptr<i16>>
    %trow2 = tt.expand_dims %t2 {axis = 0 : i32} : tensor<16xi16> -> tensor<1x16xi16>
    %sum = arith.addi %trow, %trow2 : tensor<1x16xi16>
    %c = arith.constant dense<64> : tensor<1x16xi16>
    %offset = arith.muli %sum, %c : tensor<1x16xi16>
    %basevec = tt.splat %base : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %rowptr = tt.addptr %basevec, %offset : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi16>
    %rowptrs = tt.broadcast %rowptr : tensor<1x16x!tt.ptr<f32>> -> tensor<32x16x!tt.ptr<f32>>
    %dcol = tt.expand_dims %d {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %ds = tt.broadcast %dcol : tensor<32x1xi32> -> tensor<32x16xi32>
    %p = tt.addptr %rowptrs, %ds : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
    %val = tt.load %p : tensor<32x16x!tt.ptr<f32>>
    tt.return %val : tensor<32x16xf32>
  }
}

// -----

// Keep an explicit extension before the multiplication.
// CHECK-LABEL: tt.func public @i8_extend_before_mul(
// CHECK: %[[EXTENDED:.*]] = arith.extsi {{.*}} : tensor<1x16xi8> to tensor<1x16xi32>
// CHECK: arith.muli %[[EXTENDED]], {{.*}} : tensor<1x16xi32>
// CHECK: %[[LOADED:.*]] = tt.load {{.*}} {ImplicitPermuteHandled} : tensor<16x32x!tt.ptr<f32>>
// CHECK: tt.trans %[[LOADED]] {implicit_load_permute, order = array<i32: 1, 0>}
module {
  tt.func public @i8_extend_before_mul(%base: !tt.ptr<f32>, %idx: !tt.ptr<i8>) -> tensor<32x16xf32> {
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %d = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %idxvec = tt.splat %idx : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>>
    %idxptr = tt.addptr %idxvec, %n : tensor<16x!tt.ptr<i8>>, tensor<16xi32>
    %t = tt.load %idxptr : tensor<16x!tt.ptr<i8>>
    %trow = tt.expand_dims %t {axis = 0 : i32} : tensor<16xi8> -> tensor<1x16xi8>
    %wide = arith.extsi %trow : tensor<1x16xi8> to tensor<1x16xi32>
    %c = arith.constant dense<64> : tensor<1x16xi32>
    %offset = arith.muli %wide, %c : tensor<1x16xi32>
    %basevec = tt.splat %base : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %rowptr = tt.addptr %basevec, %offset : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %rowptrs = tt.broadcast %rowptr : tensor<1x16x!tt.ptr<f32>> -> tensor<32x16x!tt.ptr<f32>>
    %dcol = tt.expand_dims %d {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %ds = tt.broadcast %dcol : tensor<32x1xi32> -> tensor<32x16xi32>
    %p = tt.addptr %rowptrs, %ds : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
    %val = tt.load %p : tensor<32x16x!tt.ptr<f32>>
    tt.return %val : tensor<32x16xf32>
  }
}

// -----

// Extend both dynamic terms of the sum before rebuilding the i32 products.
// CHECK-LABEL: tt.func public @i8_combined_offset_extend_before_mul(
// CHECK-DAG: %[[WIDE1:[0-9]+]] = arith.extsi {{.*}} : tensor<1x16xi8> to tensor<1x16xi32>
// CHECK-DAG: %[[WIDE2:[0-9]+]] = arith.extsi {{.*}} : tensor<1x16xi8> to tensor<1x16xi32>
// CHECK-DAG: arith.muli %[[WIDE1]], {{.*}} : tensor<1x16xi32>
// CHECK-DAG: arith.muli %[[WIDE2]], {{.*}} : tensor<1x16xi32>
// CHECK: %[[LOADED:.*]] = tt.load {{.*}} {ImplicitPermuteHandled} : tensor<16x32x!tt.ptr<f32>>
// CHECK: tt.trans %[[LOADED]] {implicit_load_permute, order = array<i32: 1, 0>}
module {
  tt.func public @i8_combined_offset_extend_before_mul(%base: !tt.ptr<f32>, %idx1: !tt.ptr<i8>, %idx2: !tt.ptr<i8>) -> tensor<32x16xf32> {
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %d = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %idxvec1 = tt.splat %idx1 : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>>
    %idxptr1 = tt.addptr %idxvec1, %n : tensor<16x!tt.ptr<i8>>, tensor<16xi32>
    %t1 = tt.load %idxptr1 : tensor<16x!tt.ptr<i8>>
    %trow1 = tt.expand_dims %t1 {axis = 0 : i32} : tensor<16xi8> -> tensor<1x16xi8>
    %idxvec2 = tt.splat %idx2 : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>>
    %idxptr2 = tt.addptr %idxvec2, %n : tensor<16x!tt.ptr<i8>>, tensor<16xi32>
    %t2 = tt.load %idxptr2 : tensor<16x!tt.ptr<i8>>
    %trow2 = tt.expand_dims %t2 {axis = 0 : i32} : tensor<16xi8> -> tensor<1x16xi8>
    %sum = arith.addi %trow1, %trow2 : tensor<1x16xi8>
    %wide = arith.extsi %sum : tensor<1x16xi8> to tensor<1x16xi32>
    %c = arith.constant dense<64> : tensor<1x16xi32>
    %offset = arith.muli %wide, %c : tensor<1x16xi32>
    %basevec = tt.splat %base : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %rowptr = tt.addptr %basevec, %offset : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %rowptrs = tt.broadcast %rowptr : tensor<1x16x!tt.ptr<f32>> -> tensor<32x16x!tt.ptr<f32>>
    %dcol = tt.expand_dims %d {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %ds = tt.broadcast %dcol : tensor<32x1xi32> -> tensor<32x16xi32>
    %p = tt.addptr %rowptrs, %ds : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
    %val = tt.load %p : tensor<32x16x!tt.ptr<f32>>
    tt.return %val : tensor<32x16xf32>
  }
}
