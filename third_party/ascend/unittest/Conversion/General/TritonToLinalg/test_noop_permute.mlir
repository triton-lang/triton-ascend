// RUN: triton-opt '--triton-to-linalg=compile-on-910-95=true' --split-input-file %s | FileCheck %s

// A permutation that only moves size-1 dims does not move data. It must lower
// to a reshape, not to a real cube transpose.

// CHECK-LABEL: func.func @noop_permute_2d(
// CHECK: tensor.expand_shape {{.*}} output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
// CHECK-NOT: linalg.transpose

module {
  tt.func public @noop_permute_2d(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %sp = tt.splat %in : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %p = tt.addptr %sp, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %v = tt.load %p : tensor<16x!tt.ptr<f32>>
    %e = tt.expand_dims %v {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
    %t = tt.trans %e {order = array<i32: 1, 0>} : tensor<1x16xf32> -> tensor<16x1xf32>
    %z = arith.constant dense<0> : tensor<16x1xi32>
    %so = tt.splat %out : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %po = tt.addptr %so, %z : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    tt.store %po, %t : tensor<16x1x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @noop_permute_3d(
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 1, 4] : tensor<4xf32> into tensor<1x1x4xf32>
// CHECK-NOT: linalg.transpose

module {
  tt.func public @noop_permute_3d(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %sp = tt.splat %in : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %p = tt.addptr %sp, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %p : tensor<4x!tt.ptr<f32>>
    %e1 = tt.expand_dims %v {axis = 0 : i32} : tensor<4xf32> -> tensor<1x4xf32>
    %e2 = tt.expand_dims %e1 {axis = 2 : i32} : tensor<1x4xf32> -> tensor<1x4x1xf32>
    %t = tt.trans %e2 {order = array<i32: 0, 2, 1>} : tensor<1x4x1xf32> -> tensor<1x1x4xf32>
    %z = arith.constant dense<0> : tensor<1x1x4xi32>
    %so = tt.splat %out : !tt.ptr<f32> -> tensor<1x1x4x!tt.ptr<f32>>
    %po = tt.addptr %so, %z : tensor<1x1x4x!tt.ptr<f32>>, tensor<1x1x4xi32>
    tt.store %po, %t : tensor<1x1x4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A real transpose (two non-singleton dims swap) must keep linalg.transpose.

// CHECK-LABEL: func.func @real_permute_2d(
// CHECK-NOT: tensor.collapse_shape
// CHECK: linalg.transpose ins({{.*}} : tensor<4x16xf32>) outs({{.*}} : tensor<16x4xf32>) permutation = [1, 0]

module {
  tt.func public @real_permute_2d(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %r4 = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %r16 = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %a = tt.expand_dims %r4 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %b = tt.expand_dims %r16 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %ab = tt.broadcast %a : tensor<4x1xi32> -> tensor<4x16xi32>
    %bb = tt.broadcast %b : tensor<1x16xi32> -> tensor<4x16xi32>
    %off = arith.addi %ab, %bb : tensor<4x16xi32>
    %sp = tt.splat %in : !tt.ptr<f32> -> tensor<4x16x!tt.ptr<f32>>
    %p = tt.addptr %sp, %off : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    %v = tt.load %p : tensor<4x16x!tt.ptr<f32>>
    %t = tt.trans %v {order = array<i32: 1, 0>} : tensor<4x16xf32> -> tensor<16x4xf32>
    %offt = tt.trans %off {order = array<i32: 1, 0>} : tensor<4x16xi32> -> tensor<16x4xi32>
    %so = tt.splat %out : !tt.ptr<f32> -> tensor<16x4x!tt.ptr<f32>>
    %po = tt.addptr %so, %offt : tensor<16x4x!tt.ptr<f32>>, tensor<16x4xi32>
    tt.store %po, %t : tensor<16x4x!tt.ptr<f32>>
    tt.return
  }
}
