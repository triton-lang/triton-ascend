// RUN: triton-opt '--triton-to-structured=compile-on-910-95=true' --verify-each %s | FileCheck %s

// Reduced address pattern from the bucket-softmax regression:
//   base + y + 8 * bucket, with result shape [8, 1, 2048].
// The loaded indices stand in for the data-dependent bucket calculation.
// Permuting to [2048, 8, 1] gives only 8 * sizeof(f32) = 32 contiguous
// bytes. Keep the original load shape and do not insert tt.trans.
// CHECK-LABEL: tt.func public @skip_short_contiguous_load(
// CHECK-NOT: ImplicitPermuteHandled
// CHECK-NOT: tt.trans
// CHECK: %[[LOAD:.*]] = tt.load {{.*}} : tensor<8x1x2048x!tt.ptr<f32>>
// CHECK-NOT: ImplicitPermuteHandled
// CHECK-NOT: tt.trans
// CHECK: tt.return %[[LOAD]] : tensor<8x1x2048xf32>
module {
  tt.func public @skip_short_contiguous_load(%base: !tt.ptr<f32>, %indices: !tt.ptr<i32>) -> tensor<8x1x2048xf32> {
    %irange = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32>
    %ibase = tt.splat %indices : !tt.ptr<i32> -> tensor<2048x!tt.ptr<i32>>
    %iptr = tt.addptr %ibase, %irange : tensor<2048x!tt.ptr<i32>>, tensor<2048xi32>
    %token = tt.load %iptr : tensor<2048x!tt.ptr<i32>>
    %token_2d = tt.expand_dims %token {axis = 0 : i32} : tensor<2048xi32> -> tensor<1x2048xi32>
    %data = tt.expand_dims %token_2d {axis = 0 : i32} : tensor<1x2048xi32> -> tensor<1x1x2048xi32>
    %scale = arith.constant dense<8> : tensor<1x1x2048xi32>
    %scaled = arith.muli %data, %scale : tensor<1x1x2048xi32>
    %cols = tt.broadcast %scaled : tensor<1x1x2048xi32> -> tensor<8x1x2048xi32>
    %y = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %y_2d = tt.expand_dims %y {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %y_3d = tt.expand_dims %y_2d {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %rows = tt.broadcast %y_3d : tensor<8x1x1xi32> -> tensor<8x1x2048xi32>
    %offset = arith.addi %rows, %cols : tensor<8x1x2048xi32>
    %base_tile = tt.splat %base : !tt.ptr<f32> -> tensor<8x1x2048x!tt.ptr<f32>>
    %ptr = tt.addptr %base_tile, %offset : tensor<8x1x2048x!tt.ptr<f32>>, tensor<8x1x2048xi32>
    %value = tt.load %ptr : tensor<8x1x2048x!tt.ptr<f32>>
    tt.return %value : tensor<8x1x2048xf32>
  }
}
