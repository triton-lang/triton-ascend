// RUN: triton-opt --split-input-file '--triton-to-structured=compile-on-910-95=true' --verify-each %s | FileCheck %s
// RUN: triton-opt --split-input-file '--triton-to-structured=compile-on-910-95=true force-explicit-load-transpose=false' --verify-each %s | FileCheck %s
// RUN: triton-opt --split-input-file '--triton-to-structured=compile-on-910-95=true force-explicit-load-transpose=true' --verify-each %s | FileCheck %s

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

  // A full 64-byte row is eligible even with a data-dependent column index.
  // CHECK-LABEL: tt.func public @indirect_load
  // CHECK: tt.load {{.*}} : tensor<8x16x!tt.ptr<f32>>
  // CHECK: tt.trans {{.*}}implicit_load_permute{{.*}}tensor<8x16xf32> -> tensor<16x8xf32>
  tt.func public @indirect_load(%base: !tt.ptr<f32>, %indices: !tt.ptr<i32>) -> tensor<16x8xf32> {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %p = tt.splat %indices : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %iptr = tt.addptr %p, %r : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %token = tt.load %iptr : tensor<8x!tt.ptr<i32>>
    %stride = arith.constant dense<16> : tensor<8xi32>
    %row = arith.muli %token, %stride : tensor<8xi32>
    %x = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %xx = tt.expand_dims %x {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %rr = tt.expand_dims %row {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %xb = tt.broadcast %xx : tensor<16x1xi32> -> tensor<16x8xi32>
    %rb = tt.broadcast %rr : tensor<1x8xi32> -> tensor<16x8xi32>
    %off = arith.addi %xb, %rb : tensor<16x8xi32>
    %bp = tt.splat %base : !tt.ptr<f32> -> tensor<16x8x!tt.ptr<f32>>
    %ptr = tt.addptr %bp, %off : tensor<16x8x!tt.ptr<f32>>, tensor<16x8xi32>
    %v = tt.load %ptr : tensor<16x8x!tt.ptr<f32>>
    tt.return %v : tensor<16x8xf32>
  }

}

// -----

module {
  // Cube modules retain their explicit load-layout restore for all option values.
  // CHECK-LABEL: tt.func public @dot_load
  // CHECK: tt.trans {{.*}}implicit_load_permute
  // CHECK: tt.dot
  tt.func public @dot_load(%base: !tt.ptr<f16>, %b: tensor<16x16xf16>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i64
    %c16 = arith.constant 16 : i64
    %p = tt.make_tensor_ptr %base, [%c16, %c16], [%c1, %c16], [%c0, %c0] {order = array<i32: 0, 1>} : <tensor<16x16xf16>>
    %a = tt.load %p : !tt.ptr<tensor<16x16xf16>>
    %v = tt.dot %a, %b, %c : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    tt.return %v : tensor<16x16xf32>
  }
}
