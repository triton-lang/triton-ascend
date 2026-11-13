// RUN: triton-opt %s --triton-to-unstructure --verify-each --split-input-file | FileCheck %s

// Equal addresses do not allow an unconditional load when all lanes can be
// masked. Keep the extracted predicate on the actual scalar access.
// CHECK-LABEL: tt.func @uniform_masked
// CHECK: scf.for
// CHECK: %[[MASK:.*]] = tensor.extract
// CHECK: tt.load {{.*}}, %[[MASK]],
// CHECK: tt.return
tt.func @uniform_masked(%base: !tt.ptr<f32>, %mask: tensor<4xi1>) -> tensor<4xf32> {
  %ptr = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %other = arith.constant dense<0.0> : tensor<4xf32>
  %value = tt.load %ptr, %mask, %other : tensor<4x!tt.ptr<f32>>
  tt.return %value : tensor<4xf32>
}

// -----

// Zero address stride is independent of the parent-coordinate boundary.
// CHECK-LABEL: tt.func @zero_stride_block
// CHECK-NOT: scf.for
// CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32}
// CHECK: tt.return
tt.func @zero_stride_block(%base: !tt.ptr<f32>) -> tensor<8xf32> {
  %shape = arith.constant 3 : i64
  %stride = arith.constant 0 : i64
  %offset = arith.constant 0 : i32
  %ptr = tt.make_tensor_ptr %base, [%shape], [%stride], [%offset] {order = array<i32: 0>} : <tensor<8xf32>>
  %value = tt.load %ptr {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
  tt.return %value : tensor<8xf32>
}

// -----

// A block coordinate is an accumulated index, not an i32 field. An unchanged
// loop slot must reuse the original descriptor without truncating 2^31.
// CHECK-LABEL: tt.func @wide_invariant_block
// CHECK: tt.advance
// CHECK: scf.for
// CHECK-NOT: iter_args
// CHECK: tt.load
// CHECK: tt.return
tt.func @wide_invariant_block(%base: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %shape = arith.constant 2147483664 : i64
  %stride = arith.constant 1 : i64
  %offset = arith.constant 2147483647 : i32
  %one = arith.constant 1 : i32
  %initial = tt.make_tensor_ptr %base, [%shape], [%stride], [%offset] {order = array<i32: 0>} : <tensor<8xf32>>
  %wide = tt.advance %initial, [%one] : <tensor<8xf32>>
  %outs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %outptr = tt.addptr %outs, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%ptr = %wide) -> !tt.ptr<tensor<8xf32>> {
    %value = tt.load %ptr {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
    tt.store %outptr, %value : tensor<8x!tt.ptr<f32>>
    scf.yield %ptr : !tt.ptr<tensor<8xf32>>
  }
  tt.return
}
