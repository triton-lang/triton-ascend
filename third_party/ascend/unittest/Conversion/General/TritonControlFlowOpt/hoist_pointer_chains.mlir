// RUN: triton-opt --triton-hoist-pointer-chains %s | FileCheck %s

module {
  tt.func public @hoist_advance_from_scope(%base: !tt.ptr<f32>) -> !tt.ptr<tensor<128x128xf32>> {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %ptr0 = tt.make_tensor_ptr %base, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x128xf32>>
    %final = scf.for %iv = %c0 to %c128 step %c1 iter_args(%ptr = %ptr0) -> (!tt.ptr<tensor<128x128xf32>>) {
      %scope_res = scope.scope : () -> !tt.ptr<tensor<128x128xf32>> {
        %next = tt.advance %ptr, [%c0_i32, %c128_i32] : <tensor<128x128xf32>>
        scope.return %next : !tt.ptr<tensor<128x128xf32>>
      } {no_inline}
      scf.yield %scope_res : !tt.ptr<tensor<128x128xf32>>
    }
    tt.return %final : !tt.ptr<tensor<128x128xf32>>
  }
}

// CHECK-LABEL: tt.func public @hoist_advance_from_scope
// CHECK:         scope.scope : () -> !tt.ptr<tensor<128x128xf32>>
// CHECK:           scope.return %[[PTR:.*]] : !tt.ptr<tensor<128x128xf32>>
// CHECK-NOT:       tt.advance
// CHECK:         %[[ADV:.*]] = tt.advance %[[PTR]], [%c0_i32, %c128_i32] : <tensor<128x128xf32>>
// CHECK:         scf.yield %[[ADV]] : !tt.ptr<tensor<128x128xf32>>
