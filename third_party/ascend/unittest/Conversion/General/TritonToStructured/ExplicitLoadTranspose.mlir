// RUN: triton-opt '--triton-to-structured=compile-on-910-95=true force-explicit-load-transpose=true' --verify-each %s | FileCheck %s --check-prefix=ON
// RUN: triton-opt '--triton-to-structured=compile-on-910-95=true' --verify-each %s | FileCheck %s --check-prefix=OFF
// RUN: triton-opt --triton-to-structured --verify-each %s | FileCheck %s --check-prefix=LEGACY
// RUN: triton-opt '--triton-to-structured=force-explicit-load-transpose=false' --verify-each %s | FileCheck %s --check-prefix=LEGACY
// RUN: triton-opt '--triton-to-structured=force-explicit-load-transpose=true' --verify-each %s | FileCheck %s --check-prefix=LEGACY
// RUN: triton-opt '--triton-to-structured=compile-on-910-95=true force-explicit-load-transpose=true' '--triton-to-linalg=compile-on-910-95=true' --verify-each %s | FileCheck %s --check-prefix=HANDOFF

// The GM rows are indexed [r,x] and the caller consumes [x,r].
// LEGACY-LABEL: tt.func @strided_load
// LEGACY: tt.load {{.*}} : tensor<8x16x!tt.ptr<f32>>
// LEGACY: tt.trans {{.*}}implicit_load_permute{{.*}}tensor<8x16xf32> -> tensor<16x8xf32>
// ON-LABEL: tt.func @strided_load
// ON: tt.load {{.*}} : tensor<8x16x!tt.ptr<f32>>
// ON: tt.trans {{.*}}implicit_load_permute{{.*}}tensor<8x16xf32> -> tensor<16x8xf32>
// OFF-LABEL: tt.func @strided_load
// OFF: tt.load {{.*}} : tensor<16x8x!tt.ptr<f32>>
// OFF-NOT: tt.trans
// OFF: tt.return
// HANDOFF-LABEL: func.func @strided_load
// HANDOFF: linalg.transpose {{.*}}implicit_load_permute
tt.func @strided_load(%base: !tt.ptr<f32>) -> tensor<16x8xf32> {
  %x = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %stride = arith.constant dense<32> : tensor<8xi32>
  %row = arith.muli %r, %stride : tensor<8xi32>
  %xx = tt.expand_dims %x {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %rr = tt.expand_dims %row {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %xb = tt.broadcast %xx : tensor<16x1xi32> -> tensor<16x8xi32>
  %rb = tt.broadcast %rr : tensor<1x8xi32> -> tensor<16x8xi32>
  %off = arith.addi %xb, %rb : tensor<16x8xi32>
  %p = tt.splat %base : !tt.ptr<f32> -> tensor<16x8x!tt.ptr<f32>>
  %ptr = tt.addptr %p, %off : tensor<16x8x!tt.ptr<f32>>, tensor<16x8xi32>
  %v = tt.load %ptr : tensor<16x8x!tt.ptr<f32>>
  tt.return %v : tensor<16x8xf32>
}

// A user transpose has the same op type but different provenance.
// ON-LABEL: tt.func @user_transpose
// ON-NOT: implicit_load_permute
// ON: tt.return
// HANDOFF-LABEL: func.func @user_transpose
// HANDOFF: linalg.transpose
// HANDOFF-NOT: implicit_load_permute
// HANDOFF: return
tt.func @user_transpose(%base: !tt.ptr<f32>) -> tensor<16x8xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %c32 = arith.constant 32 : i64
  %p = tt.make_tensor_ptr %base, [%c32, %c32], [%c32, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
  %x = tt.load %p : !tt.ptr<tensor<8x16xf32>>
  %v = tt.trans %x {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  tt.return %v : tensor<16x8xf32>
}

// A store also transposes to match memory order, in the opposite direction.
// ON-LABEL: tt.func @strided_store
// ON: tt.trans {{%[^ ]+}} {order = array<i32: 1, 0>}
// ON: tt.store
// ON-NOT: implicit_load_permute
// ON: tt.return
tt.func @strided_store(%base: !tt.ptr<f32>, %src: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %c32 = arith.constant 32 : i64
  %p = tt.make_tensor_ptr %base, [%c32, %c32], [%c1, %c32], [%c0, %c0] {order = array<i32: 0, 1>} : <tensor<4x8xf32>>
  %q = tt.make_tensor_ptr %src, [%c32, %c32], [%c32, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
  %v = tt.load %q : !tt.ptr<tensor<4x8xf32>>
  tt.store %p, %v : !tt.ptr<tensor<4x8xf32>>
  tt.return
}
