// RUN: triton-opt %s --triton-to-linalg --verify-each | FileCheck %s

// CHECK-LABEL: func.func @caller(
// CHECK: call @pair(
// CHECK-SAME: -> (i64, i64)
// CHECK-NOT: call @pair
// CHECK: hivm.hir.pointer_cast
// CHECK: hivm.hir.pointer_cast
// CHECK-NOT: tt.call
// CHECK: return
// CHECK-LABEL: func.func private @pair(
// CHECK-SAME: -> (i64, i64)
// CHECK-SAME: hacc.function_kind = #hacc.function_kind<DEVICE>
// CHECK-SAME: no_inline
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: return %{{.*}}, %{{.*}} : i64, i64
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @caller(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>,
                         %out: !tt.ptr<f32>, %shift: i32) {
    %base = tt.addptr %x, %shift : !tt.ptr<f32>, i32
    %p:2 = tt.call @pair(%base, %y, %shift)
        : (!tt.ptr<f32>, !tt.ptr<f32>, i32) -> (!tt.ptr<f32>, !tt.ptr<f32>)
    %offsets = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %px = tt.splat %p#0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %py = tt.splat %p#1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %po = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %xp = tt.addptr %px, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %yp = tt.addptr %py, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %op = tt.addptr %po, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %vx = tt.load %xp : tensor<16x!tt.ptr<f32>>
    %vy = tt.load %yp : tensor<16x!tt.ptr<f32>>
    %sum = arith.addf %vx, %vy : tensor<16xf32>
    tt.store %op, %sum : tensor<16x!tt.ptr<f32>>
    tt.return
  }
  tt.func private @pair(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %shift: i32)
      -> (!tt.ptr<f32>, !tt.ptr<f32>) attributes {noinline = true} {
    %px = tt.addptr %x, %shift : !tt.ptr<f32>, i32
    %py = tt.addptr %y, %shift : !tt.ptr<f32>, i32
    tt.return %px, %py : !tt.ptr<f32>, !tt.ptr<f32>
  }
}
