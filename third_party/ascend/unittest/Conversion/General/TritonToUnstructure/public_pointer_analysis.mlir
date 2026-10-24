// RUN: triton-opt %s --triton-to-unstructure | FileCheck %s
// RUN: triton-opt %s --triton-to-unstructure=compile-on-910-95=true | FileCheck %s --check-prefix=TEMPLATE

// Each narrow addptr delta is extended independently. The shared i32 sum must
// retain modular source arithmetic, rather than adding two widened operands.
tt.func @common_complete_offset(%base: !tt.ptr<i32>, %x: tensor<8xi32>, %y: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %a = tt.addptr %p, %x : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  %sum = arith.addi %x, %y : tensor<8xi32>
  %b = tt.addptr %a, %sum : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  %v = tt.load %b : tensor<8x!tt.ptr<i32>>
  tt.return %v, %sum : tensor<8xi32>, tensor<8xi32>
}
// CHECK-LABEL: @common_complete_offset
// CHECK: %[[SUM:.*]] = arith.addi {{.*}} : tensor<8xi32>
// CHECK: arith.extsi %[[SUM]] : tensor<8xi32> to tensor<8xi64>
// CHECK: arith.addi {{.*}} : tensor<8xi64>
// CHECK: scf.for
// CHECK: tt.load {{.*}} {DiscreteMemAccess}
// CHECK: tt.return {{.*}}, %[[SUM]]
// TEMPLATE-LABEL: @common_complete_offset
// TEMPLATE: arith.extsi {{.*}} : tensor<8xi32> to tensor<8xi64>
// TEMPLATE: arith.addi {{.*}} : tensor<8xi64>
// TEMPLATE: ascend.indirect_load

// A same-base select is represented by a selected complete integer offset.
// Pointer selection must not survive into the scalar memory loop.
tt.func @same_base_select(%base: !tt.ptr<f32>, %x: tensor<8xi64>, %y: tensor<8xi64>, %cond: tensor<8xi1>) -> tensor<8xf32> {
  %p = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %a = tt.addptr %p, %x : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
  %b = tt.addptr %p, %y : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
  %selected = arith.select %cond, %a, %b : tensor<8xi1>, tensor<8x!tt.ptr<f32>>
  %v = tt.load %selected : tensor<8x!tt.ptr<f32>>
  tt.return %v : tensor<8xf32>
}
// CHECK-LABEL: @same_base_select
// CHECK: arith.select {{.*}} : tensor<8xi1>, tensor<8xi64>
// CHECK-NOT: tensor<8x!tt.ptr<f32>>
// CHECK: scf.for
// CHECK: tt.load {{.*}} {DiscreteMemAccess}
// TEMPLATE-LABEL: @same_base_select
// TEMPLATE: arith.select {{.*}} : tensor<8xi1>, tensor<8xi64>
// TEMPLATE: ascend.indirect_load

// A local reshape boundary must be normalized before a following common
// addptr is analyzed. The exact complete offset uses the same lane mapping.
tt.func @reshape_then_common(%base: !tt.ptr<f32>, %x: tensor<2x4xi64>, %delta: tensor<8xi32>) -> tensor<8xf32> {
  %p = tt.splat %base : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
  %a = tt.addptr %p, %x : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi64>
  %flat = tt.reshape %a allow_reorder : tensor<2x4x!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
  %b = tt.addptr %flat, %delta : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %v = tt.load %b : tensor<8x!tt.ptr<f32>>
  tt.return %v : tensor<8xf32>
}
// CHECK-LABEL: @reshape_then_common
// CHECK-DAG: tt.reshape {{.*}} : tensor<2x4xi64> -> tensor<8xi64>
// CHECK-DAG: arith.extsi {{.*}} : tensor<8xi32> to tensor<8xi64>
// CHECK: arith.addi {{.*}} : tensor<8xi64>
// CHECK: scf.for
// TEMPLATE-LABEL: @reshape_then_common
// TEMPLATE: tt.reshape {{.*}} : tensor<2x4xi64> -> tensor<8xi64>
// TEMPLATE: ascend.indirect_load

// A fully structured pointer needs no Unstructure rewrite. Numeric helpers
// created while classifying it must disappear, while the source remains.
tt.func @abandoned_numeric_helpers(%base: !tt.ptr<f32>) -> tensor<8xf32> {
  %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %p = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %a = tt.addptr %p, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %v = tt.load %a : tensor<8x!tt.ptr<f32>>
  tt.return %v : tensor<8xf32>
}
// CHECK-LABEL: @abandoned_numeric_helpers
// CHECK-NOT: i64
// CHECK-NOT: scf.for
// CHECK: tt.load {{.*}} : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: tt.return
// TEMPLATE-LABEL: @abandoned_numeric_helpers
// TEMPLATE-NOT: i64
// TEMPLATE: tt.load {{.*}} : tensor<8x!tt.ptr<f32>>

// The opaque outer axis and affine inner axis remain distinct in i64 address
// arithmetic. Only the outer dimension should become an explicit loop.
tt.func @partial_inner_axis(%base: !tt.ptr<f32>, %indices: !tt.ptr<i64>) -> tensor<2x16xf32> {
  %r2 = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %ip = tt.splat %indices : !tt.ptr<i64> -> tensor<2x!tt.ptr<i64>>
  %ips = tt.addptr %ip, %r2 : tensor<2x!tt.ptr<i64>>, tensor<2xi32>
  %rows = tt.load %ips : tensor<2x!tt.ptr<i64>>
  %row2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
  %row = tt.broadcast %row2 : tensor<2x1xi64> -> tensor<2x16xi64>
  %r16 = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %col64 = arith.extsi %r16 : tensor<16xi32> to tensor<16xi64>
  %col2 = tt.expand_dims %col64 {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64>
  %col = tt.broadcast %col2 : tensor<1x16xi64> -> tensor<2x16xi64>
  %offset = arith.addi %row, %col : tensor<2x16xi64>
  %p = tt.splat %base : !tt.ptr<f32> -> tensor<2x16x!tt.ptr<f32>>
  %a = tt.addptr %p, %offset : tensor<2x16x!tt.ptr<f32>>, tensor<2x16xi64>
  %v = tt.load %a : tensor<2x16x!tt.ptr<f32>>
  tt.return %v : tensor<2x16xf32>
}
// CHECK-LABEL: @partial_inner_axis
// CHECK: scf.for
// CHECK-NOT: scf.for
// CHECK: tt.load {{.*}} : tensor<1x16x!tt.ptr<f32>>
// CHECK-NOT: scf.for
// CHECK: tt.return
// TEMPLATE-LABEL: @partial_inner_axis
// TEMPLATE: scf.for
// TEMPLATE-NOT: ascend.indirect_load
// TEMPLATE: tt.return

// Descriptor metadata stays with the local adapter even when its base chain
// has already been decomposed by the public ordinary-pointer analysis.
tt.func @descriptor_owned_addptr(%base: !tt.ptr<f32>, %offset: tensor<8xi64>) -> tensor<8xf32> {
  %p = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %a = tt.addptr %p, %offset {PointerDescriptorRebuild, PointerDescriptorStructuredAxes = array<i32: 0>} : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
  %v = tt.load %a : tensor<8x!tt.ptr<f32>>
  tt.return %v : tensor<8xf32>
}
// CHECK-LABEL: @descriptor_owned_addptr
// CHECK: scf.for
// CHECK: tt.load {{.*}} {DiscreteMemAccess}
// TEMPLATE-LABEL: @descriptor_owned_addptr
// TEMPLATE: ascend.indirect_load

// Narrow tensor arithmetic is an explicitly unmigrated local subset. Preserve
// its existing classification without binding its source tags as public address
// proofs. The ordinary i64 path is tested separately above.
tt.func @dynamic_i32_widen(%base: !tt.ptr<i32>, %start: i32) -> tensor<8xi32> {
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %s = tt.splat %start : i32 -> tensor<8xi32>
  %o = arith.addi %r, %s : tensor<8xi32>
  %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %a = tt.addptr %p, %o : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  %v = tt.load %a : tensor<8x!tt.ptr<i32>>
  tt.return %v : tensor<8xi32>
}
// CHECK-LABEL: @dynamic_i32_widen
// CHECK: %[[NARROW:.*]] = arith.addi {{.*}} : tensor<8xi32>
// CHECK-NOT: scf.for
// CHECK: tt.addptr {{.*}}, %[[NARROW]]
// CHECK: tt.load {{.*}} : tensor<8x!tt.ptr<i32>>
// TEMPLATE-LABEL: @dynamic_i32_widen
// TEMPLATE: arith.addi {{.*}} : tensor<8xi32>
// TEMPLATE-NOT: ascend.indirect_load
// TEMPLATE: tt.load {{.*}} : tensor<8x!tt.ptr<i32>>

// Two local normalizations may reset a session while preparing one public
// select. Rebind both boundaries, and preserve facts for an earlier load too.
tt.func @reshape_select_boundaries(%base: !tt.ptr<i32>, %x: tensor<2x4xi64>, %y: tensor<2x4xi64>, %cond: tensor<8xi1>) -> (tensor<2x4xi32>, tensor<8xi32>) {
  %p = tt.splat %base : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
  %a = tt.addptr %p, %x : tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi64>
  %earlier = tt.load %a : tensor<2x4x!tt.ptr<i32>>
  %b = tt.addptr %p, %y : tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi64>
  %af = tt.reshape %a allow_reorder : tensor<2x4x!tt.ptr<i32>> -> tensor<8x!tt.ptr<i32>>
  %bf = tt.reshape %b allow_reorder : tensor<2x4x!tt.ptr<i32>> -> tensor<8x!tt.ptr<i32>>
  %selected = arith.select %cond, %af, %bf : tensor<8xi1>, tensor<8x!tt.ptr<i32>>
  %later = tt.load %selected : tensor<8x!tt.ptr<i32>>
  tt.return %earlier, %later : tensor<2x4xi32>, tensor<8xi32>
}
// CHECK-LABEL: @reshape_select_boundaries
// CHECK: tt.load {{.*}} {DiscreteMemAccess}
// CHECK: arith.select {{.*}} : tensor<8xi1>, tensor<8xi64>
// CHECK: tt.load {{.*}} {DiscreteMemAccess}
// TEMPLATE-LABEL: @reshape_select_boundaries
// TEMPLATE: ascend.indirect_load
// TEMPLATE: arith.select {{.*}} : tensor<8xi1>, tensor<8xi64>
// TEMPLATE: ascend.indirect_load

// A truncation on the public i64 arithmetic path is still modular source
// arithmetic. It must execute before sign extension into the complete address;
// the wrapped lanes cannot be promoted to a proven wide affine progression.
tt.func @public_truncated_widen(%base: !tt.ptr<i32>, %start: i64) -> tensor<8xi32> {
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %wide = arith.extsi %r : tensor<8xi32> to tensor<8xi64>
  %s = tt.splat %start : i64 -> tensor<8xi64>
  %sum = arith.addi %wide, %s : tensor<8xi64>
  %narrow = arith.trunci %sum : tensor<8xi64> to tensor<8xi32>
  %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %a = tt.addptr %p, %narrow : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  %v = tt.load %a : tensor<8x!tt.ptr<i32>>
  tt.return %v : tensor<8xi32>
}
// CHECK-LABEL: @public_truncated_widen
// CHECK: %[[TRUNC:.*]] = arith.trunci {{.*}} : tensor<8xi64> to tensor<8xi32>
// CHECK: arith.extsi %[[TRUNC]] : tensor<8xi32> to tensor<8xi64>
// CHECK: scf.for
// CHECK: tt.load {{.*}} {DiscreteMemAccess}
// TEMPLATE-LABEL: @public_truncated_widen
// TEMPLATE: arith.trunci {{.*}} : tensor<8xi64> to tensor<8xi32>
// TEMPLATE: arith.extsi {{.*}} : tensor<8xi32> to tensor<8xi64>
// TEMPLATE: ascend.indirect_load

// Floating-point propagation is explicitly local, including arithmetic users
// of the resulting integer tensor. Do not lose its existing uniform-lane fact
// by importing the special producer as an opaque tensor in a public query.
tt.func @local_float_offset(%base: !tt.ptr<i32>, %offset: f32) -> tensor<8xi32> {
  %s = tt.splat %offset : f32 -> tensor<8xf32>
  %i = arith.fptosi %s : tensor<8xf32> to tensor<8xi64>
  %one = arith.constant dense<1> : tensor<8xi64>
  %o = arith.addi %i, %one : tensor<8xi64>
  %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %a = tt.addptr %p, %o : tensor<8x!tt.ptr<i32>>, tensor<8xi64>
  %v = tt.load %a : tensor<8x!tt.ptr<i32>>
  tt.return %v : tensor<8xi32>
}
// CHECK-LABEL: @local_float_offset
// CHECK-NOT: scf.for
// CHECK: %[[SCALAR:.*]] = tt.load {{.*}} : !tt.ptr<i32>
// CHECK: tt.splat %[[SCALAR]] : i32 -> tensor<8xi32>
// TEMPLATE-LABEL: @local_float_offset
// TEMPLATE-NOT: ascend.indirect_load
// TEMPLATE: %[[SCALAR:.*]] = tt.load {{.*}} : !tt.ptr<i32>
// TEMPLATE: tt.splat %[[SCALAR]] : i32 -> tensor<8xi32>
