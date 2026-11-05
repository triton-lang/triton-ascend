// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=262144' | FileCheck %s
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 ub-capacity-bytes=1024' | FileCheck %s --check-prefix=KEEP --implicit-check-not=hacc.program_grid_transforms
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 compile-mode=simt_only' | FileCheck %s --check-prefix=KEEP --implicit-check-not=hacc.program_grid_transforms

// Fast path removes proven chunk/column masks; slow path keeps tail protection.
// CHECK: axis = 0 : i64, factor = 32 : i64
// CHECK-LABEL: tt.func public @merge_rows(
// CHECK: %[[TAIL:.*]] = arith.cmpi ult, {{.*}} {IATRuntimeExtentUnsignedMask}
// CHECK: %[[SAFE:.*]] = arith.andi {{.*}}, %[[TAIL]]
// CHECK: %[[MASK:.*]] = arith.extui %[[SAFE]]
// CHECK: "tt.reduce"(%[[MASK]]) <{axis = 0 : i32}>
// CHECK: arith.minsi
// CHECK: %[[COLS:.*]] = arith.cmpi sge, {{.*}} : i32
// CHECK-NEXT: %[[GUARD:.*]] = arith.andi {{.*}}, %[[COLS]] : i1
// CHECK-NEXT: scf.if %[[GUARD]] {
// CHECK: tt.load %{{[a-zA-Z0-9_]+}} : tensor<32x!tt.ptr<f32>>
// CHECK: tt.load %{{[a-zA-Z0-9_]+}} : tensor<32x128x!tt.ptr<f32>>
// CHECK: } else {
// CHECK: tt.load {{.*}}, {{.*}}, {{.*}} : tensor<32x!tt.ptr<f32>>
// CHECK: %[[MASK2D:.*]] = arith.andi {{.*}} : tensor<32x128xi1>
// CHECK-NEXT: tt.load {{.*}}, %[[MASK2D]], {{.*}} : tensor<32x128x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, {{.*}}, %[[MASK2D]] : tensor<32x128x!tt.ptr<f32>>
// KEEP-NOT: scf.if
// KEEP: tt.store {{.*}}, {{.*}}, {{.*}} : tensor<128x!tt.ptr<f32>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @merge_rows(%counts: !tt.ptr<i32>, %lse: !tt.ptr<f32>, %chunks: !tt.ptr<f32>, %output: !tt.ptr<f32>, %dim: i32) {
    %width = arith.constant 128 : i32
    %one = arith.constant 1 : i32
    %zero = arith.constant 0.0 : f32
    %zeros = arith.constant dense<0.0> : tensor<128xf32>
    %pid = tt.get_program_id x : i32
    %cp = tt.addptr %counts, %pid : !tt.ptr<i32>, i32
    %count = tt.load %cp : !tt.ptr<i32>
    %valid = arith.cmpi sgt, %count, %one : i32
    %lp = tt.addptr %lse, %pid : !tt.ptr<f32>, i32
    %l = tt.load %lp, %valid, %zero : !tt.ptr<f32>
    %weight = math.exp2 %l : f32
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %dims = tt.splat %dim : i32 -> tensor<128xi32>
    %columns = arith.cmpi slt, %r, %dims : tensor<128xi32>
    %base = arith.muli %pid, %width : i32
    %row = tt.splat %base : i32 -> tensor<128xi32>
    %index = arith.addi %row, %r : tensor<128xi32>
    %ip = tt.splat %chunks : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %op = tt.splat %output : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %src = tt.addptr %ip, %index : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %dst = tt.addptr %op, %index : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %input = tt.load %src, %columns, %zeros : tensor<128x!tt.ptr<f32>>
    %weights = tt.splat %weight : f32 -> tensor<128xf32>
    %result = arith.mulf %input, %weights : tensor<128xf32>
    tt.store %dst, %result, %columns : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
