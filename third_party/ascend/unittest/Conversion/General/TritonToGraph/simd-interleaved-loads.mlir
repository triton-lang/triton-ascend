// RUN: triton-opt %s --split-input-file --graph-optimize="rule-mask=32768 compile-mode=simd_simt_template" --verify-each | FileCheck %s
// RUN: triton-opt %s --split-input-file --graph-optimize="rule-mask=32768 compile-mode=simt_only" --verify-each | FileCheck %s --check-prefix=SIMT
// SIMT-NOT: tensor<16x2x!tt.ptr<f32>>
// SIMT-NOT: tensor<16x4x!tt.ptr<f32>>
// SIMT-NOT: tensor<16x8x!tt.ptr<f32>>
// CHECK-LABEL: @w2(
// CHECK: tt.load {{.*}} : tensor<16x2x!tt.ptr<f32>>

module { tt.func public @w2(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<2> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v0 = tt.load %p0, %m, %z : tensor<16x!tt.ptr<f32>>
%v1 = tt.load %p1, %m, %z : tensor<16x!tt.ptr<f32>>
%sum1 = arith.addf %v0, %v1 : tensor<16xf32>
tt.store %outs, %sum1, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
// -----
// CHECK-LABEL: @w4(
// CHECK: tt.load {{.*}} : tensor<16x4x!tt.ptr<f32>>

module { tt.func public @w4(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<4> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c2 = arith.constant dense<2> : tensor<16xi32>
%p2 = tt.addptr %rows, %c2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c3 = arith.constant dense<3> : tensor<16xi32>
%p3 = tt.addptr %rows, %c3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v0 = tt.load %p0, %m, %z : tensor<16x!tt.ptr<f32>>
%v1 = tt.load %p1, %m, %z : tensor<16x!tt.ptr<f32>>
%v2 = tt.load %p2, %m, %z : tensor<16x!tt.ptr<f32>>
%v3 = tt.load %p3, %m, %z : tensor<16x!tt.ptr<f32>>
%sum1 = arith.addf %v0, %v1 : tensor<16xf32>
%sum2 = arith.addf %sum1, %v2 : tensor<16xf32>
%sum3 = arith.addf %sum2, %v3 : tensor<16xf32>
tt.store %outs, %sum3, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
// -----
// CHECK-LABEL: @w8(
// CHECK: tt.load {{.*}} : tensor<16x8x!tt.ptr<f32>>

module { tt.func public @w8(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<8> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c2 = arith.constant dense<2> : tensor<16xi32>
%p2 = tt.addptr %rows, %c2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c3 = arith.constant dense<3> : tensor<16xi32>
%p3 = tt.addptr %rows, %c3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c4 = arith.constant dense<4> : tensor<16xi32>
%p4 = tt.addptr %rows, %c4 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c5 = arith.constant dense<5> : tensor<16xi32>
%p5 = tt.addptr %rows, %c5 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c6 = arith.constant dense<6> : tensor<16xi32>
%p6 = tt.addptr %rows, %c6 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c7 = arith.constant dense<7> : tensor<16xi32>
%p7 = tt.addptr %rows, %c7 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v0 = tt.load %p0, %m, %z : tensor<16x!tt.ptr<f32>>
%v1 = tt.load %p1, %m, %z : tensor<16x!tt.ptr<f32>>
%v2 = tt.load %p2, %m, %z : tensor<16x!tt.ptr<f32>>
%v3 = tt.load %p3, %m, %z : tensor<16x!tt.ptr<f32>>
%v4 = tt.load %p4, %m, %z : tensor<16x!tt.ptr<f32>>
%v5 = tt.load %p5, %m, %z : tensor<16x!tt.ptr<f32>>
%v6 = tt.load %p6, %m, %z : tensor<16x!tt.ptr<f32>>
%v7 = tt.load %p7, %m, %z : tensor<16x!tt.ptr<f32>>
%sum1 = arith.addf %v0, %v1 : tensor<16xf32>
%sum2 = arith.addf %sum1, %v2 : tensor<16xf32>
%sum3 = arith.addf %sum2, %v3 : tensor<16xf32>
%sum4 = arith.addf %sum3, %v4 : tensor<16xf32>
%sum5 = arith.addf %sum4, %v5 : tensor<16xf32>
%sum6 = arith.addf %sum5, %v6 : tensor<16xf32>
%sum7 = arith.addf %sum6, %v7 : tensor<16xf32>
tt.store %outs, %sum7, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
// -----
// CHECK-LABEL: @missing_first(
// CHECK: tt.load {{.*}} : tensor<16x4x!tt.ptr<f32>>

module { tt.func public @missing_first(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<4> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c2 = arith.constant dense<2> : tensor<16xi32>
%p2 = tt.addptr %rows, %c2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c3 = arith.constant dense<3> : tensor<16xi32>
%p3 = tt.addptr %rows, %c3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v1 = tt.load %p1, %m, %z : tensor<16x!tt.ptr<f32>>
%v2 = tt.load %p2, %m, %z : tensor<16x!tt.ptr<f32>>
%v3 = tt.load %p3, %m, %z : tensor<16x!tt.ptr<f32>>
%sum2 = arith.addf %v1, %v2 : tensor<16xf32>
%sum3 = arith.addf %sum2, %v3 : tensor<16xf32>
tt.store %outs, %sum3, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
// -----
// CHECK-LABEL: @safe_store(
// CHECK: tt.load {{.*}} : tensor<16x4x!tt.ptr<f32>>

module { tt.func public @safe_store(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<4> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c2 = arith.constant dense<2> : tensor<16xi32>
%p2 = tt.addptr %rows, %c2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c3 = arith.constant dense<3> : tensor<16xi32>
%p3 = tt.addptr %rows, %c3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v1 = tt.load %p1, %m, %z : tensor<16x!tt.ptr<f32>>
tt.store %p0, %v1, %m : tensor<16x!tt.ptr<f32>>
%v2 = tt.load %p2, %m, %z : tensor<16x!tt.ptr<f32>>
%v3 = tt.load %p3, %m, %z : tensor<16x!tt.ptr<f32>>
%sum2 = arith.addf %v1, %v2 : tensor<16xf32>
%sum3 = arith.addf %sum2, %v3 : tensor<16xf32>
tt.store %outs, %sum3, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
// -----
// CHECK-LABEL: @unknown_alias(
// CHECK: tt.load {{.*}} : tensor<16x!tt.ptr<f32>>
// CHECK: tt.store
// CHECK: tt.load {{.*}} : tensor<16x4x!tt.ptr<f32>>

module { tt.func public @unknown_alias(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<4> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c2 = arith.constant dense<2> : tensor<16xi32>
%p2 = tt.addptr %rows, %c2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c3 = arith.constant dense<3> : tensor<16xi32>
%p3 = tt.addptr %rows, %c3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v0 = tt.load %p0, %m, %z : tensor<16x!tt.ptr<f32>>
tt.store %outs, %v0, %m : tensor<16x!tt.ptr<f32>>
%v1 = tt.load %p1, %m, %z : tensor<16x!tt.ptr<f32>>
%v2 = tt.load %p2, %m, %z : tensor<16x!tt.ptr<f32>>
%v3 = tt.load %p3, %m, %z : tensor<16x!tt.ptr<f32>>
%sum1 = arith.addf %v0, %v1 : tensor<16xf32>
%sum2 = arith.addf %sum1, %v2 : tensor<16xf32>
%sum3 = arith.addf %sum2, %v3 : tensor<16xf32>
tt.store %outs, %sum3, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
// -----
// CHECK-LABEL: @clobber(
// CHECK: tt.load {{.*}} : tensor<16x!tt.ptr<f32>>
// CHECK: tt.store
// CHECK: tt.load {{.*}} : tensor<16x4x!tt.ptr<f32>>

module { tt.func public @clobber(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<4> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c2 = arith.constant dense<2> : tensor<16xi32>
%p2 = tt.addptr %rows, %c2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c3 = arith.constant dense<3> : tensor<16xi32>
%p3 = tt.addptr %rows, %c3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v0 = tt.load %p0, %m, %z : tensor<16x!tt.ptr<f32>>
tt.store %p1, %v0, %m : tensor<16x!tt.ptr<f32>>
%v1 = tt.load %p1, %m, %z : tensor<16x!tt.ptr<f32>>
%v2 = tt.load %p2, %m, %z : tensor<16x!tt.ptr<f32>>
%v3 = tt.load %p3, %m, %z : tensor<16x!tt.ptr<f32>>
%sum1 = arith.addf %v0, %v1 : tensor<16xf32>
%sum2 = arith.addf %sum1, %v2 : tensor<16xf32>
%sum3 = arith.addf %sum2, %v3 : tensor<16xf32>
tt.store %outs, %sum3, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
// -----
// CHECK-LABEL: @volatile(
// CHECK-NOT: tensor<16x4x!tt.ptr<f32>>
// CHECK: tt.return

module { tt.func public @volatile(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<4> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c2 = arith.constant dense<2> : tensor<16xi32>
%p2 = tt.addptr %rows, %c2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c3 = arith.constant dense<3> : tensor<16xi32>
%p3 = tt.addptr %rows, %c3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v0 = tt.load %p0, %m, %z : tensor<16x!tt.ptr<f32>>
%v1 = tt.load %p1, %m, %z {isVolatile = true} : tensor<16x!tt.ptr<f32>>
%v2 = tt.load %p2, %m, %z : tensor<16x!tt.ptr<f32>>
%v3 = tt.load %p3, %m, %z : tensor<16x!tt.ptr<f32>>
%sum1 = arith.addf %v0, %v1 : tensor<16xf32>
%sum2 = arith.addf %sum1, %v2 : tensor<16xf32>
%sum3 = arith.addf %sum2, %v3 : tensor<16xf32>
tt.store %outs, %sum3, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
// -----
// CHECK-LABEL: @mask_mismatch(
// CHECK-NOT: tensor<16x4x!tt.ptr<f32>>
// CHECK: tt.return

module { tt.func public @mask_mismatch(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<4> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c2 = arith.constant dense<2> : tensor<16xi32>
%p2 = tt.addptr %rows, %c2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c3 = arith.constant dense<3> : tensor<16xi32>
%p3 = tt.addptr %rows, %c3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v0 = tt.load %p0, %m, %z : tensor<16x!tt.ptr<f32>>
%v1 = tt.load %p1, %false, %z : tensor<16x!tt.ptr<f32>>
%v2 = tt.load %p2, %m, %z : tensor<16x!tt.ptr<f32>>
%v3 = tt.load %p3, %m, %z : tensor<16x!tt.ptr<f32>>
%sum1 = arith.addf %v0, %v1 : tensor<16xf32>
%sum2 = arith.addf %sum1, %v2 : tensor<16xf32>
%sum3 = arith.addf %sum2, %v3 : tensor<16xf32>
tt.store %outs, %sum3, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
// -----
// CHECK-LABEL: @other_mismatch(
// CHECK-NOT: tensor<16x4x!tt.ptr<f32>>
// CHECK: tt.return

module { tt.func public @other_mismatch(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<4> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c2 = arith.constant dense<2> : tensor<16xi32>
%p2 = tt.addptr %rows, %c2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c3 = arith.constant dense<3> : tensor<16xi32>
%p3 = tt.addptr %rows, %c3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v0 = tt.load %p0, %m, %z : tensor<16x!tt.ptr<f32>>
%v1 = tt.load %p1, %m, %one : tensor<16x!tt.ptr<f32>>
%v2 = tt.load %p2, %m, %z : tensor<16x!tt.ptr<f32>>
%v3 = tt.load %p3, %m, %z : tensor<16x!tt.ptr<f32>>
%sum1 = arith.addf %v0, %v1 : tensor<16xf32>
%sum2 = arith.addf %sum1, %v2 : tensor<16xf32>
%sum3 = arith.addf %sum2, %v3 : tensor<16xf32>
tt.store %outs, %sum3, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
// -----
// CHECK-LABEL: @stride3(
// CHECK-NOT: tensor<16x4x!tt.ptr<f32>>
// CHECK: tt.return

module { tt.func public @stride3(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<3> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c2 = arith.constant dense<2> : tensor<16xi32>
%p2 = tt.addptr %rows, %c2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v0 = tt.load %p0, %m, %z : tensor<16x!tt.ptr<f32>>
%v1 = tt.load %p1, %m, %z : tensor<16x!tt.ptr<f32>>
%v2 = tt.load %p2, %m, %z : tensor<16x!tt.ptr<f32>>
%sum1 = arith.addf %v0, %v1 : tensor<16xf32>
%sum2 = arith.addf %sum1, %v2 : tensor<16xf32>
tt.store %outs, %sum2, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
// -----
// CHECK-LABEL: @checked_update(
// CHECK-NOT: arith.cmpi ne
// CHECK: tt.load {{.*}} : tensor<16x4x!tt.ptr<f32>>
// CHECK: tt.store {{.*}} : tensor<16x4x!tt.ptr<f32>>

module { tt.func public @checked_update(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>, %flag: i1) attributes {ascend.experimental_checked_full_rows} {
%i = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
%w = arith.constant dense<4> : tensor<16xi32>
%idx = arith.muli %i, %w : tensor<16xi32>
%base = tt.splat %p : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%rows = tt.addptr %base, %idx : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%out = tt.splat %q : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
%outs = tt.addptr %out, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%m = tt.splat %flag : i1 -> tensor<16xi1>
%false = arith.constant dense<false> : tensor<16xi1>
%z = arith.constant dense<0.0> : tensor<16xf32>
%one = arith.constant dense<1.0> : tensor<16xf32>
%c0 = arith.constant dense<0> : tensor<16xi32>
%p0 = tt.addptr %rows, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c1 = arith.constant dense<1> : tensor<16xi32>
%p1 = tt.addptr %rows, %c1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c2 = arith.constant dense<2> : tensor<16xi32>
%p2 = tt.addptr %rows, %c2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%c3 = arith.constant dense<3> : tensor<16xi32>
%p3 = tt.addptr %rows, %c3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
%v1 = tt.load %p1, %m, %z : tensor<16x!tt.ptr<f32>>
%v2 = tt.load %p2, %m, %z : tensor<16x!tt.ptr<f32>>
%v3 = tt.load %p3, %m, %z : tensor<16x!tt.ptr<f32>>
tt.store %p0, %v1, %m : tensor<16x!tt.ptr<f32>>
tt.store %p1, %v2, %m : tensor<16x!tt.ptr<f32>>
tt.store %p2, %v3, %m : tensor<16x!tt.ptr<f32>>
tt.store %p3, %z, %m : tensor<16x!tt.ptr<f32>>
tt.return
}}
