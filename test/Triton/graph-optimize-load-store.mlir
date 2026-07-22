// RUN: triton-opt %s --verify-each -graph-optimize -o - | FileCheck %s --implicit-check-not=tensor.insert_slice

// CHECK-LABEL: tt.func @transpose_square_load_store(
// The original destination root is immediately followed by the rebuilt load
// root.  The store root is separately inserted before its original store so a
// late scalar store base remains in SSA scope.
// CHECK: %[[ORIGINAL_DST_BASE:.*]] = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
// CHECK-NEXT: %[[ORIGINAL_DST_POINTER:.*]] = tt.addptr %[[ORIGINAL_DST_BASE]], %{{.*}} : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
// The rebuilt source map is original(source) composed with P=[1, 0].  The
// first term is therefore the original column provenance on output axis 0.
// CHECK-NEXT: %[[SRC_NEW_AXIS0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK-NEXT: %[[SRC_NEW_AXIS0_EXPANDED:.*]] = tt.expand_dims %[[SRC_NEW_AXIS0]] {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
// CHECK-NEXT: %[[SRC_NEW_STRIDE0:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[SRC_NEW_STRIDE0_SPLAT:.*]] = tt.splat %[[SRC_NEW_STRIDE0]] : i32 -> tensor<4x1xi32>
// CHECK-NEXT: %[[SRC_NEW_TERM0:.*]] = arith.muli %[[SRC_NEW_AXIS0_EXPANDED]], %[[SRC_NEW_STRIDE0_SPLAT]] : tensor<4x1xi32>
// CHECK-NEXT: %[[SRC_NEW_TERM0_BROADCAST:.*]] = tt.broadcast %[[SRC_NEW_TERM0]] : tensor<4x1xi32> -> tensor<4x4xi32>
// The second term is the original row provenance on output axis 1.
// CHECK-NEXT: %[[SRC_NEW_AXIS1:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK-NEXT: %[[SRC_NEW_AXIS1_EXPANDED:.*]] = tt.expand_dims %[[SRC_NEW_AXIS1]] {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
// CHECK-NEXT: %[[SRC_NEW_STRIDE1:.*]] = arith.constant 4 : i32
// CHECK-NEXT: %[[SRC_NEW_STRIDE1_SPLAT:.*]] = tt.splat %[[SRC_NEW_STRIDE1]] : i32 -> tensor<1x4xi32>
// CHECK-NEXT: %[[SRC_NEW_TERM1:.*]] = arith.muli %[[SRC_NEW_AXIS1_EXPANDED]], %[[SRC_NEW_STRIDE1_SPLAT]] : tensor<1x4xi32>
// CHECK-NEXT: %[[SRC_NEW_TERM1_BROADCAST:.*]] = tt.broadcast %[[SRC_NEW_TERM1]] : tensor<1x4xi32> -> tensor<4x4xi32>
// CHECK-NEXT: %[[SRC_NEW_OFFSETS:.*]] = arith.addi %[[SRC_NEW_TERM0_BROADCAST]], %[[SRC_NEW_TERM1_BROADCAST]] : tensor<4x4xi32>
// CHECK-NEXT: %[[SRC_NEW_BASE:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
// CHECK-NEXT: %[[SRC_NEW_POINTER:.*]] = tt.addptr %[[SRC_NEW_BASE]], %[[SRC_NEW_OFFSETS]] : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
// CHECK-NEXT: %[[LOADED:.*]] = tt.load %[[SRC_NEW_POINTER]] : tensor<4x4x!tt.ptr<f32>>
// CHECK-NEXT: %[[NEGATED:.*]] = arith.negf %[[LOADED]] : tensor<4x4xf32>
// CHECK-NEXT: %[[DST_NEW_AXIS0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK-NEXT: %[[DST_NEW_AXIS0_EXPANDED:.*]] = tt.expand_dims %[[DST_NEW_AXIS0]] {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
// CHECK-NEXT: %[[DST_NEW_STRIDE0:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[DST_NEW_STRIDE0_SPLAT:.*]] = tt.splat %[[DST_NEW_STRIDE0]] : i32 -> tensor<4x1xi32>
// CHECK-NEXT: %[[DST_NEW_TERM0:.*]] = arith.muli %[[DST_NEW_AXIS0_EXPANDED]], %[[DST_NEW_STRIDE0_SPLAT]] : tensor<4x1xi32>
// CHECK-NEXT: %[[DST_NEW_TERM0_BROADCAST:.*]] = tt.broadcast %[[DST_NEW_TERM0]] : tensor<4x1xi32> -> tensor<4x4xi32>
// CHECK-NEXT: %[[DST_NEW_AXIS1:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK-NEXT: %[[DST_NEW_AXIS1_EXPANDED:.*]] = tt.expand_dims %[[DST_NEW_AXIS1]] {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
// CHECK-NEXT: %[[DST_NEW_STRIDE1:.*]] = arith.constant 4 : i32
// CHECK-NEXT: %[[DST_NEW_STRIDE1_SPLAT:.*]] = tt.splat %[[DST_NEW_STRIDE1]] : i32 -> tensor<1x4xi32>
// CHECK-NEXT: %[[DST_NEW_TERM1:.*]] = arith.muli %[[DST_NEW_AXIS1_EXPANDED]], %[[DST_NEW_STRIDE1_SPLAT]] : tensor<1x4xi32>
// CHECK-NEXT: %[[DST_NEW_TERM1_BROADCAST:.*]] = tt.broadcast %[[DST_NEW_TERM1]] : tensor<1x4xi32> -> tensor<4x4xi32>
// CHECK-NEXT: %[[DST_NEW_OFFSETS:.*]] = arith.addi %[[DST_NEW_TERM0_BROADCAST]], %[[DST_NEW_TERM1_BROADCAST]] : tensor<4x4xi32>
// CHECK-NEXT: %[[DST_NEW_BASE:.*]] = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
// CHECK-NEXT: %[[DST_NEW_POINTER:.*]] = tt.addptr %[[DST_NEW_BASE]], %[[DST_NEW_OFFSETS]] : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
// CHECK-NEXT: tt.store %[[DST_NEW_POINTER]], %[[NEGATED]] : tensor<4x4x!tt.ptr<f32>>
// CHECK-NOT: tt.trans
tt.func @transpose_square_load_store(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>) {
  %source_row = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %source_column = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %source_row_expanded = tt.expand_dims %source_row {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %source_column_expanded = tt.expand_dims %source_column {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %source_row_stride_value = arith.constant 4 : i32
  %source_column_stride_value = arith.constant 1 : i32
  %source_row_stride = tt.splat %source_row_stride_value : i32 -> tensor<4x1xi32>
  %source_column_stride = tt.splat %source_column_stride_value : i32 -> tensor<1x4xi32>
  %source_row_offsets = arith.muli %source_row_expanded, %source_row_stride : tensor<4x1xi32>
  %source_column_offsets = arith.muli %source_column_expanded, %source_column_stride : tensor<1x4xi32>
  %source_row_broadcast = tt.broadcast %source_row_offsets : tensor<4x1xi32> -> tensor<4x4xi32>
  %source_column_broadcast = tt.broadcast %source_column_offsets : tensor<1x4xi32> -> tensor<4x4xi32>
  %source_offsets = arith.addi %source_row_broadcast, %source_column_broadcast : tensor<4x4xi32>
  %source_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
  %source_addresses = tt.addptr %source_base_splat, %source_offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>

  %destination_row = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %destination_column = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %destination_row_expanded = tt.expand_dims %destination_row {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %destination_column_expanded = tt.expand_dims %destination_column {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %destination_row_stride_value = arith.constant 4 : i32
  %destination_column_stride_value = arith.constant 1 : i32
  %destination_row_stride = tt.splat %destination_row_stride_value : i32 -> tensor<4x1xi32>
  %destination_column_stride = tt.splat %destination_column_stride_value : i32 -> tensor<1x4xi32>
  %destination_row_offsets = arith.muli %destination_row_expanded, %destination_row_stride : tensor<4x1xi32>
  %destination_column_offsets = arith.muli %destination_column_expanded, %destination_column_stride : tensor<1x4xi32>
  %destination_row_broadcast = tt.broadcast %destination_row_offsets : tensor<4x1xi32> -> tensor<4x4xi32>
  %destination_column_broadcast = tt.broadcast %destination_column_offsets : tensor<1x4xi32> -> tensor<4x4xi32>
  %destination_offsets = arith.addi %destination_row_broadcast, %destination_column_broadcast : tensor<4x4xi32>
  %destination_base_splat = tt.splat %destination : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
  %destination_addresses = tt.addptr %destination_base_splat, %destination_offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>

  %loaded = tt.load %source_addresses : tensor<4x4x!tt.ptr<f32>>
  %negated = arith.negf %loaded : tensor<4x4xf32>
  tt.store %destination_addresses, %negated : tensor<4x4x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_rank3_non_self_inverse(
// CHECK: %[[RANK3_LOADED:.*]] = tt.load %{{.*}} : tensor<2x2x2x!tt.ptr<f32>>
// CHECK-NEXT: %[[RANK3_NEGATED:.*]] = arith.negf %[[RANK3_LOADED]] : tensor<2x2x2xf32>
// CHECK-NEXT: tt.store %{{.*}}, %[[RANK3_NEGATED]] : tensor<2x2x2x!tt.ptr<f32>>
tt.func @reject_rank3_non_self_inverse(%base: !tt.ptr<f32>) {
  %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis0_expand1 = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %axis0_expand2 = tt.expand_dims %axis0_expand1 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
  %axis1_expand0 = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %axis1_expand2 = tt.expand_dims %axis1_expand0 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32>
  %axis2_expand0a = tt.expand_dims %axis2 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %axis2_expand0b = tt.expand_dims %axis2_expand0a {axis = 0 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32>
  %stride0_value = arith.constant 4 : i32
  %stride1_value = arith.constant 2 : i32
  %stride2_value = arith.constant 1 : i32
  %stride0 = tt.splat %stride0_value : i32 -> tensor<2x1x1xi32>
  %stride1 = tt.splat %stride1_value : i32 -> tensor<1x2x1xi32>
  %stride2 = tt.splat %stride2_value : i32 -> tensor<1x1x2xi32>
  %term0 = arith.muli %axis0_expand2, %stride0 : tensor<2x1x1xi32>
  %term1 = arith.muli %axis1_expand2, %stride1 : tensor<1x2x1xi32>
  %term2 = arith.muli %axis2_expand0b, %stride2 : tensor<1x1x2xi32>
  %term0_broadcast = tt.broadcast %term0 : tensor<2x1x1xi32> -> tensor<2x2x2xi32>
  %term1_broadcast = tt.broadcast %term1 : tensor<1x2x1xi32> -> tensor<2x2x2xi32>
  %term2_broadcast = tt.broadcast %term2 : tensor<1x1x2xi32> -> tensor<2x2x2xi32>
  %offsets0 = arith.addi %term0_broadcast, %term1_broadcast : tensor<2x2x2xi32>
  %offsets = arith.addi %offsets0, %term2_broadcast : tensor<2x2x2xi32>
  %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<2x2x2x!tt.ptr<f32>>
  %addresses = tt.addptr %base_splat, %offsets : tensor<2x2x2x!tt.ptr<f32>>, tensor<2x2x2xi32>
  %loaded = tt.load %addresses : tensor<2x2x2x!tt.ptr<f32>>
  %negated = arith.negf %loaded : tensor<2x2x2xf32>
  tt.store %addresses, %negated : tensor<2x2x2x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_dynamic_stride(
// CHECK: tt.splat %{{.*}} : i32 -> tensor<4xi32>
// CHECK: arith.muli
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_dynamic_stride(%base: !tt.ptr<i32>, %stride: i32) {
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %stride_splat = tt.splat %stride : i32 -> tensor<4xi32>
  %offsets = arith.muli %range, %stride_splat : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %addresses = tt.addptr %base_splat, %offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %loaded = tt.load %addresses : tensor<4x!tt.ptr<i32>>
  tt.store %addresses, %loaded : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_equal_stride(
// CHECK: arith.muli
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: tt.store {{.*}}, {{.*}} : tensor<2x2x!tt.ptr<i32>>
tt.func @reject_equal_stride(%base: !tt.ptr<i32>) {
  %row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %row_expanded = tt.expand_dims %row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %column_expanded = tt.expand_dims %column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %row_broadcast = tt.broadcast %row_expanded : tensor<2x1xi32> -> tensor<2x2xi32>
  %column_broadcast = tt.broadcast %column_expanded : tensor<1x2xi32> -> tensor<2x2xi32>
  %one = arith.constant 1 : i32
  %row_stride = tt.splat %one : i32 -> tensor<2x2xi32>
  %column_stride = tt.splat %one : i32 -> tensor<2x2xi32>
  %row_offsets = arith.muli %row_broadcast, %row_stride : tensor<2x2xi32>
  %column_offsets = arith.muli %column_broadcast, %column_stride : tensor<2x2xi32>
  %offsets = arith.addi %row_offsets, %column_offsets : tensor<2x2xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<2x2x!tt.ptr<i32>>
  %addresses = tt.addptr %base_splat, %offsets : tensor<2x2x!tt.ptr<i32>>, tensor<2x2xi32>
  %loaded = tt.load %addresses : tensor<2x2x!tt.ptr<i32>>
  tt.store %addresses, %loaded : tensor<2x2x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_negative_stride(
// CHECK: arith.constant -1 : i32
// CHECK: arith.muli
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_negative_stride(%base: !tt.ptr<i32>) {
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %negative_one = arith.constant -1 : i32
  %stride_splat = tt.splat %negative_one : i32 -> tensor<4xi32>
  %offsets = arith.muli %range, %stride_splat : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %addresses = tt.addptr %base_splat, %offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %loaded = tt.load %addresses : tensor<4x!tt.ptr<i32>>
  tt.store %addresses, %loaded : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_zero_stride(
// CHECK: arith.constant 0 : i32
// CHECK: arith.muli
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_zero_stride(%base: !tt.ptr<i32>) {
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %zero = arith.constant 0 : i32
  %stride_splat = tt.splat %zero : i32 -> tensor<4xi32>
  %offsets = arith.muli %range, %stride_splat : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %addresses = tt.addptr %base_splat, %offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %loaded = tt.load %addresses : tensor<4x!tt.ptr<i32>>
  tt.store %addresses, %loaded : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_masked_store(
// CHECK: tt.store {{.*}}, {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_masked_store(%base: !tt.ptr<i32>, %mask: tensor<4xi1>) {
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %addresses = tt.addptr %base_splat, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %value = arith.constant dense<0> : tensor<4xi32>
  tt.store %addresses, %value, %mask : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_volatile_load(
// CHECK: tt.load {{.*}} {isVolatile = true} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_volatile_load(%base: !tt.ptr<i32>) {
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %addresses = tt.addptr %base_splat, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %loaded = tt.load %addresses {isVolatile = true} : tensor<4x!tt.ptr<i32>>
  tt.store %addresses, %loaded : tensor<4x!tt.ptr<i32>>
  tt.return
}

// Structural access-analysis fixture only: this tensor-pointer/boundary form
// is rejected before the V1 load -> negf -> store candidate matcher.
// CHECK-LABEL: tt.func @reject_boundary_checked_load(
// CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<4xi32>>
// CHECK: tt.store {{.*}}, {{.*}} : !tt.ptr<tensor<4xi32>>
tt.func @reject_boundary_checked_load(%source: !tt.ptr<tensor<4xi32>>, %destination: !tt.ptr<tensor<4xi32>>) {
  %value = tt.load %source {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<4xi32>>
  tt.store %destination, %value : !tt.ptr<tensor<4xi32>>
  tt.return
}

tt.func private @opaque_effect() {
  tt.return
}

// CHECK-LABEL: tt.func @reject_call_and_barrier_between_accesses(
// CHECK: tt.load
// CHECK: tt.call @opaque_effect() : () -> ()
// CHECK: gpu.barrier
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_call_and_barrier_between_accesses(%base: !tt.ptr<i32>) {
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %addresses = tt.addptr %base_splat, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %loaded = tt.load %addresses : tensor<4x!tt.ptr<i32>>
  tt.call @opaque_effect() : () -> ()
  gpu.barrier
  tt.store %addresses, %loaded : tensor<4x!tt.ptr<i32>>
  tt.return
}

// The store base is deliberately produced after the load.  A rewrite must
// build the load replacement before the load and the store replacement before
// the store; placing both before the load would use %late_destination before
// its definition and violate SSA dominance.
// CHECK-LABEL: tt.func @transpose_square_late_store_base(
// CHECK: %[[LATE_SOURCE_ORIGINAL:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
// CHECK: %[[LATE_NEW_SOURCE_RANGE:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK: %[[LATE_NEW_SOURCE_POINTER:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
// CHECK: %[[LATE_LOADED:.*]] = tt.load %[[LATE_NEW_SOURCE_POINTER]] : tensor<4x4x!tt.ptr<f32>>
// CHECK-NEXT: %[[LATE_NEGATED:.*]] = arith.negf %[[LATE_LOADED]] : tensor<4x4xf32>
// CHECK-NEXT: %[[LATE_ZERO:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[LATE_BASE:.*]] = tt.addptr %arg1, %[[LATE_ZERO]] : !tt.ptr<f32>, i32
// CHECK: %[[LATE_DESTINATION_ORIGINAL:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
// CHECK: %[[LATE_NEW_DESTINATION_RANGE:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK: %[[LATE_NEW_DESTINATION_POINTER:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
// CHECK: tt.store %[[LATE_NEW_DESTINATION_POINTER]], %[[LATE_NEGATED]] : tensor<4x4x!tt.ptr<f32>>
tt.func @transpose_square_late_store_base(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>) {
  %source_row = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %source_column = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %source_row_expanded = tt.expand_dims %source_row {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %source_column_expanded = tt.expand_dims %source_column {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %source_row_stride_value = arith.constant 4 : i32
  %source_column_stride_value = arith.constant 1 : i32
  %source_row_stride = tt.splat %source_row_stride_value : i32 -> tensor<4x1xi32>
  %source_column_stride = tt.splat %source_column_stride_value : i32 -> tensor<1x4xi32>
  %source_row_offsets = arith.muli %source_row_expanded, %source_row_stride : tensor<4x1xi32>
  %source_column_offsets = arith.muli %source_column_expanded, %source_column_stride : tensor<1x4xi32>
  %source_row_broadcast = tt.broadcast %source_row_offsets : tensor<4x1xi32> -> tensor<4x4xi32>
  %source_column_broadcast = tt.broadcast %source_column_offsets : tensor<1x4xi32> -> tensor<4x4xi32>
  %source_offsets = arith.addi %source_row_broadcast, %source_column_broadcast : tensor<4x4xi32>
  %source_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
  %source_addresses = tt.addptr %source_base_splat, %source_offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>

  %loaded = tt.load %source_addresses : tensor<4x4x!tt.ptr<f32>>
  %negated = arith.negf %loaded : tensor<4x4xf32>
  %zero = arith.constant 0 : i32
  %late_destination = tt.addptr %destination, %zero : !tt.ptr<f32>, i32
  %destination_row = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %destination_column = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %destination_row_expanded = tt.expand_dims %destination_row {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %destination_column_expanded = tt.expand_dims %destination_column {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %destination_row_stride_value = arith.constant 4 : i32
  %destination_column_stride_value = arith.constant 1 : i32
  %destination_row_stride = tt.splat %destination_row_stride_value : i32 -> tensor<4x1xi32>
  %destination_column_stride = tt.splat %destination_column_stride_value : i32 -> tensor<1x4xi32>
  %destination_row_offsets = arith.muli %destination_row_expanded, %destination_row_stride : tensor<4x1xi32>
  %destination_column_offsets = arith.muli %destination_column_expanded, %destination_column_stride : tensor<1x4xi32>
  %destination_row_broadcast = tt.broadcast %destination_row_offsets : tensor<4x1xi32> -> tensor<4x4xi32>
  %destination_column_broadcast = tt.broadcast %destination_column_offsets : tensor<1x4xi32> -> tensor<4x4xi32>
  %destination_offsets = arith.addi %destination_row_broadcast, %destination_column_broadcast : tensor<4x4xi32>
  %destination_base_splat = tt.splat %late_destination : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
  %destination_addresses = tt.addptr %destination_base_splat, %destination_offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
  tt.store %destination_addresses, %negated : tensor<4x4x!tt.ptr<f32>>
  tt.return
}

// Overflow flags carry poison semantics.  The V1 rebuild uses ordinary
// arithmetic, so a candidate with any flagged root or term must remain
// untouched rather than silently dropping nsw/nuw.
// CHECK-LABEL: tt.func @reject_overflow_flagged_access(
// CHECK: arith.muli %{{.*}}, %{{.*}} overflow<nsw> : tensor<2x1xi32>
// CHECK: arith.muli %{{.*}}, %{{.*}} overflow<nuw> : tensor<1x2xi32>
// CHECK: arith.addi %{{.*}}, %{{.*}} overflow<nsw> : tensor<2x2xi32>
// CHECK: %[[FLAGGED_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK: %[[FLAGGED_DESTINATION:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: tt.make_range
// CHECK: %[[FLAGGED_LOADED:.*]] = tt.load %[[FLAGGED_SOURCE]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-NEXT: %[[FLAGGED_NEGATED:.*]] = arith.negf %[[FLAGGED_LOADED]] : tensor<2x2xf32>
// CHECK-NOT: tt.make_range
// CHECK: tt.store %[[FLAGGED_DESTINATION]], %[[FLAGGED_NEGATED]] : tensor<2x2x!tt.ptr<f32>>
tt.func @reject_overflow_flagged_access(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>) {
  %source_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_row_expanded = tt.expand_dims %source_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %source_column_expanded = tt.expand_dims %source_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %two = arith.constant 2 : i32
  %one = arith.constant 1 : i32
  %source_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %source_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %source_row_offsets = arith.muli %source_row_expanded, %source_row_stride overflow<nsw> : tensor<2x1xi32>
  %source_column_offsets = arith.muli %source_column_expanded, %source_column_stride overflow<nuw> : tensor<1x2xi32>
  %source_row_broadcast = tt.broadcast %source_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %source_column_broadcast = tt.broadcast %source_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %source_offsets = arith.addi %source_row_broadcast, %source_column_broadcast overflow<nsw> : tensor<2x2xi32>
  %source_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %source_addresses = tt.addptr %source_base_splat, %source_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %destination_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_row_expanded = tt.expand_dims %destination_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %destination_column_expanded = tt.expand_dims %destination_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %destination_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %destination_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %destination_row_offsets = arith.muli %destination_row_expanded, %destination_row_stride : tensor<2x1xi32>
  %destination_column_offsets = arith.muli %destination_column_expanded, %destination_column_stride : tensor<1x2xi32>
  %destination_row_broadcast = tt.broadcast %destination_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %destination_column_broadcast = tt.broadcast %destination_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %destination_offsets = arith.addi %destination_row_broadcast, %destination_column_broadcast : tensor<2x2xi32>
  %destination_base_splat = tt.splat %destination : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %destination_addresses = tt.addptr %destination_base_splat, %destination_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %loaded = tt.load %source_addresses : tensor<2x2x!tt.ptr<f32>>
  %negated = arith.negf %loaded : tensor<2x2xf32>
  tt.store %destination_addresses, %negated : tensor<2x2x!tt.ptr<f32>>
  tt.return
}

// Each load below has the complete V1 shape: independent unencoded square
// rank-2 pointer DAGs and load -> negf -> store.  Only the named source
// stride differs, so these checks exercise the rule's actual candidate gate
// rather than a generic rank-1 proof fixture.
// CHECK-LABEL: tt.func @reject_complete_bad_strides(
// CHECK: %[[BAD_STRIDES_DESTINATION:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK: %[[DYNAMIC_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: tt.make_range
// CHECK: %[[DYNAMIC_LOADED:.*]] = tt.load %[[DYNAMIC_SOURCE]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-NEXT: %[[DYNAMIC_NEGATED:.*]] = arith.negf %[[DYNAMIC_LOADED]] : tensor<2x2xf32>
// CHECK-NOT: tt.make_range
// CHECK: tt.store %[[BAD_STRIDES_DESTINATION]], %[[DYNAMIC_NEGATED]] : tensor<2x2x!tt.ptr<f32>>
// CHECK: %[[EQUAL_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: tt.make_range
// CHECK: %[[EQUAL_LOADED:.*]] = tt.load %[[EQUAL_SOURCE]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-NEXT: %[[EQUAL_NEGATED:.*]] = arith.negf %[[EQUAL_LOADED]] : tensor<2x2xf32>
// CHECK-NOT: tt.make_range
// CHECK: tt.store %[[BAD_STRIDES_DESTINATION]], %[[EQUAL_NEGATED]] : tensor<2x2x!tt.ptr<f32>>
// CHECK: %[[NEGATIVE_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: tt.make_range
// CHECK: %[[NEGATIVE_LOADED:.*]] = tt.load %[[NEGATIVE_SOURCE]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-NEXT: %[[NEGATIVE_NEGATED:.*]] = arith.negf %[[NEGATIVE_LOADED]] : tensor<2x2xf32>
// CHECK-NOT: tt.make_range
// CHECK: tt.store %[[BAD_STRIDES_DESTINATION]], %[[NEGATIVE_NEGATED]] : tensor<2x2x!tt.ptr<f32>>
// CHECK: %[[ZERO_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: tt.make_range
// CHECK: %[[ZERO_LOADED:.*]] = tt.load %[[ZERO_SOURCE]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-NEXT: %[[ZERO_NEGATED:.*]] = arith.negf %[[ZERO_LOADED]] : tensor<2x2xf32>
// CHECK-NOT: tt.make_range
// CHECK: tt.store %[[BAD_STRIDES_DESTINATION]], %[[ZERO_NEGATED]] : tensor<2x2x!tt.ptr<f32>>
tt.func @reject_complete_bad_strides(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>, %dynamic_stride: i32) {
  %destination_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_row_expanded = tt.expand_dims %destination_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %destination_column_expanded = tt.expand_dims %destination_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %two = arith.constant 2 : i32
  %one = arith.constant 1 : i32
  %destination_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %destination_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %destination_row_offsets = arith.muli %destination_row_expanded, %destination_row_stride : tensor<2x1xi32>
  %destination_column_offsets = arith.muli %destination_column_expanded, %destination_column_stride : tensor<1x2xi32>
  %destination_row_broadcast = tt.broadcast %destination_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %destination_column_broadcast = tt.broadcast %destination_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %destination_offsets = arith.addi %destination_row_broadcast, %destination_column_broadcast : tensor<2x2xi32>
  %destination_base_splat = tt.splat %destination : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %destination_addresses = tt.addptr %destination_base_splat, %destination_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %dynamic_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %dynamic_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %dynamic_row_expanded = tt.expand_dims %dynamic_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %dynamic_column_expanded = tt.expand_dims %dynamic_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %dynamic_row_stride = tt.splat %dynamic_stride : i32 -> tensor<2x1xi32>
  %dynamic_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %dynamic_row_offsets = arith.muli %dynamic_row_expanded, %dynamic_row_stride : tensor<2x1xi32>
  %dynamic_column_offsets = arith.muli %dynamic_column_expanded, %dynamic_column_stride : tensor<1x2xi32>
  %dynamic_row_broadcast = tt.broadcast %dynamic_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %dynamic_column_broadcast = tt.broadcast %dynamic_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %dynamic_offsets = arith.addi %dynamic_row_broadcast, %dynamic_column_broadcast : tensor<2x2xi32>
  %dynamic_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %dynamic_addresses = tt.addptr %dynamic_base_splat, %dynamic_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  %dynamic_loaded = tt.load %dynamic_addresses : tensor<2x2x!tt.ptr<f32>>
  %dynamic_negated = arith.negf %dynamic_loaded : tensor<2x2xf32>
  tt.store %destination_addresses, %dynamic_negated : tensor<2x2x!tt.ptr<f32>>

  %equal_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %equal_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %equal_row_expanded = tt.expand_dims %equal_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %equal_column_expanded = tt.expand_dims %equal_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %equal_row_stride = tt.splat %one : i32 -> tensor<2x1xi32>
  %equal_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %equal_row_offsets = arith.muli %equal_row_expanded, %equal_row_stride : tensor<2x1xi32>
  %equal_column_offsets = arith.muli %equal_column_expanded, %equal_column_stride : tensor<1x2xi32>
  %equal_row_broadcast = tt.broadcast %equal_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %equal_column_broadcast = tt.broadcast %equal_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %equal_offsets = arith.addi %equal_row_broadcast, %equal_column_broadcast : tensor<2x2xi32>
  %equal_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %equal_addresses = tt.addptr %equal_base_splat, %equal_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  %equal_loaded = tt.load %equal_addresses : tensor<2x2x!tt.ptr<f32>>
  %equal_negated = arith.negf %equal_loaded : tensor<2x2xf32>
  tt.store %destination_addresses, %equal_negated : tensor<2x2x!tt.ptr<f32>>

  %negative_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %negative_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %negative_row_expanded = tt.expand_dims %negative_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %negative_column_expanded = tt.expand_dims %negative_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %negative_two = arith.constant -2 : i32
  %negative_row_stride = tt.splat %negative_two : i32 -> tensor<2x1xi32>
  %negative_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %negative_row_offsets = arith.muli %negative_row_expanded, %negative_row_stride : tensor<2x1xi32>
  %negative_column_offsets = arith.muli %negative_column_expanded, %negative_column_stride : tensor<1x2xi32>
  %negative_row_broadcast = tt.broadcast %negative_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %negative_column_broadcast = tt.broadcast %negative_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %negative_offsets = arith.addi %negative_row_broadcast, %negative_column_broadcast : tensor<2x2xi32>
  %negative_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %negative_addresses = tt.addptr %negative_base_splat, %negative_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  %negative_loaded = tt.load %negative_addresses : tensor<2x2x!tt.ptr<f32>>
  %negative_negated = arith.negf %negative_loaded : tensor<2x2xf32>
  tt.store %destination_addresses, %negative_negated : tensor<2x2x!tt.ptr<f32>>

  %zero_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %zero_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %zero_row_expanded = tt.expand_dims %zero_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %zero_column_expanded = tt.expand_dims %zero_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %zero_value = arith.constant 0 : i32
  %zero_row_stride = tt.splat %zero_value : i32 -> tensor<2x1xi32>
  %zero_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %zero_row_offsets = arith.muli %zero_row_expanded, %zero_row_stride : tensor<2x1xi32>
  %zero_column_offsets = arith.muli %zero_column_expanded, %zero_column_stride : tensor<1x2xi32>
  %zero_row_broadcast = tt.broadcast %zero_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %zero_column_broadcast = tt.broadcast %zero_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %zero_offsets = arith.addi %zero_row_broadcast, %zero_column_broadcast : tensor<2x2xi32>
  %zero_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %zero_addresses = tt.addptr %zero_base_splat, %zero_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  %zero_loaded = tt.load %zero_addresses : tensor<2x2x!tt.ptr<f32>>
  %zero_negated = arith.negf %zero_loaded : tensor<2x2xf32>
  tt.store %destination_addresses, %zero_negated : tensor<2x2x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_complete_mask_and_volatile(
// CHECK: %[[MASK_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK: %[[MASK_DESTINATION:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: tt.make_range
// CHECK: %[[MASK_LOADED:.*]] = tt.load %[[MASK_SOURCE]], %{{.*}} : tensor<2x2x!tt.ptr<f32>>
// CHECK-NEXT: %[[MASK_NEGATED:.*]] = arith.negf %[[MASK_LOADED]] : tensor<2x2xf32>
// CHECK-NOT: tt.make_range
// CHECK: tt.store %[[MASK_DESTINATION]], %[[MASK_NEGATED]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-NOT: tt.make_range
// CHECK: %[[VOLATILE_LOADED:.*]] = tt.load %[[MASK_SOURCE]] {isVolatile = true} : tensor<2x2x!tt.ptr<f32>>
// CHECK-NEXT: %[[VOLATILE_NEGATED:.*]] = arith.negf %[[VOLATILE_LOADED]] : tensor<2x2xf32>
// CHECK-NOT: tt.make_range
// CHECK: tt.store %[[MASK_DESTINATION]], %[[VOLATILE_NEGATED]] : tensor<2x2x!tt.ptr<f32>>
tt.func @reject_complete_mask_and_volatile(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>, %mask: tensor<2x2xi1>) {
  %source_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_row_expanded = tt.expand_dims %source_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %source_column_expanded = tt.expand_dims %source_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %two = arith.constant 2 : i32
  %one = arith.constant 1 : i32
  %source_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %source_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %source_row_offsets = arith.muli %source_row_expanded, %source_row_stride : tensor<2x1xi32>
  %source_column_offsets = arith.muli %source_column_expanded, %source_column_stride : tensor<1x2xi32>
  %source_row_broadcast = tt.broadcast %source_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %source_column_broadcast = tt.broadcast %source_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %source_offsets = arith.addi %source_row_broadcast, %source_column_broadcast : tensor<2x2xi32>
  %source_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %source_addresses = tt.addptr %source_base_splat, %source_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %destination_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_row_expanded = tt.expand_dims %destination_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %destination_column_expanded = tt.expand_dims %destination_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %destination_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %destination_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %destination_row_offsets = arith.muli %destination_row_expanded, %destination_row_stride : tensor<2x1xi32>
  %destination_column_offsets = arith.muli %destination_column_expanded, %destination_column_stride : tensor<1x2xi32>
  %destination_row_broadcast = tt.broadcast %destination_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %destination_column_broadcast = tt.broadcast %destination_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %destination_offsets = arith.addi %destination_row_broadcast, %destination_column_broadcast : tensor<2x2xi32>
  %destination_base_splat = tt.splat %destination : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %destination_addresses = tt.addptr %destination_base_splat, %destination_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %masked_loaded = tt.load %source_addresses, %mask : tensor<2x2x!tt.ptr<f32>>
  %masked_negated = arith.negf %masked_loaded : tensor<2x2xf32>
  tt.store %destination_addresses, %masked_negated : tensor<2x2x!tt.ptr<f32>>
  %volatile_loaded = tt.load %source_addresses {isVolatile = true} : tensor<2x2x!tt.ptr<f32>>
  %volatile_negated = arith.negf %volatile_loaded : tensor<2x2xf32>
  tt.store %destination_addresses, %volatile_negated : tensor<2x2x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_complete_call_and_barrier_interval(
// CHECK: %[[EFFECT_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK: %[[EFFECT_DESTINATION:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: tt.make_range
// CHECK: %[[EFFECT_LOADED:.*]] = tt.load %[[EFFECT_SOURCE]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-NEXT: %[[EFFECT_NEGATED:.*]] = arith.negf %[[EFFECT_LOADED]] : tensor<2x2xf32>
// CHECK-NEXT: tt.call @opaque_effect() : () -> ()
// CHECK-NEXT: gpu.barrier
// CHECK-NOT: tt.make_range
// CHECK: tt.store %[[EFFECT_DESTINATION]], %[[EFFECT_NEGATED]] : tensor<2x2x!tt.ptr<f32>>
tt.func @reject_complete_call_and_barrier_interval(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>) {
  %source_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_row_expanded = tt.expand_dims %source_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %source_column_expanded = tt.expand_dims %source_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %two = arith.constant 2 : i32
  %one = arith.constant 1 : i32
  %source_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %source_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %source_row_offsets = arith.muli %source_row_expanded, %source_row_stride : tensor<2x1xi32>
  %source_column_offsets = arith.muli %source_column_expanded, %source_column_stride : tensor<1x2xi32>
  %source_row_broadcast = tt.broadcast %source_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %source_column_broadcast = tt.broadcast %source_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %source_offsets = arith.addi %source_row_broadcast, %source_column_broadcast : tensor<2x2xi32>
  %source_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %source_addresses = tt.addptr %source_base_splat, %source_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %destination_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_row_expanded = tt.expand_dims %destination_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %destination_column_expanded = tt.expand_dims %destination_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %destination_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %destination_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %destination_row_offsets = arith.muli %destination_row_expanded, %destination_row_stride : tensor<2x1xi32>
  %destination_column_offsets = arith.muli %destination_column_expanded, %destination_column_stride : tensor<1x2xi32>
  %destination_row_broadcast = tt.broadcast %destination_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %destination_column_broadcast = tt.broadcast %destination_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %destination_offsets = arith.addi %destination_row_broadcast, %destination_column_broadcast : tensor<2x2xi32>
  %destination_base_splat = tt.splat %destination : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %destination_addresses = tt.addptr %destination_base_splat, %destination_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %loaded = tt.load %source_addresses : tensor<2x2x!tt.ptr<f32>>
  %negated = arith.negf %loaded : tensor<2x2xf32>
  tt.call @opaque_effect() : () -> ()
  gpu.barrier
  tt.store %destination_addresses, %negated : tensor<2x2x!tt.ptr<f32>>
  tt.return
}

// Keep the value slice itself canonical and place an unrelated explicit
// transpose in the protected interval.  This reaches containsExplicitTranspose
// after the direct load -> negf -> store use checks have succeeded.
// CHECK-LABEL: tt.func @reject_explicit_transpose_interval(
// CHECK: %[[INTERVAL_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK: %[[INTERVAL_DESTINATION:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: tt.make_range
// CHECK: %[[INTERVAL_LOADED:.*]] = tt.load %[[INTERVAL_SOURCE]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-NEXT: %[[INTERVAL_NEGATED:.*]] = arith.negf %[[INTERVAL_LOADED]] : tensor<2x2xf32>
// CHECK-NEXT: %{{.*}} = tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<2x2xf32> -> tensor<2x2xf32>
// CHECK-NOT: tt.make_range
// CHECK: tt.store %[[INTERVAL_DESTINATION]], %[[INTERVAL_NEGATED]] : tensor<2x2x!tt.ptr<f32>>
tt.func @reject_explicit_transpose_interval(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>, %unrelated: tensor<2x2xf32>) {
  %source_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_row_expanded = tt.expand_dims %source_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %source_column_expanded = tt.expand_dims %source_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %two = arith.constant 2 : i32
  %one = arith.constant 1 : i32
  %source_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %source_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %source_row_offsets = arith.muli %source_row_expanded, %source_row_stride : tensor<2x1xi32>
  %source_column_offsets = arith.muli %source_column_expanded, %source_column_stride : tensor<1x2xi32>
  %source_row_broadcast = tt.broadcast %source_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %source_column_broadcast = tt.broadcast %source_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %source_offsets = arith.addi %source_row_broadcast, %source_column_broadcast : tensor<2x2xi32>
  %source_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %source_addresses = tt.addptr %source_base_splat, %source_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %destination_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_row_expanded = tt.expand_dims %destination_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %destination_column_expanded = tt.expand_dims %destination_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %destination_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %destination_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %destination_row_offsets = arith.muli %destination_row_expanded, %destination_row_stride : tensor<2x1xi32>
  %destination_column_offsets = arith.muli %destination_column_expanded, %destination_column_stride : tensor<1x2xi32>
  %destination_row_broadcast = tt.broadcast %destination_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %destination_column_broadcast = tt.broadcast %destination_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %destination_offsets = arith.addi %destination_row_broadcast, %destination_column_broadcast : tensor<2x2xi32>
  %destination_base_splat = tt.splat %destination : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %destination_addresses = tt.addptr %destination_base_splat, %destination_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %loaded = tt.load %source_addresses : tensor<2x2x!tt.ptr<f32>>
  %negated = arith.negf %loaded : tensor<2x2xf32>
  %interval_transpose = tt.trans %unrelated {order = array<i32: 1, 0>} : tensor<2x2xf32> -> tensor<2x2xf32>
  tt.store %destination_addresses, %negated : tensor<2x2x!tt.ptr<f32>>
  tt.return
}

// These are otherwise complete V1 candidates.  Extra users of the loaded
// value, including a tt.trans or tensor slice, must reject the entire slice
// before any pointer reindexing is considered.
// CHECK-LABEL: tt.func @reject_external_user_and_transpose_slice(
// CHECK: %[[USER_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK: %[[USER_DESTINATION:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// CHECK-NOT: tt.make_range
// CHECK: %[[USER_LOADED:.*]] = tt.load %[[USER_SOURCE]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-NEXT: %{{.*}} = arith.addf %[[USER_LOADED]], %[[USER_LOADED]] : tensor<2x2xf32>
// CHECK-NEXT: %[[USER_NEGATED:.*]] = arith.negf %[[USER_LOADED]] : tensor<2x2xf32>
// CHECK-NOT: tt.make_range
// CHECK: tt.store %[[USER_DESTINATION]], %[[USER_NEGATED]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-NOT: tt.make_range
// CHECK: %[[SLICE_LOADED:.*]] = tt.load %[[USER_SOURCE]] : tensor<2x2x!tt.ptr<f32>>
// CHECK-NEXT: %{{.*}} = tt.trans %[[SLICE_LOADED]] {order = array<i32: 1, 0>} : tensor<2x2xf32> -> tensor<2x2xf32>
// CHECK-NEXT: %{{.*}} = tensor.extract_slice %[[SLICE_LOADED]][0, 0] [1, 2] [1, 1] : tensor<2x2xf32> to tensor<1x2xf32>
// CHECK-NEXT: %[[SLICE_NEGATED:.*]] = arith.negf %[[SLICE_LOADED]] : tensor<2x2xf32>
// CHECK-NOT: tt.make_range
// CHECK: tt.store %[[USER_DESTINATION]], %[[SLICE_NEGATED]] : tensor<2x2x!tt.ptr<f32>>
tt.func @reject_external_user_and_transpose_slice(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>) {
  %source_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_row_expanded = tt.expand_dims %source_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %source_column_expanded = tt.expand_dims %source_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %two = arith.constant 2 : i32
  %one = arith.constant 1 : i32
  %source_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %source_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %source_row_offsets = arith.muli %source_row_expanded, %source_row_stride : tensor<2x1xi32>
  %source_column_offsets = arith.muli %source_column_expanded, %source_column_stride : tensor<1x2xi32>
  %source_row_broadcast = tt.broadcast %source_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %source_column_broadcast = tt.broadcast %source_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %source_offsets = arith.addi %source_row_broadcast, %source_column_broadcast : tensor<2x2xi32>
  %source_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %source_addresses = tt.addptr %source_base_splat, %source_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %destination_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_row_expanded = tt.expand_dims %destination_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %destination_column_expanded = tt.expand_dims %destination_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %destination_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %destination_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %destination_row_offsets = arith.muli %destination_row_expanded, %destination_row_stride : tensor<2x1xi32>
  %destination_column_offsets = arith.muli %destination_column_expanded, %destination_column_stride : tensor<1x2xi32>
  %destination_row_broadcast = tt.broadcast %destination_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %destination_column_broadcast = tt.broadcast %destination_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %destination_offsets = arith.addi %destination_row_broadcast, %destination_column_broadcast : tensor<2x2xi32>
  %destination_base_splat = tt.splat %destination : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %destination_addresses = tt.addptr %destination_base_splat, %destination_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %external_loaded = tt.load %source_addresses : tensor<2x2x!tt.ptr<f32>>
  %external_user = arith.addf %external_loaded, %external_loaded : tensor<2x2xf32>
  %external_negated = arith.negf %external_loaded : tensor<2x2xf32>
  tt.store %destination_addresses, %external_negated : tensor<2x2x!tt.ptr<f32>>
  %slice_loaded = tt.load %source_addresses : tensor<2x2x!tt.ptr<f32>>
  %explicit_transpose = tt.trans %slice_loaded {order = array<i32: 1, 0>} : tensor<2x2xf32> -> tensor<2x2xf32>
  %slice_user = tensor.extract_slice %slice_loaded [0, 0] [1, 2] [1, 1] : tensor<2x2xf32> to tensor<1x2xf32>
  %slice_negated = arith.negf %slice_loaded : tensor<2x2xf32>
  tt.store %destination_addresses, %slice_negated : tensor<2x2x!tt.ptr<f32>>
  tt.return
}
