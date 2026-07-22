// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=4 ub-capacity-bytes=64' -o - | FileCheck %s --check-prefix=CHECK
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=4 ub-capacity-bytes=0' -o - | FileCheck %s --check-prefix=CAP0

// Address order, rather than program order, determines the packed value
// layout.  The high interval is deliberately stored first.  The replacement
// must still be anchored at the program-order last store (the low interval),
// after both source values dominate that anchor.
// CHECK-LABEL: tt.func @pack_reverse_program_order(
// CHECK-NOT: tt.store
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8xi32>
// CHECK-NEXT: %[[LOW:.*]] = tensor.insert_slice %{{.*}} into %[[EMPTY]][0] [4] [1] : tensor<4xi32> into tensor<8xi32>
// CHECK-NEXT: %[[PACKED:.*]] = tensor.insert_slice %{{.*}} into %[[LOW]][4] [4] [1] : tensor<4xi32> into tensor<8xi32>
// CHECK-NEXT: %[[RANGE:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %[[SPLAT:.*]] = tt.splat %{{.*}} : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
// CHECK-NEXT: %[[ADDRESS:.*]] = tt.addptr %[[SPLAT]], %[[RANGE]] : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
// CHECK-NEXT: tt.store %[[ADDRESS]], %[[PACKED]] : tensor<8x!tt.ptr<i32>>
// CHECK-NOT: tt.store
// CHECK-NOT: tensor.insert_slice
tt.func @pack_reverse_program_order(%base: !tt.ptr<i32>) {
  %low_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %high_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %low_addresses = tt.addptr %base_splat, %low_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %high_addresses = tt.addptr %base_splat, %high_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %low_value = arith.constant dense<1> : tensor<4xi32>
  %high_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %high_addresses, %high_value : tensor<4x!tt.ptr<i32>>
  tt.store %low_addresses, %low_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_gap(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_gap(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 9 : i32, start = 5 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// Any overlap poisons the entire bucket, rather than merely being skipped.
// CHECK-LABEL: tt.func @reject_overlap(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_overlap(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 6 : i32, start = 2 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_mask(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_mask(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %mask = arith.constant dense<true> : tensor<4xi1>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value, %mask : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// i1 has no byte-addressable element width, so capacity accounting must not
// round it up and pack the stores.
// CHECK-LABEL: tt.func @reject_i1_element(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i1>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i1>>
tt.func @reject_i1_element(%base: !tt.ptr<i1>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i1> -> tensor<4x!tt.ptr<i1>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
  %first_value = arith.constant dense<true> : tensor<4xi1>
  %second_value = arith.constant dense<false> : tensor<4xi1>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i1>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i1>>
  tt.return
}

// Complete store attributes must match; cache policy alone makes these two
// stores distinct buckets.
// CHECK-LABEL: tt.func @reject_different_cache(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} cacheModifier = ca : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} cacheModifier = cg : tensor<4x!tt.ptr<i32>>
tt.func @reject_different_cache(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value cacheModifier = ca : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value cacheModifier = cg : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_different_evict(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} evictionPolicy = evict_first : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} evictionPolicy = evict_last : tensor<4x!tt.ptr<i32>>
tt.func @reject_different_evict(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value evictionPolicy = evict_first : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value evictionPolicy = evict_last : tensor<4x!tt.ptr<i32>>
  tt.return
}

// Exact SSA base identity, not an alias guess, is required for a bucket.
// CHECK-LABEL: tt.func @reject_different_base(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_different_base(%first_base: !tt.ptr<i32>, %second_base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %first_splat = tt.splat %first_base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %second_splat = tt.splat %second_base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %first_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %second_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// A protected program-order interval cannot contain any memory effect.
// CHECK-LABEL: tt.func @reject_load_between_stores(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.load {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_load_between_stores(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  %unused = tt.load %first_addresses : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_store_between_stores(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_store_between_stores(%base: !tt.ptr<i32>, %other_base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %other_splat = tt.splat %other_base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %other_addresses = tt.addptr %other_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  %other_value = arith.constant dense<3> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %other_addresses, %other_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_barrier_between_stores(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: gpu.barrier
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_barrier_between_stores(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  gpu.barrier
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// A valid direct-body same-block value defined before the anchor naturally
// dominates it.  The matcher still checks that condition defensively during
// discovery and revalidation.  This nested-region fixture exercises the
// accompanying direct tt.func body/same-block gate: it must remain untouched.
// CHECK-LABEL: tt.func @reject_nested_region_stores(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_nested_region_stores(%base: !tt.ptr<i32>, %condition: i1) {
  scf.if %condition {
    %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
    %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %first_value = arith.constant dense<1> : tensor<4xi32>
    %second_value = arith.constant dense<2> : tensor<4xi32>
    tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
    tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  }
  tt.return
}

// CAP0-LABEL: tt.func @reject_capacity_zero(
// CAP0-NOT: tensor.empty
// CAP0-NOT: tensor.insert_slice
// CAP0: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CAP0: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_capacity_zero(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}
