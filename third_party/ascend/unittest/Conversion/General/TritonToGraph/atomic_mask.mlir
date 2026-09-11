// RUN: triton-opt --graph-optimize='rule-mask=0' --split-input-file %s | FileCheck %s --check-prefix=DISABLED
// RUN: triton-opt --graph-optimize='rule-mask=512' --split-input-file %s | FileCheck %s

// Recover nested value predicates without classifying or lowering the mask.
// Pure SIMT can use this result directly for native predicated lowering.
// CHECK-LABEL: tt.func @recover_nested
// CHECK-NOT: arith.select
// CHECK: %[[ENABLED:.*]] = tt.broadcast %arg2 : tensor<2x1xi1> -> tensor<2x4xi1>
// CHECK: %[[ACTIVE:.*]] = arith.xori %arg3,
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[ACTIVE]] : tensor<2x1xi1> -> tensor<2x4xi1>
// CHECK: %[[ROWS:.*]] = arith.andi %[[BROADCAST]], %[[ENABLED]] : tensor<2x4xi1>
// CHECK: %[[MASK:.*]] = arith.andi %arg4, %[[ROWS]] : tensor<2x4xi1>
// CHECK: tt.atomic_rmw fadd, acq_rel, gpu, %arg0, %arg1, %[[MASK]] : (tensor<2x4x!tt.ptr<f32>>, tensor<2x4xf32>, tensor<2x4xi1>)
// CHECK: tt.return
tt.func @recover_nested(%ptr: tensor<2x4x!tt.ptr<f32>>, %value: tensor<2x4xf32>, %enabled: tensor<2x1xi1>, %inactive: tensor<2x1xi1>, %bounds: tensor<2x4xi1>) {
  %zero = arith.constant dense<0.0> : tensor<2x4xf32>
  %on = tt.broadcast %enabled : tensor<2x1xi1> -> tensor<2x4xi1>
  %off = tt.broadcast %inactive : tensor<2x1xi1> -> tensor<2x4xi1>
  %inner = arith.select %on, %value, %zero : tensor<2x4xi1>, tensor<2x4xf32>
  %outer = arith.select %off, %zero, %inner : tensor<2x4xi1>, tensor<2x4xf32>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %outer, %bounds : (tensor<2x4x!tt.ptr<f32>>, tensor<2x4xf32>, tensor<2x4xi1>) -> tensor<2x4xf32>
  tt.return
}

// -----

// Explicit masks must remain on loads, stores and atomics, even when a row
// mask could otherwise use the SIMD row-skip rewrite.
// CHECK-LABEL: tt.func @preserve_masked_accesses
// CHECK: %[[MASK:.*]] = tt.broadcast %arg1 : tensor<2x1xi1> -> tensor<2x4xi1>
// CHECK: %[[VALUE:.*]] = tt.load %arg0, %[[MASK]] : tensor<2x4x!tt.ptr<f32>>
// CHECK: tt.store %arg0, %[[VALUE]], %[[MASK]] : tensor<2x4x!tt.ptr<f32>>
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %arg0, %[[VALUE]], %[[MASK]] : (tensor<2x4x!tt.ptr<f32>>, tensor<2x4xf32>, tensor<2x4xi1>)
// CHECK: tt.return %[[VALUE]]
tt.func @preserve_masked_accesses(%ptr: tensor<2x4x!tt.ptr<f32>>, %row: tensor<2x1xi1>) -> tensor<2x4xf32> {
  %mask = tt.broadcast %row : tensor<2x1xi1> -> tensor<2x4xi1>
  %value = tt.load %ptr, %mask : tensor<2x4x!tt.ptr<f32>>
  tt.store %ptr, %value, %mask : tensor<2x4x!tt.ptr<f32>>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %value, %mask : (tensor<2x4x!tt.ptr<f32>>, tensor<2x4xf32>, tensor<2x4xi1>) -> tensor<2x4xf32>
  tt.return %value : tensor<2x4xf32>
}

// -----

// Integer identities also become lane masks. A wrong identity, XCHG or a
// consumed old value must retain its select and original atomic semantics.
// CHECK-LABEL: tt.func @integer_identities
// CHECK: %[[SELECT:.*]] = arith.select %arg2, %arg1,
// CHECK: tt.atomic_rmw and, relaxed, gpu, %arg0, %arg1, %arg2 :
// CHECK: tt.atomic_rmw umin, relaxed, gpu, %arg0, %arg1, %arg2 :
// CHECK: tt.atomic_rmw add, relaxed, gpu, %arg0, %[[SELECT]] :
// CHECK: tt.atomic_rmw exch, relaxed, gpu, %arg0, %[[SELECT]] :
// CHECK: %[[USED:.*]] = tt.atomic_rmw and, relaxed, gpu, %arg0, %[[SELECT]] :
// CHECK: tt.return %[[USED]]
tt.func @integer_identities(%ptr: tensor<4x!tt.ptr<i32>>, %value: tensor<4xi32>, %lane: tensor<4xi1>) -> tensor<4xi32> {
  %ones = arith.constant dense<-1> : tensor<4xi32>
  %selected = arith.select %lane, %value, %ones : tensor<4xi1>, tensor<4xi32>
  %a = tt.atomic_rmw and, relaxed, gpu, %ptr, %selected : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  %b = tt.atomic_rmw umin, relaxed, gpu, %ptr, %selected : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  %wrong = tt.atomic_rmw add, relaxed, gpu, %ptr, %selected : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  %exchange = tt.atomic_rmw exch, relaxed, gpu, %ptr, %selected : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  %used = tt.atomic_rmw and, relaxed, gpu, %ptr, %selected : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  tt.return %used : tensor<4xi32>
}

// -----

// Do not cross the bitcasts in frontend-expanded floating MIN/MAX. Their
// reconstruction remains in the existing lowering canonicalization.
// CHECK-LABEL: tt.func @expanded_float_min
// CHECK: %[[SELECT:.*]] = arith.select %arg2, %arg1,
// CHECK: %[[BITS:.*]] = tt.bitcast %[[SELECT]]
// CHECK: %[[PTR:.*]] = tt.bitcast %arg0
// CHECK: %[[POS:.*]] = arith.cmpf oge, %[[SELECT]],
// CHECK: %[[NEG:.*]] = arith.cmpf olt, %[[SELECT]],
// CHECK: %[[POSMASK:.*]] = arith.andi %arg3, %[[POS]]
// CHECK: %[[NEGMASK:.*]] = arith.andi %arg3, %[[NEG]]
// CHECK: tt.atomic_rmw min, relaxed, gpu, %[[PTR]], %[[BITS]], %[[POSMASK]] :
// CHECK: tt.atomic_rmw umax, relaxed, gpu, %[[PTR]], %[[BITS]], %[[NEGMASK]] :
// CHECK: tt.return
tt.func @expanded_float_min(%ptr: tensor<4x!tt.ptr<f32>>, %value: tensor<4xf32>, %active: tensor<4xi1>, %bounds: tensor<4xi1>) {
  %inf = arith.constant dense<0x7F800000> : tensor<4xf32>
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %selected = arith.select %active, %value, %inf : tensor<4xi1>, tensor<4xf32>
  %bits = tt.bitcast %selected : tensor<4xf32> -> tensor<4xi32>
  %iptr = tt.bitcast %ptr : tensor<4x!tt.ptr<f32>> -> tensor<4x!tt.ptr<i32>>
  %pos = arith.cmpf oge, %selected, %zero : tensor<4xf32>
  %neg = arith.cmpf olt, %selected, %zero : tensor<4xf32>
  %posmask = arith.andi %bounds, %pos : tensor<4xi1>
  %negmask = arith.andi %bounds, %neg : tensor<4xi1>
  %a = tt.atomic_rmw min, relaxed, gpu, %iptr, %bits, %posmask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  %b = tt.atomic_rmw umax, relaxed, gpu, %iptr, %bits, %negmask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// Sharing a module with an atomic must not trigger cleanup in other functions.
// CHECK-LABEL: tt.func @preserve_non_atomic_function
// CHECK-NEXT: %{{.*}} = arith.addi %arg0, %arg0 : i32
// CHECK-NEXT: tt.return %arg0 : i32
tt.func @preserve_non_atomic_function(%value: i32) -> i32 {
  %unused = arith.addi %value, %value : i32
  tt.return %value : i32
}

// -----

// Recover both users of a shared select and remove a pure identity atomic.
// Disabling the rule must preserve the original values and all three atomics.
// CHECK-LABEL: tt.func @shared_select_and_identity
// CHECK-NOT: arith.select
// CHECK: tt.atomic_rmw add, relaxed, gpu, %arg0, %arg1, %arg2 :
// CHECK: tt.atomic_rmw add, relaxed, gpu, %arg0, %arg1, %arg2 :
// CHECK-NOT: tt.atomic_rmw
// CHECK: tt.return
// DISABLED-LABEL: tt.func @shared_select_and_identity
// DISABLED: %[[SELECT:.*]] = arith.select %arg2, %arg1,
// DISABLED: tt.atomic_rmw add, relaxed, gpu, %arg0, %[[SELECT]] :
// DISABLED: tt.atomic_rmw add, relaxed, gpu, %arg0, %[[SELECT]] :
// DISABLED: tt.atomic_rmw and, relaxed, gpu,
// DISABLED: tt.return
tt.func @shared_select_and_identity(%ptr: tensor<4x!tt.ptr<i32>>, %value: tensor<4xi32>, %lane: tensor<4xi1>) {
  %zero = arith.constant dense<0> : tensor<4xi32>
  %ones = arith.constant dense<-1> : tensor<4xi32>
  %selected = arith.select %lane, %value, %zero : tensor<4xi1>, tensor<4xi32>
  %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %selected : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  %b = tt.atomic_rmw add, relaxed, gpu, %ptr, %selected : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  %c = tt.atomic_rmw and, relaxed, gpu, %ptr, %ones : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  tt.return
}

// -----

// Original Inductor form: two value selects, with only bounds on the atomic.
// CHECK-LABEL: tt.func @identity_nested_bounds
// CHECK-NOT: arith.select
// CHECK: %[[VALID:.*]] = arith.cmpi slt
// CHECK: %[[VALID_ROW:.*]] = tt.expand_dims %[[VALID]]
// CHECK: %[[BOUNDS:.*]] = tt.broadcast %[[VALID_ROW]] : tensor<2x1xi1> -> tensor<2x4xi1>
// CHECK: %[[ACTIVE:.*]] = arith.xori %arg3,
// CHECK: %[[NOT_SENTINEL:.*]] = tt.broadcast %[[ACTIVE]]
// CHECK: %[[ROW:.*]] = arith.andi %[[NOT_SENTINEL]],
// CHECK: %[[MASK:.*]] = arith.andi %[[BOUNDS]], %[[ROW]]
// CHECK: tt.atomic_rmw fadd, acq_rel, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.return
tt.func @identity_nested_bounds(%ptr: tensor<2x4x!tt.ptr<f32>>, %value: tensor<2x4xf32>, %enabled: tensor<2x1xi1>, %sentinel: tensor<2x1xi1>, %rows: i32) {
  %zero = arith.constant dense<0.0> : tensor<2x4xf32>
  %p = tt.broadcast %enabled : tensor<2x1xi1> -> tensor<2x4xi1>
  %q = tt.broadcast %sentinel : tensor<2x1xi1> -> tensor<2x4xi1>
  %a = arith.select %p, %value, %zero : tensor<2x4xi1>, tensor<2x4xf32>
  %b = arith.select %q, %zero, %a : tensor<2x4xi1>, tensor<2x4xf32>
  %r = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %n = tt.splat %rows : i32 -> tensor<2xi32>
  %valid = arith.cmpi slt, %r, %n : tensor<2xi32>
  %valid_row = tt.expand_dims %valid {axis = 1 : i32} : tensor<2xi1> -> tensor<2x1xi1>
  %bounds = tt.broadcast %valid_row : tensor<2x1xi1> -> tensor<2x4xi1>
  %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %b, %bounds : (tensor<2x4x!tt.ptr<f32>>, tensor<2x4xf32>, tensor<2x4xi1>) -> tensor<2x4xf32>
  tt.return
}

// -----

// Every integer RMW uses its own identity, derived from the opcode and width.
// CHECK-LABEL: tt.func @identity_integer_operations
// CHECK-NOT: arith.select
// CHECK: %[[MASK:.*]] = tt.broadcast %arg2 : tensor<2x1xi1> -> tensor<2x4xi1>
// CHECK: tt.atomic_rmw add, relaxed, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.atomic_rmw or, relaxed, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.atomic_rmw xor, relaxed, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.atomic_rmw umax, relaxed, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.atomic_rmw and, relaxed, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.atomic_rmw umin, relaxed, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.atomic_rmw min, relaxed, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.atomic_rmw max, relaxed, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.return
tt.func @identity_integer_operations(%ptr: tensor<2x4x!tt.ptr<i8>>, %value: tensor<2x4xi8>, %row: tensor<2x1xi1>) {
  %zero = arith.constant dense<0> : tensor<2x4xi8>
  %ones = arith.constant dense<-1> : tensor<2x4xi8>
  %max = arith.constant dense<127> : tensor<2x4xi8>
  %min = arith.constant dense<-128> : tensor<2x4xi8>
  %p = tt.broadcast %row : tensor<2x1xi1> -> tensor<2x4xi1>
  %v0 = arith.select %p, %value, %zero : tensor<2x4xi1>, tensor<2x4xi8>
  %v1 = arith.select %p, %value, %ones : tensor<2x4xi1>, tensor<2x4xi8>
  %vmax = arith.select %p, %value, %max : tensor<2x4xi1>, tensor<2x4xi8>
  %vmin = arith.select %p, %value, %min : tensor<2x4xi1>, tensor<2x4xi8>
  %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %v0 : (tensor<2x4x!tt.ptr<i8>>, tensor<2x4xi8>) -> tensor<2x4xi8>
  %b = tt.atomic_rmw or, relaxed, gpu, %ptr, %v0 : (tensor<2x4x!tt.ptr<i8>>, tensor<2x4xi8>) -> tensor<2x4xi8>
  %c = tt.atomic_rmw xor, relaxed, gpu, %ptr, %v0 : (tensor<2x4x!tt.ptr<i8>>, tensor<2x4xi8>) -> tensor<2x4xi8>
  %d = tt.atomic_rmw umax, relaxed, gpu, %ptr, %v0 : (tensor<2x4x!tt.ptr<i8>>, tensor<2x4xi8>) -> tensor<2x4xi8>
  %e = tt.atomic_rmw and, relaxed, gpu, %ptr, %v1 : (tensor<2x4x!tt.ptr<i8>>, tensor<2x4xi8>) -> tensor<2x4xi8>
  %f = tt.atomic_rmw umin, relaxed, gpu, %ptr, %v1 : (tensor<2x4x!tt.ptr<i8>>, tensor<2x4xi8>) -> tensor<2x4xi8>
  %g = tt.atomic_rmw min, relaxed, gpu, %ptr, %vmax : (tensor<2x4x!tt.ptr<i8>>, tensor<2x4xi8>) -> tensor<2x4xi8>
  %h = tt.atomic_rmw max, relaxed, gpu, %ptr, %vmin : (tensor<2x4x!tt.ptr<i8>>, tensor<2x4xi8>) -> tensor<2x4xi8>
  tt.return
}

// -----

// Floating MIN/MAX require +inf/-inf, not zero or the largest finite value.
// CHECK-LABEL: tt.func @identity_float_operations
// CHECK-NOT: arith.select
// CHECK: %[[MASK:.*]] = tt.broadcast %arg2 : tensor<2x1xi1> -> tensor<2x4xi1>
// CHECK: tt.atomic_rmw min, acq_rel, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.atomic_rmw max, acq_rel, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.atomic_rmw fadd, acq_rel, gpu, %arg0, %arg1, %[[MASK]] :
// CHECK: tt.return
tt.func @identity_float_operations(%ptr: tensor<2x4x!tt.ptr<f16>>, %value: tensor<2x4xf16>, %row: tensor<2x1xi1>) {
  %posinf = arith.constant dense<0x7C00> : tensor<2x4xf16>
  %neginf = arith.constant dense<0xFC00> : tensor<2x4xf16>
  %negzero = arith.constant dense<0x8000> : tensor<2x4xf16>
  %p = tt.broadcast %row : tensor<2x1xi1> -> tensor<2x4xi1>
  %vmin = arith.select %p, %value, %posinf : tensor<2x4xi1>, tensor<2x4xf16>
  %vmax = arith.select %p, %value, %neginf : tensor<2x4xi1>, tensor<2x4xf16>
  %vadd = arith.select %p, %value, %negzero : tensor<2x4xi1>, tensor<2x4xf16>
  %a = tt.atomic_rmw min, acq_rel, gpu, %ptr, %vmin : (tensor<2x4x!tt.ptr<f16>>, tensor<2x4xf16>) -> tensor<2x4xf16>
  %b = tt.atomic_rmw max, acq_rel, gpu, %ptr, %vmax : (tensor<2x4x!tt.ptr<f16>>, tensor<2x4xf16>) -> tensor<2x4xf16>
  %c = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %vadd : (tensor<2x4x!tt.ptr<f16>>, tensor<2x4xf16>) -> tensor<2x4xf16>
  tt.return
}

// -----

// Shape-only operations transport both the contribution and its condition.
// CHECK-LABEL: tt.func @identity_shapes_and_shared_value
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[VR:.*]] = tt.expand_dims %arg1
// CHECK: %[[MR:.*]] = tt.expand_dims %arg2
// CHECK: %[[V:.*]] = tt.broadcast %[[VR]]
// CHECK: %[[M:.*]] = tt.broadcast %[[MR]]
// CHECK: tt.atomic_rmw add, relaxed, gpu, %arg0, %[[V]], %[[M]] :
// CHECK: tt.return %[[SELECT]] : tensor<2xi64>
tt.func @identity_shapes_and_shared_value(%ptr: tensor<2x4x!tt.ptr<i64>>, %value: tensor<2xi64>, %row: tensor<2xi1>) -> tensor<2xi64> {
  %zero = arith.constant 0 : i64
  %zeros = tt.splat %zero : i64 -> tensor<2xi64>
  %v = arith.select %row, %value, %zeros : tensor<2xi1>, tensor<2xi64>
  %e = tt.expand_dims %v {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
  %b = tt.broadcast %e : tensor<2x1xi64> -> tensor<2x4xi64>
  %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %b : (tensor<2x4x!tt.ptr<i64>>, tensor<2x4xi64>) -> tensor<2x4xi64>
  tt.return %v : tensor<2xi64>
}

// -----

// Scalar predicates and values can be recovered before a splat, with no rank
// or row-width restriction. Also cover a reshape/transpose around a select.
// CHECK-LABEL: tt.func @identity_splat_transpose
// CHECK-NOT: arith.select
// CHECK: %[[V:.*]] = tt.splat %arg1 : i32 -> tensor<1x4xi32>
// CHECK: %[[M:.*]] = tt.splat %arg2 : i1 -> tensor<1x4xi1>
// CHECK: %[[VT:.*]] = tt.trans %[[V]]
// CHECK: %[[MT:.*]] = tt.trans %[[M]]
// CHECK: %[[VR:.*]] = tt.reshape %[[VT]]
// CHECK: %[[MR:.*]] = tt.reshape %[[MT]]
// CHECK: tt.atomic_rmw add, relaxed, gpu, %arg0, %[[VR]], %[[MR]] :
// CHECK: tt.return
tt.func @identity_splat_transpose(%ptr: tensor<4x!tt.ptr<i32>>, %value: i32, %active: i1) {
  %zero = arith.constant 0 : i32
  %v = arith.select %active, %value, %zero : i32
  %s = tt.splat %v : i32 -> tensor<1x4xi32>
  %t = tt.trans %s {order = array<i32: 1, 0>} : tensor<1x4xi32> -> tensor<4x1xi32>
  %r = tt.reshape %t : tensor<4x1xi32> -> tensor<4xi32>
  %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %r : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  tt.return
}

// -----

// A pure identity fill disappears under the numerical reduction contract.
// CHECK-LABEL: tt.func @identity_fill
// CHECK-NOT: tt.atomic_rmw
// CHECK: tt.return
tt.func @identity_fill(%ptr: tensor<4x!tt.ptr<i64>>) {
  %ones = arith.constant dense<-1> : tensor<4xi64>
  %old = tt.atomic_rmw and, acq_rel, gpu, %ptr, %ones : (tensor<4x!tt.ptr<i64>>, tensor<4xi64>) -> tensor<4xi64>
  tt.return
}

// -----

// The old value is observable: even an identity atomic must stay.
// CHECK-LABEL: tt.func @identity_used_result
// CHECK: %[[V:.*]] = arith.select
// CHECK: %[[OLD:.*]] = tt.atomic_rmw add, relaxed, gpu, %arg0, %[[V]] :
// CHECK: tt.return %[[OLD]]
tt.func @identity_used_result(%ptr: tensor<4x!tt.ptr<i32>>, %value: tensor<4xi32>, %active: i1) -> tensor<4xi32> {
  %zero = arith.constant dense<0> : tensor<4xi32>
  %v = arith.select %active, %value, %zero : i1, tensor<4xi32>
  %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %v : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  tt.return %old : tensor<4xi32>
}

// -----

// Wrong identities must not become access predicates. In particular, INT_MAX
// is neither the unsigned-min identity nor the AND identity for signless i32.
// CHECK-LABEL: tt.func @identity_wrong_defaults
// CHECK: %[[V:.*]] = arith.select
// CHECK: %[[V0:.*]] = arith.select
// CHECK: tt.atomic_rmw and, relaxed, gpu, %arg0, %[[V]] :
// CHECK: tt.atomic_rmw umin, relaxed, gpu, %arg0, %[[V]] :
// CHECK: tt.atomic_rmw min, relaxed, gpu, %arg0, %[[V0]] :
// CHECK: tt.atomic_rmw max, relaxed, gpu, %arg0, %[[V0]] :
// CHECK: tt.atomic_rmw exch, relaxed, gpu, %arg0, %[[V0]] :
// CHECK: tt.return
tt.func @identity_wrong_defaults(%ptr: tensor<4x!tt.ptr<i32>>, %value: tensor<4xi32>, %active: i1) {
  %max = arith.constant dense<2147483647> : tensor<4xi32>
  %zero = arith.constant dense<0> : tensor<4xi32>
  %v = arith.select %active, %value, %max : i1, tensor<4xi32>
  %v0 = arith.select %active, %value, %zero : i1, tensor<4xi32>
  %a = tt.atomic_rmw and, relaxed, gpu, %ptr, %v : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  %b = tt.atomic_rmw umin, relaxed, gpu, %ptr, %v : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  %c = tt.atomic_rmw min, relaxed, gpu, %ptr, %v0 : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  %d = tt.atomic_rmw max, relaxed, gpu, %ptr, %v0 : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  %e = tt.atomic_rmw exch, relaxed, gpu, %ptr, %v0 : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  tt.return
}

// -----

// Non-identity FP fills and arithmetic boundaries remain ordinary values.
// CHECK-LABEL: tt.func @identity_float_boundaries
// CHECK: %[[VMIN:.*]] = arith.select
// CHECK: %[[V0:.*]] = arith.select
// CHECK: %[[PRODUCT:.*]] = arith.mulf %[[V0]], %arg1
// CHECK: tt.atomic_rmw min, relaxed, gpu, %arg0, %[[VMIN]] :
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %arg0, %[[PRODUCT]] :
// CHECK: tt.return
tt.func @identity_float_boundaries(%ptr: tensor<4x!tt.ptr<f16>>, %value: tensor<4xf16>, %active: i1) {
  %max = arith.constant dense<65504.0> : tensor<4xf16>
  %zero = arith.constant dense<0.0> : tensor<4xf16>
  %vmin = arith.select %active, %value, %max : i1, tensor<4xf16>
  %v0 = arith.select %active, %value, %zero : i1, tensor<4xf16>
  %product = arith.mulf %v0, %value : tensor<4xf16>
  %a = tt.atomic_rmw min, relaxed, gpu, %ptr, %vmin : (tensor<4x!tt.ptr<f16>>, tensor<4xf16>) -> tensor<4xf16>
  %b = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %product : (tensor<4x!tt.ptr<f16>>, tensor<4xf16>) -> tensor<4xf16>
  tt.return
}

// -----

// Lane-varying predicates remain explicit masks at the graph stage.
// CHECK-LABEL: tt.func @identity_lane_mask
// CHECK-NOT: arith.select
// CHECK: tt.atomic_rmw umin, relaxed, gpu, %arg0, %arg1, %arg2 :
// CHECK: tt.return
tt.func @identity_lane_mask(%ptr: tensor<4x!tt.ptr<i32>>, %value: tensor<4xi32>, %mask: tensor<4xi1>) {
  %ones = arith.constant dense<-1> : tensor<4xi32>
  %v = arith.select %mask, %value, %ones : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw umin, relaxed, gpu, %ptr, %v : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  tt.return
}

// -----

// Numeric equality recognizes all floating identities, including BF16 and
// identities on the true branch. +inf must not be mistaken for MAX's -inf.
// CHECK-LABEL: tt.func @identity_bf16_reverse
// CHECK: %[[WRONG:.*]] = arith.select
// CHECK: %[[ACTIVE0:.*]] = arith.xori %arg2,
// CHECK: %[[MASK0:.*]] = tt.splat %[[ACTIVE0]] : i1 -> tensor<4xi1>
// CHECK: tt.atomic_rmw min, acq_rel, gpu, %arg0, %arg1, %[[MASK0]] :
// CHECK: %[[ACTIVE1:.*]] = arith.xori %arg2,
// CHECK: %[[MASK1:.*]] = tt.splat %[[ACTIVE1]] : i1 -> tensor<4xi1>
// CHECK: tt.atomic_rmw max, acq_rel, gpu, %arg0, %arg1, %[[MASK1]] :
// CHECK: %[[ACTIVE2:.*]] = arith.xori %arg2,
// CHECK: %[[MASK2:.*]] = tt.splat %[[ACTIVE2]] : i1 -> tensor<4xi1>
// CHECK: tt.atomic_rmw fadd, acq_rel, gpu, %arg0, %arg1, %[[MASK2]] :
// CHECK: tt.atomic_rmw max, acq_rel, gpu, %arg0, %[[WRONG]] :
// CHECK: tt.return
tt.func @identity_bf16_reverse(%ptr: tensor<4x!tt.ptr<bf16>>, %value: tensor<4xbf16>, %inactive: i1) {
  %posinf = arith.constant dense<0x7F80> : tensor<4xbf16>
  %neginf = arith.constant dense<0xFF80> : tensor<4xbf16>
  %zero = arith.constant dense<0.0> : tensor<4xbf16>
  %vmin = arith.select %inactive, %posinf, %value : i1, tensor<4xbf16>
  %a = tt.atomic_rmw min, acq_rel, gpu, %ptr, %vmin : (tensor<4x!tt.ptr<bf16>>, tensor<4xbf16>) -> tensor<4xbf16>
  %vmax = arith.select %inactive, %neginf, %value : i1, tensor<4xbf16>
  %b = tt.atomic_rmw max, acq_rel, gpu, %ptr, %vmax : (tensor<4x!tt.ptr<bf16>>, tensor<4xbf16>) -> tensor<4xbf16>
  %vadd = arith.select %inactive, %zero, %value : i1, tensor<4xbf16>
  %c = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %vadd : (tensor<4x!tt.ptr<bf16>>, tensor<4xbf16>) -> tensor<4xbf16>
  %wrong = tt.atomic_rmw max, acq_rel, gpu, %ptr, %vmin : (tensor<4x!tt.ptr<bf16>>, tensor<4xbf16>) -> tensor<4xbf16>
  tt.return
}

// -----

// Unsigned Max must use the actual element width, not an i32 all-ones value.
// CHECK-LABEL: tt.func @identity_i64_all_ones
// CHECK-NOT: arith.select
// CHECK: %[[ACTIVE0:.*]] = arith.xori %arg2,
// CHECK: %[[MASK0:.*]] = tt.splat %[[ACTIVE0]] : i1 -> tensor<4xi1>
// CHECK: tt.atomic_rmw and, relaxed, gpu, %arg0, %arg1, %[[MASK0]] :
// CHECK-SAME: (tensor<4x!tt.ptr<i64>>, tensor<4xi64>, tensor<4xi1>)
// CHECK: %[[ACTIVE1:.*]] = arith.xori %arg2,
// CHECK: %[[MASK1:.*]] = tt.splat %[[ACTIVE1]] : i1 -> tensor<4xi1>
// CHECK: tt.atomic_rmw umin, relaxed, gpu, %arg0, %arg1, %[[MASK1]] :
// CHECK-SAME: (tensor<4x!tt.ptr<i64>>, tensor<4xi64>, tensor<4xi1>)
// CHECK: tt.return
tt.func @identity_i64_all_ones(%ptr: tensor<4x!tt.ptr<i64>>, %value: tensor<4xi64>, %inactive: i1) {
  %ones = arith.constant dense<-1> : tensor<4xi64>
  %v = arith.select %inactive, %ones, %value : i1, tensor<4xi64>
  %a = tt.atomic_rmw and, relaxed, gpu, %ptr, %v : (tensor<4x!tt.ptr<i64>>, tensor<4xi64>) -> tensor<4xi64>
  %b = tt.atomic_rmw umin, relaxed, gpu, %ptr, %v : (tensor<4x!tt.ptr<i64>>, tensor<4xi64>) -> tensor<4xi64>
  tt.return
}
