// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true compile-mode=simd_simt_template' --verify-each | FileCheck %s --implicit-check-not=tt.load
// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true compile-mode=simd_simt_template' --verify-each --mlir-print-op-generic | FileCheck %s --check-prefix=VOLATILE

// VOLATILE-LABEL: sym_name = "volatile_singleton"
// VOLATILE: "ascend.indirect_load"{{.*}}isVolatile = true
// VOLATILE-LABEL: sym_name = "written_singleton"
// VOLATILE: "ascend.indirect_load"{{.*}}isVolatile = true

// A discrete predicate does not require SIMT for a single address. The two
// offsets contribute 7 + 5 = 12 elements. A false predicate must not load even
// if that address is invalid; the supplied, potentially nonzero other wins.
// CHECK-LABEL: tt.func @masked_singleton(
// CHECK-SAME: %[[BASE:[^:]+]]: !tt.ptr<f32>, %[[MASK:[^:]+]]: tensor<1xi1>, %[[OTHER:[^:]+]]: tensor<1xf32>
// CHECK: %[[OFFSET:.*]] = arith.constant 12 : i64
// CHECK: %[[PTR:.*]] = tt.addptr %[[BASE]], %[[OFFSET]] : !tt.ptr<f32>, i64
// CHECK: %[[PRED:.*]] = tensor.extract %[[MASK]]
// CHECK: %[[PAD:.*]] = tensor.extract %[[OTHER]]
// CHECK-NOT: tt.load
// CHECK: %[[RESULT:.*]] = scf.if %[[PRED]] -> (f32) {
// CHECK-NEXT: %[[VALUE:.*]] = tt.load %[[PTR]] : !tt.ptr<f32>
// CHECK-NEXT: scf.yield %[[VALUE]] : f32
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[PAD]] : f32
// CHECK-NEXT: }
// CHECK: tt.splat %[[RESULT]] : f32 -> tensor<1xf32>
// CHECK-NOT: ascend.indirect_load
// CHECK: tt.return
tt.func @masked_singleton(%base: !tt.ptr<f32>, %mask: tensor<1xi1>, %other: tensor<1xf32>) -> tensor<1xf32> {
  %seven = arith.constant 7 : i64
  %five = arith.constant dense<5> : tensor<1xi64>
  %shifted = tt.addptr %base, %seven : !tt.ptr<f32>, i64
  %bases = tt.splat %shifted : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
  %ptr = tt.addptr %bases, %five : tensor<1x!tt.ptr<f32>>, tensor<1xi64>
  %value = tt.load %ptr, %mask, %other {MixCompileDiscreteMask} : tensor<1x!tt.ptr<f32>>
  tt.return %value : tensor<1xf32>
}

// Without other, the inactive value is unspecified. Zero is valid, and the
// false branch must still avoid the memory access.
// CHECK-LABEL: tt.func @masked_singleton_no_other(
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: scf.if %{{.*}} -> (i32) {
// CHECK-NEXT: %[[VALUE:.*]] = tt.load %{{.*}} : !tt.ptr<i32>
// CHECK-NEXT: scf.yield %[[VALUE]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[ZERO]] : i32
// CHECK-NEXT: }
// CHECK-NOT: ascend.indirect_load
// CHECK: tt.return
tt.func @masked_singleton_no_other(%base: !tt.ptr<i32>, %mask: tensor<1xi1>) -> tensor<1xi32> {
  %ptr = tt.splat %base : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
  %value = tt.load %ptr, %mask {MixCompileDiscreteMask} : tensor<1x!tt.ptr<i32>>
  tt.return %value : tensor<1xi32>
}

// Rank does not change the one-address contract.
// CHECK-LABEL: tt.func @masked_singleton_rank2(
// CHECK: scf.if %{{.*}} -> (f32) {
// CHECK-NEXT: %[[VALUE:.*]] = tt.load %{{.*}} : !tt.ptr<f32>
// CHECK-NEXT: scf.yield %[[VALUE]] : f32
// CHECK: tt.splat %{{.*}} : f32 -> tensor<1x1xf32>
// CHECK-NOT: ascend.indirect_load
// CHECK: tt.return
tt.func @masked_singleton_rank2(%base: !tt.ptr<f32>, %mask: tensor<1x1xi1>, %other: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %ptr = tt.splat %base : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
  %value = tt.load %ptr, %mask, %other {MixCompileDiscreteMask} : tensor<1x1x!tt.ptr<f32>>
  tt.return %value : tensor<1x1xf32>
}

// The annotation does not imply that a mask exists.
// CHECK-LABEL: tt.func @singleton_without_mask(
// CHECK-NOT: scf.if
// CHECK: tt.load %{{.*}} : !tt.ptr<f32>
// CHECK-NOT: scf.if
// CHECK: tt.return
tt.func @singleton_without_mask(%base: !tt.ptr<f32>) -> tensor<1xf32> {
  %ptr = tt.splat %base : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
  %value = tt.load %ptr {MixCompileDiscreteMask} : tensor<1x!tt.ptr<f32>>
  tt.return %value : tensor<1xf32>
}

// Explicit and inferred volatile loads keep the existing indirect path.
// CHECK-LABEL: tt.func @volatile_singleton(
// CHECK-NOT: scf.if
// CHECK: ascend.indirect_load
// CHECK: tt.return
tt.func @volatile_singleton(%base: !tt.ptr<f32>, %mask: tensor<1xi1>, %other: tensor<1xf32>) -> tensor<1xf32> {
  %ptr = tt.splat %base : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
  %value = tt.load %ptr, %mask, %other {MixCompileDiscreteMask, isVolatile = true} : tensor<1x!tt.ptr<f32>>
  tt.return %value : tensor<1xf32>
}

// CHECK-LABEL: tt.func @written_singleton(
// CHECK: tt.store
// CHECK-NOT: scf.if
// CHECK: ascend.indirect_load
// CHECK: tt.return
tt.func @written_singleton(%base: !tt.ptr<f32>, %mask: tensor<1xi1>, %other: tensor<1xf32>, %written: f32) -> tensor<1xf32> {
  tt.store %base, %written : !tt.ptr<f32>
  %ptr = tt.splat %base : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
  %value = tt.load %ptr, %mask, %other {MixCompileDiscreteMask} : tensor<1x!tt.ptr<f32>>
  tt.return %value : tensor<1xf32>
}

// Bool storage normalization is outside the scalar numeric fast path.
// CHECK-LABEL: tt.func @bool_singleton(
// CHECK-NOT: scf.if
// CHECK: ascend.indirect_load
// CHECK: tt.return
tt.func @bool_singleton(%base: !tt.ptr<i1>, %mask: tensor<1xi1>, %other: tensor<1xi1>) -> tensor<1xi1> {
  %ptr = tt.splat %base : !tt.ptr<i1> -> tensor<1x!tt.ptr<i1>>
  %value = tt.load %ptr, %mask, %other {MixCompileDiscreteMask} : tensor<1x!tt.ptr<i1>>
  tt.return %value : tensor<1xi1>
}

// CHECK-LABEL: tt.func @normalized_bool_singleton(
// CHECK-NOT: scf.if
// CHECK: ascend.indirect_load
// CHECK: tt.return
tt.func @normalized_bool_singleton(%base: !tt.ptr<i8>, %mask: tensor<1xi1>, %other: tensor<1xi8>) -> tensor<1xi8> {
  %ptr = tt.splat %base : !tt.ptr<i8> -> tensor<1x!tt.ptr<i8>>
  %value = tt.load %ptr, %mask, %other {MixCompileDiscreteMask, was_bool_to_int8 = true} : tensor<1x!tt.ptr<i8>>
  tt.return %value : tensor<1xi8>
}

// Multiple lanes still need the original discrete access implementation.
// CHECK-LABEL: tt.func @multiple_masked_lanes(
// CHECK-NOT: scf.if
// CHECK: ascend.indirect_load
// CHECK: tt.return
tt.func @multiple_masked_lanes(%base: !tt.ptr<f32>, %mask: tensor<2xi1>, %other: tensor<2xf32>) -> tensor<2xf32> {
  %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %bases = tt.splat %base : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
  %ptr = tt.addptr %bases, %range : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
  %value = tt.load %ptr, %mask, %other {MixCompileDiscreteMask} : tensor<2x!tt.ptr<f32>>
  tt.return %value : tensor<2xf32>
}
