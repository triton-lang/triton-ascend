// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// tt.atomic_cas must lower to a single hfusion.atomic_cas.
// When the CAS result is used, keep it as the op result (old value) — do not
// split into memref.copy(dst) + CAS. That matters for spin-lock loops where
// NPU-IR must scope sync to the CAS itself inside each scf.while iteration.

// CHECK-LABEL: func.func @atomic_cas_used_return
// CHECK-NOT: GenericAtomicRMW
// CHECK-NOT: Software
// CHECK: %[[OLD:.*]] = hfusion.atomic_cas ins(%{{.*}}, %{{.*}} : memref<8xi32>, memref<8xi32>) outs(%{{.*}} : memref<8xi32, strided<[1]>>) -> tensor<8xi32>
// CHECK: bufferization.materialize_in_destination %[[OLD]] in writable %{{.*}}
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @atomic_cas_used_return(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %cmp: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %val: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %ptr_splat = tt.splat %ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %cmp_splat = tt.splat %cmp : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %val_splat = tt.splat %val : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %out_splat = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptrs = tt.addptr %ptr_splat, %offs : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %cmps = tt.addptr %cmp_splat, %offs : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %vals = tt.addptr %val_splat, %offs : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %outs = tt.addptr %out_splat, %offs : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %cmp_v = tt.load %cmps : tensor<8x!tt.ptr<i32>>
    %val_v = tt.load %vals : tensor<8x!tt.ptr<i32>>
    %old = tt.atomic_cas acq_rel, gpu, %ptrs, %cmp_v, %val_v : (tensor<8x!tt.ptr<i32>>, tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    tt.store %outs, %old : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @atomic_cas_unused_return
// CHECK-NOT: GenericAtomicRMW
// CHECK-NOT: Software
// CHECK: hfusion.atomic_cas ins(%{{.*}}, %{{.*}} : memref<8xi32>, memref<8xi32>) outs(%{{.*}} : memref<8xi32, strided<[1]>>)
// CHECK-NOT: hfusion.atomic_cas {{.*}} ->
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @atomic_cas_unused_return(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %cmp: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %val: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %ptr_splat = tt.splat %ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %cmp_splat = tt.splat %cmp : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %val_splat = tt.splat %val : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptrs = tt.addptr %ptr_splat, %offs : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %cmps = tt.addptr %cmp_splat, %offs : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %vals = tt.addptr %val_splat, %offs : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %cmp_v = tt.load %cmps : tensor<8x!tt.ptr<i32>>
    %val_v = tt.load %vals : tensor<8x!tt.ptr<i32>>
    %old = tt.atomic_cas acq_rel, gpu, %ptrs, %cmp_v, %val_v : (tensor<8x!tt.ptr<i32>>, tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    tt.return
  }
}

// -----

// Spin-lock: scalar CAS return drives scf.while. Must keep a single
// hfusion.atomic_cas with result inside the loop — not memref.copy(lock)
// snapped outside / before the CAS.
// CHECK-LABEL: func.func @atomic_cas_spinlock
// CHECK-NOT: GenericAtomicRMW
// CHECK: scf.while
// CHECK-NOT: memref.copy
// CHECK: %[[OLD:.*]] = hfusion.atomic_cas ins(%{{.*}}, %{{.*}} : memref<1xi32>, memref<1xi32>) outs(%{{.*}} : memref<1xi32, strided<[1]>>) -> tensor<1xi32>
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[OLD]][%{{.*}}] : tensor<1xi32>
// CHECK: arith.cmpi eq, %[[EXTRACTED]], %{{.*}}
// CHECK: scf.condition
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @atomic_cas_spinlock(%lock: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.while : () -> () {
      %old_lock = tt.atomic_cas acq_rel, gpu, %lock, %zero, %one : (!tt.ptr<i32>, i32, i32) -> i32
      %busy = arith.cmpi eq, %old_lock, %one : i32
      scf.condition(%busy)
    } do {
      scf.yield
    }
    tt.return
  }
}
