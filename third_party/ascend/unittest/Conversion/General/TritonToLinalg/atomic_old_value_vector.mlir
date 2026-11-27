// RUN: triton-opt --triton-to-linalg='named-ops=true compile-on-910-95=true' %s | FileCheck %s --check-prefixes=CHECK,A5
// RUN: triton-opt --triton-to-linalg='named-ops=true' %s | FileCheck %s --check-prefixes=CHECK,ORDERED

// Pure Vector. The first atomic is outside the loop, the second is inside.
// CHECK-LABEL: func.func @vector_masked_snapshots
// A5: %[[L:.*]] = hivm.hir.create_sync_block_lock {hivm.sync_block_lock_unordered}
// A5: hivm.hir.sync_block_lock {hivm.sync_block_lock_unordered} lock_var(%[[L]] :
// ORDERED: %[[L:.*]] = hivm.hir.create_sync_block_lock :
// ORDERED: hivm.hir.sync_block_lock lock_var(%[[L]] :
// CHECK: %[[OLD:.*]] = bufferization.to_tensor
// CHECK: hivm.hir.store {{.*}} {already_sync} atomic = <add>
// A5: hivm.hir.sync_block_unlock {hivm.sync_block_lock_unordered} lock_var(%[[L]] :
// ORDERED: hivm.hir.sync_block_unlock lock_var(%[[L]] :
// CHECK: %[[ACTIVE:.*]] = tensor.extract_slice %[[OLD]]
// CHECK: %[[PADDED:.*]] = tensor.insert_slice %[[ACTIVE]] into
// CHECK: scf.for {{.*}}iter_args({{.*}} = %[[PADDED]])
// A5: %[[INNER:.*]] = hivm.hir.create_sync_block_lock {hivm.sync_block_lock_unordered}
// A5: hivm.hir.sync_block_lock {hivm.sync_block_lock_unordered} lock_var(%[[INNER]] :
// ORDERED: %[[INNER:.*]] = hivm.hir.create_sync_block_lock :
// ORDERED: hivm.hir.sync_block_lock lock_var(%[[INNER]] :
// CHECK: hivm.hir.store {{.*}} {already_sync} atomic = <add>
// A5: hivm.hir.sync_block_unlock {hivm.sync_block_lock_unordered} lock_var(%[[INNER]] :
// ORDERED: hivm.hir.sync_block_unlock lock_var(%[[INNER]] :
// CHECK: %[[NEXT:.*]] = tensor.insert_slice
// CHECK: scf.yield %[[NEXT]]
tt.func public @vector_masked_snapshots(%x: !tt.ptr<f32>, %n: i32, %count: i32) -> tensor<16xf32> {
  %zero = arith.constant 0 : i32
  %step = arith.constant 1 : i32
  %ones = arith.constant dense<1.0> : tensor<16xf32>
  %i = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %limit = tt.splat %n : i32 -> tensor<16xi32>
  %mask = arith.cmpi slt, %i, %limit : tensor<16xi32>
  %base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %p = tt.addptr %base, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %first = tt.atomic_rmw fadd, relaxed, gpu, %p, %ones, %mask : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
  %last = scf.for %k = %zero to %count step %step iter_args(%prev = %first) -> (tensor<16xf32>) : i32 {
    %old = tt.atomic_rmw fadd, relaxed, gpu, %p, %prev, %mask : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
    scf.yield %old : tensor<16xf32>
  }
  tt.return %last : tensor<16xf32>
}

// CHECK-LABEL: func.func @vector_unmasked_snapshot
// CHECK: %[[BUF:.*]] = memref.alloc() : memref<16xf32>
// CHECK: memref.copy {{.*}}, %[[BUF]]
// CHECK: %[[FULL:.*]] = bufferization.to_tensor %[[BUF]]
// CHECK: hivm.hir.store {{.*}} atomic = <add>
// CHECK: return %[[FULL]] : tensor<16xf32>
tt.func public @vector_unmasked_snapshot(%x: !tt.ptr<f32>) -> tensor<16xf32> {
  %ones = arith.constant dense<1.0> : tensor<16xf32>
  %i = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %p = tt.addptr %base, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %p, %ones : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>) -> tensor<16xf32>
  tt.return %old : tensor<16xf32>
}
