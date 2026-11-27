// RUN: triton-opt --triton-to-linalg='named-ops=true compile-on-910-95=true' %s | FileCheck %s --check-prefixes=CHECK,A5
// RUN: triton-opt --triton-to-linalg='named-ops=true' %s | FileCheck %s --check-prefixes=CHECK,ORDERED

// CHECK-LABEL: func.func @atomic_snapshot_lock
// A5: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock {hivm.sync_block_lock_unordered}
// A5: hivm.hir.sync_block_lock {hivm.sync_block_lock_unordered} lock_var(%[[LOCK]] :
// ORDERED: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock :
// ORDERED: hivm.hir.sync_block_lock lock_var(%[[LOCK]] :
// CHECK: memref.copy %[[SRC:[^,]+]],
// CHECK: hivm.hir.store {{.*}} outs(%[[SRC]] : {{.*}}) {already_sync} atomic = <add>
// A5: hivm.hir.sync_block_unlock {hivm.sync_block_lock_unordered} lock_var(%[[LOCK]] :
// ORDERED: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] :
tt.func public @atomic_snapshot_lock(%x: !tt.ptr<i32>, %n: i32) -> tensor<16xi32> {
  %i = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %limit = tt.splat %n : i32 -> tensor<16xi32>
  %mask = arith.cmpi slt, %i, %limit : tensor<16xi32>
  %base = tt.splat %x : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %p = tt.addptr %base, %i : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %ones = arith.constant dense<1> : tensor<16xi32>
  %old = tt.atomic_rmw add, relaxed, gpu, %p, %ones, %mask : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>, tensor<16xi1>) -> tensor<16xi32>
  tt.return %old : tensor<16xi32>
}

// CHECK-LABEL: func.func @masked_atomic_old_value
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[BUF:.*]] = memref.alloc() : memref<16xf32>
// CHECK: %[[ACTIVE:.*]] = memref.subview %[[BUF]][0] [%[[SIZE:[a-zA-Z_0-9]+]]] [1]
// CHECK: memref.copy {{.*}}, %[[ACTIVE]] :
// CHECK: %[[OLD:.*]] = bufferization.to_tensor %[[BUF]]
// CHECK: %[[ZEROS:.*]] = linalg.fill ins(%[[ZERO]] : f32)
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[OLD]][0] [%[[SIZE]]] [1]
// CHECK: %[[PADDED:.*]] = tensor.insert_slice %[[SLICE]] into %[[ZEROS]][0] [%[[SIZE]]] [1]
// CHECK: return %[[PADDED]] : tensor<16xf32>
tt.func public @masked_atomic_old_value(%x: !tt.ptr<f32>, %n: i32) -> tensor<16xf32> {
  %i = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %limit = tt.splat %n : i32 -> tensor<16xi32>
  %mask = arith.cmpi slt, %i, %limit : tensor<16xi32>
  %base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %p = tt.addptr %base, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %ones = arith.constant dense<1.0> : tensor<16xf32>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %p, %ones, %mask : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
  tt.return %old : tensor<16xf32>
}
