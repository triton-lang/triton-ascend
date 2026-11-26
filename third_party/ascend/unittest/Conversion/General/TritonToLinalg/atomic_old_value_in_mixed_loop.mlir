// RUN: triton-opt --triton-to-linalg='named-ops=true compile-on-910-95=true' %s | FileCheck %s --check-prefixes=CHECK,A5
// RUN: triton-opt --triton-to-linalg='named-ops=true' %s | FileCheck %s --check-prefixes=CHECK,ORDERED

// CHECK-LABEL: func.func @atomic_old_value_in_mixed_loop
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: scf.for
// CHECK: %[[OLD:.*]] = memref.alloc() : memref<16xf32>
// CHECK: %[[ACTIVE:.*]] = memref.subview %[[OLD]][0] [%[[VALID:[a-zA-Z_0-9]+]]] [1]
// A5: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock {hivm.sync_block_lock_unordered}
// A5: hivm.hir.sync_block_lock {hivm.sync_block_lock_unordered} lock_var(%[[LOCK]] :
// ORDERED: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock :
// ORDERED: hivm.hir.sync_block_lock lock_var(%[[LOCK]] :
// CHECK: memref.copy %[[GM:[^,]+]], %[[ACTIVE]]
// CHECK: %[[SNAPSHOT:.*]] = bufferization.to_tensor %[[OLD]] restrict writable
// CHECK: hivm.hir.store {{.*}} outs(%[[GM]] : {{.*}}) {already_sync} atomic = <add>
// A5: hivm.hir.sync_block_unlock {hivm.sync_block_lock_unordered} lock_var(%[[LOCK]] :
// ORDERED: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] :
// CHECK: %[[ZEROS:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs({{.*}} : tensor<16xf32>)
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[SNAPSHOT]][0] [%[[VALID]]] [1]
// CHECK: %[[PADDED:.*]] = tensor.insert_slice %[[SLICE]] into %[[ZEROS]][0] [%[[VALID]]] [1]
// CHECK: %[[BCAST:.*]] = linalg.broadcast ins(%[[PADDED]] :
// CHECK: %[[HALF:.*]] = arith.truncf %[[BCAST]]
// CHECK: linalg.matmul {{.*}}ins(%[[HALF]], %[[HALF]] :

// Hoisting considers all locks in an outer loop together. An independent
// ordered lock in the second loop must keep both critical sections ordered.
// CHECK: scf.for
// CHECK: %[[SNAP:.*]] = hivm.hir.create_sync_block_lock :
// CHECK: hivm.hir.sync_block_lock lock_var(%[[SNAP]] :
// CHECK: memref.copy %[[GM2:[^,]+]],
// CHECK: hivm.hir.store {{.*}} outs(%[[GM2]] : {{.*}}) {already_sync} atomic = <add>
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[SNAP]] :
// CHECK: %[[EXPLICIT:.*]] = hivm.hir.create_sync_block_lock :
// CHECK: hivm.hir.sync_block_lock lock_var(%[[EXPLICIT]] :
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[EXPLICIT]] :
tt.func public @atomic_old_value_in_mixed_loop(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %count: i32, %n: i32) {
  %zero = arith.constant 0 : i32
  %step = arith.constant 1 : i32
  %ones = arith.constant dense<1.0> : tensor<16xf32>
  %acc = arith.constant dense<0.0> : tensor<16x16xf32>
  %i = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %limit = tt.splat %n : i32 -> tensor<16xi32>
  %mask = arith.cmpi slt, %i, %limit : tensor<16xi32>
  %base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %p = tt.addptr %base, %i : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %j = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  %out = tt.splat %y : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %q = tt.addptr %out, %j : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  scf.for %k = %zero to %count step %step : i32 {
    %old = tt.atomic_rmw fadd, relaxed, gpu, %p, %ones, %mask : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
    %col = tt.expand_dims %old {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %mat = tt.broadcast %col : tensor<16x1xf32> -> tensor<16x16xf32>
    %half = arith.truncf %mat : tensor<16x16xf32> to tensor<16x16xf16>
    %dot = tt.dot %half, %half, %acc : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %flat = tt.reshape %dot : tensor<16x16xf32> -> tensor<256xf32>
    tt.store %q, %flat : tensor<256x!tt.ptr<f32>>
  }
  scf.for %k = %zero to %count step %step : i32 {
    %old = tt.atomic_rmw fadd, relaxed, gpu, %p, %ones, %mask : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
    %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
    tt.store %p, %old : tensor<16x!tt.ptr<f32>>
    hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
  }
  tt.return
}
