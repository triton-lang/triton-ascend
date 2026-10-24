// RUN: triton-opt %s --discrete-mask-access-conversion="compile-on-910-95=true compile-mode=simd_simt_template" --split-input-file | FileCheck %s

// A shared i64 affine pointer query must retain the structured atomic route.
// Its scalar base contains two independently signed narrow addptr steps.
// CHECK-LABEL: tt.func @structured_shared_pointer
// CHECK: tt.atomic_rmw exch, acq_rel, gpu, {{.*}} {DiscreteMask, ExplicitRouteToSIMD = true}
// CHECK: tt.return
tt.func @structured_shared_pointer(%base: !tt.ptr<i32>, %values: tensor<4xi32>, %mask: tensor<4xi1>) -> tensor<4xi32> {
  %max = arith.constant 127 : i8
  %one = arith.constant 1 : i8
  %a = tt.addptr %base, %max : !tt.ptr<i32>, i8
  %b = tt.addptr %a, %one : !tt.ptr<i32>, i8
  %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %offsets = arith.extsi %range : tensor<4xi32> to tensor<4xi64>
  %bases = tt.splat %b : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %bases, %offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi64>
  %result = tt.atomic_rmw exch, acq_rel, gpu, %ptrs, %values, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return %result : tensor<4xi32>
}

// -----

// Exact offsets with unknown structure still require the indirect atomic ABI.
// CHECK-LABEL: tt.func @opaque_shared_pointer
// CHECK: tt.atomic_rmw add, acq_rel, gpu, {{.*}} {ExplicitRouteToSIMD = false, MixCompileDiscreteMask}
// CHECK: tt.return
tt.func @opaque_shared_pointer(%base: !tt.ptr<i32>, %offsets: tensor<4xi64>, %values: tensor<4xi32>, %mask: tensor<4xi1>) -> tensor<4xi32> {
  %bases = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %bases, %offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi64>
  %result = tt.atomic_rmw add, acq_rel, gpu, %ptrs, %values, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return %result : tensor<4xi32>
}
