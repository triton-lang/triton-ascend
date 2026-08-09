// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// CHECK: %[[IF:.*]] = arith.cmpi slt, {{.*}}, {{.*}} : index
// CHECK: scf.if %[[IF]] {
// CHECK-NEXT: linalg.fill ins(%[[CST]] : f32) outs(%[[ALLOC]] : memref<4096xf32>)
// CHECK-NEXT: } {hivm.unlikely_condition}
tt.func public @tensor_load_with_no_other(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
  %c4096_i32 = arith.constant 4096 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c4096_i32 : i32
  %2 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
  %3 = tt.splat %1 : i32 -> tensor<4096xi32>
  %4 = arith.addi %3, %2 : tensor<4096xi32>
  %5 = tt.splat %arg2 : i32 -> tensor<4096xi32>
  %6 = arith.cmpi slt, %4, %5 : tensor<4096xi32>
  %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
  %9 = tt.load %8, %6 : tensor<4096x!tt.ptr<f32>>
  %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
  %11 = tt.addptr %10, %4 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
  tt.store %11, %9, %6 : tensor<4096x!tt.ptr<f32>>
  tt.return
}

// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// CHECK: %[[IF:.*]] = arith.cmpi slt, {{.*}}, {{.*}} : index
// CHECK: scf.if %[[IF]] {
// CHECK-NEXT: linalg.fill ins(%[[CST]] : f32) outs(%[[ALLOC]] : memref<4096xf32>)
// CHECK-NEXT: } {hivm.unlikely_condition}
tt.func public @load_with_boundary_check(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
  %c1_i64 = arith.constant 1 : i64
  %c4096_i32 = arith.constant 4096 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c4096_i32 : i32
  %2 = arith.extsi %arg2 : i32 to i64
  %3 = tt.make_tensor_ptr %arg0, [%2], [%c1_i64], [%1] {order = array<i32: 0>} : <tensor<4096xf32>>
  %4 = tt.make_tensor_ptr %arg1, [%2], [%c1_i64], [%1] {order = array<i32: 0>} : <tensor<4096xf32>>
  %5 = tt.load %3 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<4096xf32>>
  tt.store %4, %5 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<4096xf32>>
  tt.return
}
