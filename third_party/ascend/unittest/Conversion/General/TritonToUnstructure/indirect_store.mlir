// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=True force-simt-template=True' '--triton-to-unstructure=compile-on-910-95=True force-simt-template=True' %s --split-input-file | FileCheck %s

// tt.store -> ascend.indirect_store
tt.func public @triton_indirect_store_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
  %cst = arith.constant dense<32> : tensor<8x1xi32>
  %c8_i32 = arith.constant 8 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c8_i32 : i32
  %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %3 = tt.splat %1 : i32 -> tensor<8xi32>
  %4 = arith.addi %3, %2 : tensor<8xi32>
  %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>>
  %7 = tt.addptr %6, %5 : tensor<32x!tt.ptr<i64>>, tensor<32xi32>
  %8 = tt.load %7 : tensor<32x!tt.ptr<i64>>
  %9 = tt.expand_dims %2 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %10 = tt.splat %arg3 : i32 -> tensor<8x1xi32>
  %11 = arith.muli %9, %10 : tensor<8x1xi32>
  %12 = tt.expand_dims %8 {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
  %13 = arith.extsi %11 : tensor<8x1xi32> to tensor<8x1xi64>
  %14 = tt.broadcast %13 : tensor<8x1xi64> -> tensor<8x32xi64>
  %15 = tt.broadcast %12 : tensor<1x32xi64> -> tensor<8x32xi64>
  %16 = arith.addi %14, %15 : tensor<8x32xi64>
  %17 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>>
  %18 = tt.addptr %17, %16 : tensor<8x32x!tt.ptr<f32>>, tensor<8x32xi64>
  %19 = tt.load %18 : tensor<8x32x!tt.ptr<f32>>
  %20 = math.exp %19 : tensor<8x32xf32>
  %21 = tt.expand_dims %4 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %22 = arith.muli %21, %cst : tensor<8x1xi32>
  %23 = arith.extsi %22 : tensor<8x1xi32> to tensor<8x1xi64>
  %24 = tt.broadcast %23 : tensor<8x1xi64> -> tensor<8x32xi64>
  %25 = arith.addi %24, %15 : tensor<8x32xi64>
  %26 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>>
  %27 = tt.addptr %26, %25 : tensor<8x32x!tt.ptr<f32>>, tensor<8x32xi64>
  tt.store %27, %20 : tensor<8x32x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL:     tt.func public @triton_indirect_store_kernel(
// CHECK:           ascend.indirect_store %[[VAL_0:.*]] : <f32>, %[[VAL_29:.*]] : tensor<8x32xi64>, %[[VAL_24:.*]] : tensor<8x32xf32>
// CHECK:           tt.return
// CHECK:         }
