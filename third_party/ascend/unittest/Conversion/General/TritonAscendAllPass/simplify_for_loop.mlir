// RUN: triton-opt -allow-unregistered-dialect --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=False force-simt-template=False' '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-structured --triton-to-linalg --split-input-file %s | FileCheck %s
// CHECK-LABEL: func.func @matmul_kernel
// CHECK-DAG: %[[C0:.*]] = arith.constant{{.*}}0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant{{.*}}1 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant{{.*}}64 : index
// CHECK: %{{.*}} = scf.for %{{.*}} = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%{{.*}} = %{{.*}}, %[[ARG16:.*]] = %[[C0]]) -> (tensor<128x256xi32>, index)  : i32 {
// CHECK: %[[INNERFOR:.*]]:3 = scf.for {{.*}} = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %[[ARG20:.*]] = %[[ARG16]]) -> (tensor<128x256xi32>, tensor<64x256xi64>, index)  : i32 {
// CHECK: %{{.*}} = memref.reinterpret_cast %{{.*}} to offset: [%[[ARG20]]], sizes: [1, 64], strides: [%[[C1]], %[[C1]]] : memref<?xi8> to memref<1x64xi8, strided<[?, ?], offset: ?>>
// CHECK: %{{.*}} = linalg.broadcast ins(%{{.*}} : tensor<64xi8>) outs(%{{.*}} : tensor<128x64xi8>) dimensions = [0]
// CHECK: %[[RES72:.*]] = arith.addi %[[ARG20]], %[[C64]] : index
// CHECK: scf.yield %{{.*}}, %{{.*}}, %[[RES72]] : tensor<128x256xi32>, tensor<64x256xi64>, index
// CHECK: } {{{.*}}tts.simplify_tensor_iter_args.done}
// CHECK: scf.yield %[[INNERFOR]]#0, %[[INNERFOR]]#2 : tensor<128x256xi32>, index
// CHECK: } {{{.*}}tts.simplify_tensor_iter_args.done}

module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<64x256xi8>
    %cst_0 = arith.constant dense<0> : tensor<64x256xi8>
    %cst_1 = arith.constant dense<0> : tensor<128x64xi8>
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<1024> : tensor<1x256xi32>
    %cst_3 = arith.constant dense<1> : tensor<128x1xi32>
    %cst_4 = arith.constant dense<64> : tensor<128x64xi32>
    %cst_5 = arith.constant dense<0> : tensor<128x256xi32>
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_6 = arith.constant dense<8192> : tensor<64x1xi32>
    %c8192_i32 = arith.constant 8192 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_7 = arith.constant dense<1024> : tensor<256xi32>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c32_i32 : i32
    %2 = arith.muli %1, %c8_i32 : i32
    %3 = arith.subi %c1_i32, %2 : i32
    %4 = arith.minsi %3, %c8_i32 : i32
    %5 = arith.remsi %0, %c32_i32 : i32
    %6 = arith.remsi %5, %4 : i32
    %7 = arith.addi %2, %6 : i32
    %8 = arith.divsi %5, %4 : i32
    %9 = arith.muli %8, %c256_i32 : i32
    %10 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %11 = tt.splat %9 : i32 -> tensor<256xi32>
    %12 = arith.addi %11, %10 : tensor<256xi32>
    %13 = arith.remsi %12, %cst_7 : tensor<256xi32>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %15 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %16 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<1x64x!tt.ptr<i8>>
    %17 = tt.addptr %16, %15 : tensor<1x64x!tt.ptr<i8>>, tensor<1x64xi32>
    %18 = tt.broadcast %17 : tensor<1x64x!tt.ptr<i8>> -> tensor<128x64x!tt.ptr<i8>>
    %19 = tt.expand_dims %14 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %20 = tt.splat %arg4 : i32 -> tensor<64x1xi32>
    %21 = arith.muli %19, %20 : tensor<64x1xi32>
    %22 = tt.expand_dims %13 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %23 = tt.broadcast %21 : tensor<64x1xi32> -> tensor<64x256xi32>
    %24 = tt.broadcast %22 : tensor<1x256xi32> -> tensor<64x256xi32>
    %25 = arith.addi %23, %24 : tensor<64x256xi32>
    %26 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x256x!tt.ptr<i8>>
    %27 = tt.addptr %26, %25 : tensor<64x256x!tt.ptr<i8>>, tensor<64x256xi32>
    %28 = arith.cmpi slt, %19, %cst_6 : tensor<64x1xi32>
    %29 = tt.broadcast %28 : tensor<64x1xi1> -> tensor<64x256xi1>
    %30 = arith.muli %arg4, %c64_i32 : i32
    %31 = tt.splat %30 : i32 -> tensor<64x256xi32>
    %32:2 = scf.for %arg6 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg7 = %cst_5, %arg8 = %18) -> (tensor<128x256xi32>, tensor<128x64x!tt.ptr<i8>>)  : i32 {
      %51 = arith.muli %arg6, %c32_i32 : i32
      %52 = arith.muli %arg6, %c2_i32 : i32
      %53 = arith.shli %c3_i32, %52 : i32
      %54 = tt.splat %53 : i32 -> tensor<64x256xi32>
      %55 = tt.splat %52 : i32 -> tensor<64x256xi32>
      %56:3 = scf.for %arg9 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg10 = %arg7, %arg11 = %arg8, %arg12 = %27) -> (tensor<128x256xi32>, tensor<128x64x!tt.ptr<i8>>, tensor<64x256x!tt.ptr<i8>>)  : i32 {
        %57 = arith.addi %51, %arg9 : i32
        %58 = arith.muli %57, %c64_i32 : i32
        %59 = arith.subi %c8192_i32, %58 : i32
        %60 = tt.splat %59 : i32 -> tensor<1x64xi32>
        %61 = arith.cmpi slt, %15, %60 : tensor<1x64xi32>
        %62 = tt.broadcast %61 : tensor<1x64xi1> -> tensor<128x64xi1>
        %63 = tt.load %arg11, %62, %cst_1 : tensor<128x64x!tt.ptr<i8>>
        %64 = tt.load %arg12, %29, %cst_0 : tensor<64x256x!tt.ptr<i8>>
        %65 = arith.extui %64 : tensor<64x256xi8> to tensor<64x256xi32>
        %66 = arith.andi %65, %54 : tensor<64x256xi32>
        %67 = arith.shrsi %66, %55 : tensor<64x256xi32>
        %68 = arith.trunci %67 : tensor<64x256xi32> to tensor<64x256xi8>
        %69 = arith.subi %68, %cst : tensor<64x256xi8>
        %70 = tt.dot %63, %69, %arg10 : tensor<128x64xi8> * tensor<64x256xi8> -> tensor<128x256xi32>
        %71 = tt.addptr %arg11, %cst_4 : tensor<128x64x!tt.ptr<i8>>, tensor<128x64xi32>
        %72 = tt.addptr %arg12, %31 : tensor<64x256x!tt.ptr<i8>>, tensor<64x256xi32>
        scf.yield %70, %71, %72 : tensor<128x256xi32>, tensor<128x64x!tt.ptr<i8>>, tensor<64x256x!tt.ptr<i8>>
      }
      scf.yield %56#0, %56#1 : tensor<128x256xi32>, tensor<128x64x!tt.ptr<i8>>
    }
    %33 = arith.muli %7, %c128_i32 : i32
    %34 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %35 = tt.splat %33 : i32 -> tensor<128xi32>
    %36 = arith.addi %35, %34 : tensor<128xi32>
    %37 = tt.expand_dims %36 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %38 = tt.splat %arg5 : i32 -> tensor<128x1xi32>
    %39 = arith.muli %38, %37 : tensor<128x1xi32>
    %40 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>>
    %41 = tt.addptr %40, %39 : tensor<128x1x!tt.ptr<i32>>, tensor<128x1xi32>
    %42 = tt.expand_dims %12 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %43 = tt.broadcast %41 : tensor<128x1x!tt.ptr<i32>> -> tensor<128x256x!tt.ptr<i32>>
    %44 = tt.broadcast %42 : tensor<1x256xi32> -> tensor<128x256xi32>
    %45 = tt.addptr %43, %44 : tensor<128x256x!tt.ptr<i32>>, tensor<128x256xi32>
    %46 = arith.cmpi slt, %37, %cst_3 : tensor<128x1xi32>
    %47 = arith.cmpi slt, %42, %cst_2 : tensor<1x256xi32>
    %48 = tt.broadcast %46 : tensor<128x1xi1> -> tensor<128x256xi1>
    %49 = tt.broadcast %47 : tensor<1x256xi1> -> tensor<128x256xi1>
    %50 = arith.andi %48, %49 : tensor<128x256xi1>
    tt.store %45, %32#0, %50 : tensor<128x256x!tt.ptr<i32>>
    tt.return
  }
}
