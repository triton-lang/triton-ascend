// RUN: triton-opt --triton-control-flow-opt --triton-to-unstructure --split-input-file %s | FileCheck %s --implicit-check-not='scf.{{(for|while|condition|yield).*}}!tt.ptr'

tt.func public @test_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %c3_i32 = arith.constant 3 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<128> : tensor<128xi32>
  %cst_0 = arith.constant dense<0> : tensor<128xi32>
  %cst_1 = arith.constant dense<300> : tensor<128xi32>
  %c128_i32 = arith.constant 128 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c128_i32 : i32
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %3 = tt.splat %1 : i32 -> tensor<128xi32>
  %4 = arith.addi %3, %2 : tensor<128xi32>
  %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %7 = tt.addptr %6, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %8 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %9 = tt.addptr %8, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %10 = tt.load %9 : tensor<128x!tt.ptr<i32>>
  %11 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %12 = tt.addptr %11, %10 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %13:3 = scf.for %arg4 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg5 = %4, %arg6 = %7, %arg7 = %12) -> (tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>)  : i32 {
    %14:3 = scf.for %arg8 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg9 = %arg5, %arg10 = %arg6, %arg11 = %arg7) -> (tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>)  : i32 {
      %18 = arith.cmpi slt, %arg9, %cst_1 : tensor<128xi32>
      %19 = tt.addptr %5, %arg9 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %20 = tt.load %19, %18, %cst_0 : tensor<128x!tt.ptr<i32>>
      %21 = tt.load %arg11 : tensor<128x!tt.ptr<i32>>
      %22 = arith.addi %20, %21 : tensor<128xi32>
      tt.store %arg10, %22, %18 : tensor<128x!tt.ptr<i32>>
      %23 = arith.addi %arg9, %cst : tensor<128xi32>
      %24 = tt.addptr %arg10, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %25 = tt.addptr %arg11, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      scf.yield %23, %24, %25 : tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>
    }
    %15 = arith.addi %14#0, %cst : tensor<128xi32>
    %16 = tt.addptr %14#1, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %17 = tt.addptr %14#2, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    scf.yield %15, %16, %17 : tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>
  }
  tt.return
}

// CHECK-LABEL: tt.func public @test_kernel(
// CHECK: scf.for {{.*}} -> (tensor<128xi32>, i32, tensor<128xi32>)
// CHECK: scf.for {{.*}} -> (tensor<128xi32>, i32, tensor<128xi32>)
// CHECK: tt.addptr {{.*}}PointerDescriptorStructuredAxes = array<i32: 1>
// CHECK: %[[WIDE:.*]] = arith.extsi %{{.*}} : tensor<128xi32> to tensor<128xi64>
// CHECK: %[[MASK:.*]] = arith.cmpi
// CHECK: %[[DENSE:.*]] = tt.load {{.*}}, %[[MASK]],
// CHECK: %[[LOADED:.*]] = scf.for
// CHECK: %[[OFFSET:.*]] = tensor.extract %[[WIDE]]
// CHECK: %[[PTR:.*]] = tt.addptr %arg3, %[[OFFSET]] : !tt.ptr<i32>, i64
// CHECK: tt.load %[[PTR]] {DiscreteMemAccess}
// CHECK: scf.yield {DiscreteMemAccess}
// CHECK: %[[SUM:.*]] = arith.addi %[[DENSE]], %[[LOADED]]
// CHECK: tt.store {{.*}}, %[[SUM]], %[[MASK]]
// CHECK: scf.yield {{.*}} : tensor<128xi32>, i32, tensor<128xi32>
// CHECK: scf.yield {{.*}} : tensor<128xi32>, i32, tensor<128xi32>
// CHECK: tt.return

// -----

tt.func public @test_kernel2(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %c300_i32 = arith.constant 300 : i32
  %c3_i32 = arith.constant 3 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<128> : tensor<128xi32>
  %cst_0 = arith.constant dense<0> : tensor<128xi32>
  %cst_1 = arith.constant dense<300> : tensor<128xi32>
  %c128_i32 = arith.constant 128 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c128_i32 : i32
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %3 = tt.splat %1 : i32 -> tensor<128xi32>
  %4 = arith.addi %3, %2 : tensor<128xi32>
  %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %7 = tt.addptr %6, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %8 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %9 = tt.addptr %8, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %10 = tt.load %9 : tensor<128x!tt.ptr<i32>>
  %11 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %12 = tt.addptr %11, %10 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %13:4 = scf.while (%arg4 = %cst, %arg5 = %4, %arg6 = %7, %arg7 = %12) : (tensor<128xi32>, tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>) -> (tensor<128xi32>, tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>) {
    %14 = "tt.reduce"(%arg4) <{axis = 0 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %16 = arith.addi %arg8, %arg9 : i32
      tt.reduce.return %16 : i32
    }) : (tensor<128xi32>) -> i32
    %15 = arith.cmpi slt, %14, %c300_i32 : i32
    scf.condition(%15) %arg4, %arg5, %arg6, %arg7 : tensor<128xi32>, tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>
  } do {
  ^bb0(%unused: tensor<128xi32>, %arg5: tensor<128xi32>, %arg6: tensor<128x!tt.ptr<i32>>, %arg4: tensor<128x!tt.ptr<i32>>):
    %14:4 = scf.while (%arg7 = %c0_i32, %arg8 = %arg6, %arg9 = %arg4, %arg10 = %arg5) : (i32, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>, tensor<128xi32>) -> (i32, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>, tensor<128xi32>) {
      %18 = arith.cmpi slt, %arg7, %c3_i32 : i32
      scf.condition(%18) %arg7, %arg8, %arg9, %arg10 : i32, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    } do {
    ^bb0(%arg8: i32, %arg9: tensor<128x!tt.ptr<i32>>, %arg10: tensor<128x!tt.ptr<i32>>, %arg7: tensor<128xi32>):
      %18 = arith.cmpi slt, %arg7, %cst_1 : tensor<128xi32>
      %19 = tt.addptr %5, %arg7 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %20 = tt.load %19, %18, %cst_0 : tensor<128x!tt.ptr<i32>>
      %21 = tt.load %arg10 : tensor<128x!tt.ptr<i32>>
      %22 = arith.addi %20, %21 : tensor<128xi32>
      tt.store %arg9, %22, %18 : tensor<128x!tt.ptr<i32>>
      %23 = arith.addi %arg7, %cst : tensor<128xi32>
      %24 = arith.addi %arg8, %c1_i32 : i32
      %25 = tt.addptr %arg9, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %26 = tt.addptr %arg10, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      scf.yield %24, %25, %26, %23 : i32, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    }
    %15 = arith.addi %14#3, %cst : tensor<128xi32>
    %16 = tt.addptr %14#1, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %17 = tt.addptr %14#2, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    scf.yield %14#3, %15, %16, %17 : tensor<128xi32>, tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>
  }
  tt.return
}

// CHECK-LABEL: tt.func public @test_kernel2(
// CHECK: scf.while
// CHECK: scf.condition
// CHECK: scf.while
// CHECK: scf.condition
// CHECK: tt.load
// CHECK: scf.for
// CHECK: tt.load {{.*}} {DiscreteMemAccess} : !tt.ptr<i32>
// CHECK: tt.store
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: tt.return
