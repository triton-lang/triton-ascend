// RUN: triton-opt --pass-pipeline="builtin.module(auto-blockify{auto-blockify-size=1},triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},discrete-mask-access-conversion{compile-on-910-95=false force-simt-template=false},triton-to-annotation,triton-to-unstructure{compile-on-910-95=false force-scalarize-mode=false force-simt-template=false},triton-to-hivm,triton-to-hfusion,triton-to-llvm,bubble-up-operation{enable-aggressive-mode=true},triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},triton-to-linalg{compile-on-910-95=false enable-nd2nz-on-vector=false enable-select-analysis=true global-kernel=false named-ops=true})" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @atomic_add_discrete_mask
// CHECK: arith.select
// CHECK: hivm.hir.store ins(%{{.*}} : tensor<128xf32>) outs(%{{.*}} : memref<128xf32{{.*}}>) atomic = <add>

// CHECK-LABEL: func.func @atomic_and_discrete_mask
// CHECK: arith.select
// CHECK: hfusion.atomic_rmw ins(%{{.*}} : memref<128xi32{{.*}}>) outs(%{{.*}} : memref<128xi32{{.*}}>) atomic_kind = <and>

// CHECK-LABEL: func.func @atomic_max_discrete_mask
// CHECK: arith.select
// CHECK: hivm.hir.store ins(%{{.*}} : tensor<128xi32>) outs(%{{.*}} : memref<128xi32{{.*}}>) atomic = <max>

// CHECK-LABEL: func.func @atomic_min_discrete_mask
// CHECK: arith.select
// CHECK: hivm.hir.store ins(%{{.*}} : tensor<128xi32>) outs(%{{.*}} : memref<128xi32{{.*}}>) atomic = <min>

// CHECK-LABEL: func.func @atomic_or_discrete_mask
// CHECK: arith.select
// CHECK: hfusion.atomic_rmw ins(%{{.*}} : memref<128xi32{{.*}}>) outs(%{{.*}} : memref<128xi32{{.*}}>) atomic_kind = <or>

// CHECK-LABEL: func.func @atomic_xor_discrete_mask
// CHECK: arith.select
// CHECK: hfusion.atomic_rmw ins(%{{.*}} : memref<128xi32{{.*}}>) outs(%{{.*}} : memref<128xi32{{.*}}>) atomic_kind = <xor>


module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @atomic_add_discrete_mask(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<128xf32>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = arith.sitofp %4 : tensor<128xi32> to tensor<128xf32>
    %8 = math.sin %7 : tensor<128xf32>
    %9 = arith.cmpf olt, %8, %cst_0 : tensor<128xf32>
    %10 = arith.andi %6, %9 : tensor<128xi1>
    %11 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %13 = tt.load %12, %6, %cst : tensor<128x!tt.ptr<f32>>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %16 = tt.atomic_rmw fadd, acq_rel, gpu, %15, %13, %10 : (tensor<128x!tt.ptr<f32>>, tensor<128xf32>, tensor<128xi1>) -> tensor<128xf32>
    tt.return
  }


  tt.func public @atomic_and_discrete_mask(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<128xi32>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<128xf32>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = arith.sitofp %4 : tensor<128xi32> to tensor<128xf32>
    %8 = math.sin %7 : tensor<128xf32>
    %9 = arith.cmpf olt, %8, %cst_0 : tensor<128xf32>
    %10 = arith.andi %6, %9 : tensor<128xi1>
    %11 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %12 = tt.addptr %11, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %13 = tt.load %12, %6, %cst : tensor<128x!tt.ptr<i32>>
    %14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %16 = tt.atomic_rmw and, acq_rel, gpu, %15, %13, %10 : (tensor<128x!tt.ptr<i32>>, tensor<128xi32>, tensor<128xi1>) -> tensor<128xi32>
    tt.return
  }


  tt.func public @atomic_max_discrete_mask(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<128xi32>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<128xf32>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = arith.sitofp %4 : tensor<128xi32> to tensor<128xf32>
    %8 = math.sin %7 : tensor<128xf32>
    %9 = arith.cmpf olt, %8, %cst_0 : tensor<128xf32>
    %10 = arith.andi %6, %9 : tensor<128xi1>
    %11 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %12 = tt.addptr %11, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %13 = tt.load %12, %6, %cst : tensor<128x!tt.ptr<i32>>
    %14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %16 = tt.atomic_rmw max, acq_rel, gpu, %15, %13, %10 : (tensor<128x!tt.ptr<i32>>, tensor<128xi32>, tensor<128xi1>) -> tensor<128xi32>
    tt.return
  }


  tt.func public @atomic_min_discrete_mask(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<128xi32>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<128xf32>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = arith.sitofp %4 : tensor<128xi32> to tensor<128xf32>
    %8 = math.sin %7 : tensor<128xf32>
    %9 = arith.cmpf olt, %8, %cst_0 : tensor<128xf32>
    %10 = arith.andi %6, %9 : tensor<128xi1>
    %11 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %12 = tt.addptr %11, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %13 = tt.load %12, %6, %cst : tensor<128x!tt.ptr<i32>>
    %14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %16 = tt.atomic_rmw min, acq_rel, gpu, %15, %13, %10 : (tensor<128x!tt.ptr<i32>>, tensor<128xi32>, tensor<128xi1>) -> tensor<128xi32>
    tt.return
  }


  tt.func public @atomic_or_discrete_mask(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<128xi32>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<128xf32>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = arith.sitofp %4 : tensor<128xi32> to tensor<128xf32>
    %8 = math.sin %7 : tensor<128xf32>
    %9 = arith.cmpf olt, %8, %cst_0 : tensor<128xf32>
    %10 = arith.andi %6, %9 : tensor<128xi1>
    %11 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %12 = tt.addptr %11, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %13 = tt.load %12, %6, %cst : tensor<128x!tt.ptr<i32>>
    %14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %16 = tt.atomic_rmw or, acq_rel, gpu, %15, %13, %10 : (tensor<128x!tt.ptr<i32>>, tensor<128xi32>, tensor<128xi1>) -> tensor<128xi32>
    tt.return
  }


  tt.func public @atomic_xor_discrete_mask(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<128xi32>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<128xf32>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = arith.sitofp %4 : tensor<128xi32> to tensor<128xf32>
    %8 = math.sin %7 : tensor<128xf32>
    %9 = arith.cmpf olt, %8, %cst_0 : tensor<128xf32>
    %10 = arith.andi %6, %9 : tensor<128xi1>
    %11 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %12 = tt.addptr %11, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %13 = tt.load %12, %6, %cst : tensor<128x!tt.ptr<i32>>
    %14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %16 = tt.atomic_rmw xor, acq_rel, gpu, %15, %13, %10 : (tensor<128x!tt.ptr<i32>>, tensor<128xi32>, tensor<128xi1>) -> tensor<128xi32>
    tt.return
  }
}
