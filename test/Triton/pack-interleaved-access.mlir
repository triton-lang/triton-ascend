// RUN: %PYTHON %S/pack_interleaved_access.py triton-opt %s
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @causal_conv1d_update_kernel_npu(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %c24576_i64 = arith.constant 24576 : i64
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1024xf16>
    %cst_1 = arith.constant dense<3> : tensor<1024xi32>
    %cst_2 = arith.constant dense<2> : tensor<1024xi32>
    %cst_3 = arith.constant dense<4> : tensor<1024xi32>
    %cst_4 = arith.constant dense<1> : tensor<1024xi32>
    %cst_5 = arith.constant dense<6144> : tensor<1024xi32>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.extsi %1 : i32 to i64
    %3 = arith.muli %0, %c1024_i32 : i32
    %4 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %5 = tt.splat %3 : i32 -> tensor<1024xi32>
    %6 = arith.addi %5, %4 : tensor<1024xi32>
    %7 = arith.cmpi slt, %6, %cst_5 : tensor<1024xi32>
    %8 = arith.extsi %arg5 : i32 to i64
    %9 = arith.muli %2, %8 : i64
    %10 = tt.addptr %arg0, %9 : !tt.ptr<f16>, i64
    %11 = tt.splat %10 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %12 = tt.addptr %11, %6 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %13 = tt.load %12, %7, %cst_0 : tensor<1024x!tt.ptr<f16>>
    %14 = arith.extf %13 : tensor<1024xf16> to tensor<1024xf32>
    %15 = arith.muli %2, %c24576_i64 : i64
    %16 = tt.addptr %arg1, %15 : !tt.ptr<f16>, i64
    %17 = arith.muli %6, %cst_3 : tensor<1024xi32>
    %18 = tt.splat %16 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %19 = tt.addptr %18, %17 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %20 = tt.addptr %19, %cst_4 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %21 = tt.load %20, %7, %cst_0 : tensor<1024x!tt.ptr<f16>>
    %22 = arith.extf %21 : tensor<1024xf16> to tensor<1024xf32>
    tt.store %19, %21, %7 : tensor<1024x!tt.ptr<f16>>
    %23 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %24 = tt.addptr %23, %17 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %25 = tt.load %24, %7, %cst_0 : tensor<1024x!tt.ptr<f16>>
    %26 = arith.extf %25 : tensor<1024xf16> to tensor<1024xf32>
    %27 = arith.mulf %22, %26 : tensor<1024xf32>
    %28 = arith.addf %27, %cst : tensor<1024xf32>
    %29 = tt.addptr %19, %cst_2 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %30 = tt.load %29, %7, %cst_0 : tensor<1024x!tt.ptr<f16>>
    %31 = arith.extf %30 : tensor<1024xf16> to tensor<1024xf32>
    tt.store %20, %30, %7 : tensor<1024x!tt.ptr<f16>>
    %32 = tt.addptr %24, %cst_4 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %33 = tt.load %32, %7, %cst_0 : tensor<1024x!tt.ptr<f16>>
    %34 = arith.extf %33 : tensor<1024xf16> to tensor<1024xf32>
    %35 = arith.mulf %31, %34 : tensor<1024xf32>
    %36 = arith.addf %28, %35 : tensor<1024xf32>
    %37 = tt.addptr %19, %cst_1 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %38 = tt.load %37, %7, %cst_0 : tensor<1024x!tt.ptr<f16>>
    %39 = arith.extf %38 : tensor<1024xf16> to tensor<1024xf32>
    tt.store %29, %38, %7 : tensor<1024x!tt.ptr<f16>>
    %40 = tt.addptr %24, %cst_2 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %41 = tt.load %40, %7, %cst_0 : tensor<1024x!tt.ptr<f16>>
    %42 = arith.extf %41 : tensor<1024xf16> to tensor<1024xf32>
    %43 = arith.mulf %39, %42 : tensor<1024xf32>
    %44 = arith.addf %36, %43 : tensor<1024xf32>
    tt.store %37, %13, %7 : tensor<1024x!tt.ptr<f16>>
    %45 = tt.addptr %24, %cst_1 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %46 = tt.load %45, %7, %cst_0 : tensor<1024x!tt.ptr<f16>>
    %47 = arith.extf %46 : tensor<1024xf16> to tensor<1024xf32>
    %48 = arith.mulf %14, %47 : tensor<1024xf32>
    %49 = arith.addf %44, %48 : tensor<1024xf32>
    %50 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %51 = tt.addptr %50, %6 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %52 = tt.load %51, %7, %cst_0 : tensor<1024x!tt.ptr<f16>>
    %53 = arith.extf %52 : tensor<1024xf16> to tensor<1024xf32>
    %54 = arith.addf %49, %53 : tensor<1024xf32>
    %55 = arith.extsi %arg6 : i32 to i64
    %56 = arith.muli %2, %55 : i64
    %57 = tt.addptr %arg2, %56 : !tt.ptr<f16>, i64
    %58 = tt.splat %57 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %59 = tt.addptr %58, %6 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %60 = arith.truncf %54 : tensor<1024xf32> to tensor<1024xf16>
    tt.store %59, %60, %7 : tensor<1024x!tt.ptr<f16>>
    tt.return
  }
}
