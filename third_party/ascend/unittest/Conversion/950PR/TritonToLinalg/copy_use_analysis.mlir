// RUN: triton-opt -allow-unregistered-dialect '--triton-to-linalg=named-ops=True enable-nd2nz-on-vector=True compile-on-910-95=True' --split-input-file %s -verify-each 2>&1 | FileCheck %s --check-prefix=NOERR
// NOERR-NOT: failed to legalize unresolved materialization
// CHECK: module
// CHECK: func.func public @backward_dkdv

module {
  tt.func public @backward_dkdv(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: f32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<32x32xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_2 = arith.constant dense<1> : tensor<32xi32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %cst_3 = arith.constant 1.44269502 : f32
    %c1_i32 = arith.constant 1 : i32
    %alloc = memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
    %alloc_4 = memref.alloc() : memref<32x64xf32, #hivm.address_space<ub>>
    %alloc_5 = memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
    %alloc_6 = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
    %alloc_7 = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
    %alloc_8 = memref.alloc() : memref<32x64xf32, #hivm.address_space<ub>>
    %0 = tt.get_program_id x : i32
    scope.scope : () -> () {
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 10
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
      %1 = tt.get_num_programs x : i32
      %2 = arith.mulf %arg15, %cst_3 : f32
      %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
      %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %7 = tt.splat %arg14 : i32 -> tensor<1x64xi32>
      %8 = arith.cmpi slt, %6, %7 : tensor<1x64xi32>
      %9 = tt.broadcast %8 : tensor<1x64xi1> -> tensor<32x64xi1>
      %10 = tt.splat %arg27 : i32 -> tensor<32x1xi32>
      %11 = arith.muli %4, %10 : tensor<32x1xi32>
      %12 = tt.broadcast %6 : tensor<1x64xi32> -> tensor<32x64xi32>
      %13 = tt.splat %arg30 : i32 -> tensor<32x1xi32>
      %14 = arith.muli %4, %13 : tensor<32x1xi32>
      %15 = tt.splat %arg9 : i32 -> tensor<32xi32>
      %16 = arith.muli %3, %15 : tensor<32xi32>
      %17 = tt.splat %arg8 : i32 -> tensor<32xi32>
      %18 = arith.addi %16, %17 : tensor<32xi32>
      %19 = arith.subi %18, %cst_2 : tensor<32xi32>
      %20 = arith.subi %arg8, %c1_i32 : i32
      %21 = tt.expand_dims %19 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %22 = tt.broadcast %21 : tensor<32x1xi32> -> tensor<32x32xi32>
      %23 = tt.splat %2 : f32 -> tensor<32x32xf32>
      %24 = tt.splat %arg15 : f32 -> tensor<32x32xf32>
      scf.for %arg33 = %0 to %arg32 step %1  : i32 {
        %25 = arith.divsi %arg33, %arg32 : i32
        %26 = arith.remsi %arg33, %arg32 : i32
        %27 = arith.divsi %26, %arg13 : i32
        %28 = arith.remsi %26, %arg13 : i32
        %29 = tt.addptr %arg10, %25 : !tt.ptr<i32>, i32
        %30 = tt.load %29 : !tt.ptr<i32>
        %31 = tt.addptr %29, %c1_i32 : !tt.ptr<i32>, i32
        %32 = tt.load %31 : !tt.ptr<i32>
        %33 = arith.subi %32, %30 : i32
        %34 = tt.addptr %arg11, %25 : !tt.ptr<i32>, i32
        %35 = tt.load %34 : !tt.ptr<i32>
        %36 = tt.addptr %34, %c1_i32 : !tt.ptr<i32>, i32
        %37 = tt.load %36 : !tt.ptr<i32>
        %38 = arith.subi %37, %35 : i32
        %39 = tt.splat %38 : i32 -> tensor<32x1xi32>
        %40 = arith.cmpi slt, %4, %39 : tensor<32x1xi32>
        %41 = tt.broadcast %40 : tensor<32x1xi1> -> tensor<32x64xi1>
        %42 = arith.andi %41, %9 : tensor<32x64xi1>
        %43 = arith.muli %35, %arg27 : i32
        %44 = tt.addptr %arg6, %43 : !tt.ptr<f16>, i32
        %45 = arith.muli %27, %arg28 : i32
        %46 = tt.addptr %44, %45 : !tt.ptr<f16>, i32
        %47 = arith.muli %28, %arg26 : i32
        %48 = tt.addptr %46, %47 : !tt.ptr<f16>, i32
        %49 = tt.splat %48 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
        %50 = tt.addptr %49, %11 : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi32>
        %51 = tt.broadcast %50 : tensor<32x1x!tt.ptr<f16>> -> tensor<32x64x!tt.ptr<f16>>
        %52 = tt.addptr %51, %12 : tensor<32x64x!tt.ptr<f16>>, tensor<32x64xi32>
        %53 = arith.muli %35, %arg30 : i32
        %54 = tt.addptr %arg7, %53 : !tt.ptr<f16>, i32
        %55 = arith.muli %27, %arg31 : i32
        %56 = tt.addptr %54, %55 : !tt.ptr<f16>, i32
        %57 = arith.muli %28, %arg29 : i32
        %58 = tt.addptr %56, %57 : !tt.ptr<f16>, i32
        %59 = tt.splat %58 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
        %60 = tt.addptr %59, %14 : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi32>
        %61 = tt.broadcast %60 : tensor<32x1x!tt.ptr<f16>> -> tensor<32x64x!tt.ptr<f16>>
        %62 = tt.addptr %61, %12 : tensor<32x64x!tt.ptr<f16>>, tensor<32x64xi32>
        %63 = arith.extsi %33 : i32 to i64
        %64 = tt.addptr %arg4, %30 : !tt.ptr<f32>, i32
        %65 = arith.muli %26, %arg23 : i32
        %66 = tt.addptr %64, %65 : !tt.ptr<f32>, i32
        %67 = tt.addptr %arg3, %30 : !tt.ptr<f32>, i32
        %68 = arith.muli %26, %arg22 : i32
        %69 = tt.addptr %67, %68 : !tt.ptr<f32>, i32
        %70:6 = scf.for %arg34 = %20 to %33 step %c32_i32 iter_args(%arg35 = %cst, %arg36 = %cst, %arg37 = %20, %arg38 = %20, %arg39 = %20, %arg40 = %20) -> (tensor<32x64xf32>, tensor<32x64xf32>, i32, i32, i32, i32)  : i32 {
          %73 = tt.make_tensor_ptr %66, [%63], [%c1_i64], [%arg40] {order = array<i32: 0>} : <tensor<32xf32>>
          %74 = tt.make_tensor_ptr %69, [%63], [%c1_i64], [%arg39] {order = array<i32: 0>} : <tensor<32xf32>>
          %75 = tt.load %74 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32xf32>>
          %76 = tt.expand_dims %75 {axis = 0 : i32} : tensor<32xf32> -> tensor<1x32xf32>
          %77 = tt.load %73 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32xf32>>
          %78 = tt.expand_dims %77 {axis = 0 : i32} : tensor<32xf32> -> tensor<1x32xf32>
          %79 = tt.splat %arg34 : i32 -> tensor<32xi32>
          %80 = arith.addi %3, %79 : tensor<32xi32>
          %81 = tt.expand_dims %80 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
          %82 = tt.broadcast %81 : tensor<1x32xi32> -> tensor<32x32xi32>
          %83 = arith.cmpi sle, %22, %82 : tensor<32x32xi32>
          %84 = arith.select %83, %cst_1, %cst_0 : tensor<32x32xi1>, tensor<32x32xf32>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          %memspacecast = memref.memory_space_cast %alloc_7 : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
          %85 = bufferization.to_tensor %memspacecast restrict writable : memref<32x32xf32>
          %86 = arith.mulf %85, %23 : tensor<32x32xf32>
          %87 = arith.addf %84, %86 : tensor<32x32xf32>
          %88 = tt.broadcast %76 : tensor<1x32xf32> -> tensor<32x32xf32>
          %89 = arith.subf %87, %88 : tensor<32x32xf32>
          %90 = math.exp2 %89 : tensor<32x32xf32>
          %91 = arith.mulf %24, %90 : tensor<32x32xf32>
          %92 = tt.broadcast %78 : tensor<1x32xf32> -> tensor<32x32xf32>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
          %memspacecast_9 = memref.memory_space_cast %alloc_6 : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
          %93 = bufferization.to_tensor %memspacecast_9 restrict writable : memref<32x32xf32>
          %94 = arith.subf %93, %92 : tensor<32x32xf32>
          %95 = arith.mulf %91, %94 : tensor<32x32xf32>
          %96 = arith.truncf %90 : tensor<32x32xf32> to tensor<32x32xf16>
          %97 = tt.reshape %96 : tensor<32x32xf16> -> tensor<2x16x2x16xf16>
          %98 = tt.trans %97 {order = array<i32: 2, 0, 1, 3>} : tensor<2x16x2x16xf16> -> tensor<2x2x16x16xf16>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 11
          %99 = bufferization.to_memref %98 : memref<2x2x16x16xf16, #hivm.address_space<ub>>
          hivm.hir.copy ins(%99 : memref<2x2x16x16xf16, #hivm.address_space<ub>>) outs(%alloc : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
          %100 = arith.truncf %95 : tensor<32x32xf32> to tensor<32x32xf16>
          %101 = tt.reshape %100 : tensor<32x32xf16> -> tensor<2x16x2x16xf16>
          %102 = tt.trans %101 {order = array<i32: 2, 0, 1, 3>} : tensor<2x16x2x16xf16> -> tensor<2x2x16x16xf16>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 12
          %103 = bufferization.to_memref %102 : memref<2x2x16x16xf16, #hivm.address_space<ub>>
          hivm.hir.copy ins(%103 : memref<2x2x16x16xf16, #hivm.address_space<ub>>) outs(%alloc_5 : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
          %memspacecast_10 = memref.memory_space_cast %alloc_4 : memref<32x64xf32, #hivm.address_space<ub>> to memref<32x64xf32>
          %104 = bufferization.to_tensor %memspacecast_10 restrict writable : memref<32x64xf32>
          %105 = arith.addf %104, %arg35 : tensor<32x64xf32>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
          %memspacecast_11 = memref.memory_space_cast %alloc_8 : memref<32x64xf32, #hivm.address_space<ub>> to memref<32x64xf32>
          %106 = bufferization.to_tensor %memspacecast_11 restrict writable : memref<32x64xf32>
          %107 = arith.addf %106, %arg36 : tensor<32x64xf32>
          %108 = arith.addi %arg37, %c32_i32 : i32
          %109 = arith.addi %arg38, %c32_i32 : i32
          %110 = arith.addi %arg39, %c32_i32 : i32
          %111 = arith.addi %arg40, %c32_i32 : i32
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 10
          scf.yield %105, %107, %108, %109, %110, %111 : tensor<32x64xf32>, tensor<32x64xf32>, i32, i32, i32, i32
        }
        %71 = arith.truncf %70#0 : tensor<32x64xf32> to tensor<32x64xf16>
        tt.store %52, %71, %42 : tensor<32x64x!tt.ptr<f16>>
        %72 = arith.truncf %70#1 : tensor<32x64xf32> to tensor<32x64xf16>
        tt.store %62, %72, %42 : tensor<32x64x!tt.ptr<f16>>
      }
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 11
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 12
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 12
      hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 11
      %1 = tt.get_num_programs x : i32
      %2 = arith.extsi %arg14 : i32 to i64
      %3 = arith.extsi %arg18 : i32 to i64
      %4 = arith.extsi %arg20 : i32 to i64
      %5 = arith.subi %arg8, %c1_i32 : i32
      %6 = arith.extsi %arg16 : i32 to i64
      %7 = arith.extsi %arg24 : i32 to i64
      scf.for %arg33 = %0 to %arg32 step %1  : i32 {
        %8 = arith.divsi %arg33, %arg32 : i32
        %9 = arith.remsi %arg33, %arg32 : i32
        %10 = arith.divsi %9, %arg13 : i32
        %11 = tt.addptr %arg10, %8 : !tt.ptr<i32>, i32
        %12 = tt.load %11 : !tt.ptr<i32>
        %13 = tt.addptr %11, %c1_i32 : !tt.ptr<i32>, i32
        %14 = tt.load %13 : !tt.ptr<i32>
        %15 = arith.subi %14, %12 : i32
        %16 = tt.addptr %arg11, %8 : !tt.ptr<i32>, i32
        %17 = tt.load %16 : !tt.ptr<i32>
        %18 = tt.addptr %16, %c1_i32 : !tt.ptr<i32>, i32
        %19 = tt.load %18 : !tt.ptr<i32>
        %20 = arith.subi %19, %17 : i32
        %21 = arith.muli %17, %arg18 : i32
        %22 = tt.addptr %arg1, %21 : !tt.ptr<f16>, i32
        %23 = arith.muli %10, %arg19 : i32
        %24 = tt.addptr %22, %23 : !tt.ptr<f16>, i32
        %25 = arith.extsi %20 : i32 to i64
        %26 = tt.make_tensor_ptr %24, [%25, %2], [%3, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16>>
        %27 = arith.muli %17, %arg20 : i32
        %28 = tt.addptr %arg2, %27 : !tt.ptr<f16>, i32
        %29 = arith.muli %10, %arg21 : i32
        %30 = tt.addptr %28, %29 : !tt.ptr<f16>, i32
        %31 = tt.make_tensor_ptr %30, [%25, %2], [%4, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16>>
        %32 = tt.load %26 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16>>
        %33 = tt.load %31 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16>>
        %34 = arith.muli %12, %arg16 : i32
        %35 = tt.addptr %arg0, %34 : !tt.ptr<f16>, i32
        %36 = arith.muli %9, %arg17 : i32
        %37 = tt.addptr %35, %36 : !tt.ptr<f16>, i32
        %38 = arith.extsi %15 : i32 to i64
        %39 = arith.muli %12, %arg24 : i32
        %40 = tt.addptr %arg5, %39 : !tt.ptr<f16>, i32
        %41 = arith.muli %9, %arg25 : i32
        %42 = tt.addptr %40, %41 : !tt.ptr<f16>, i32
        %43:4 = scf.for %arg34 = %5 to %15 step %c32_i32 iter_args(%arg35 = %5, %arg36 = %5, %arg37 = %5, %arg38 = %5) -> (i32, i32, i32, i32)  : i32 {
          %44 = tt.make_tensor_ptr %42, [%2, %38], [%c1_i64, %7], [%c0_i32, %arg36] {order = array<i32: 0, 1>} : <tensor<64x32xf16>>
          %45 = tt.make_tensor_ptr %37, [%2, %38], [%c1_i64, %6], [%c0_i32, %arg35] {order = array<i32: 0, 1>} : <tensor<64x32xf16>>
          %46 = tt.load %45 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16>>
          %47 = tt.load %44 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16>>
          %48 = tt.dot %32, %46, %cst_1 : tensor<32x64xf16> * tensor<64x32xf16> -> tensor<32x32xf32>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 7
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%48 : tensor<32x32xf32>) outs(%alloc_7 : memref<32x32xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          %49 = tt.dot %33, %47, %cst_1 : tensor<32x64xf16> * tensor<64x32xf16> -> tensor<32x32xf32>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 8
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%49 : tensor<32x32xf32>) outs(%alloc_6 : memref<32x32xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
          %50 = tt.trans %46 {order = array<i32: 1, 0>} : tensor<64x32xf16> -> tensor<32x64xf16>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
          %51 = hivm.hir.convert_layout %alloc_5 {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<2x2x16x16xf16, #hivm.address_space<cbuf>>) -> memref<32x32xf16, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %51 : memref<32x32xf16, #hivm.address_space<cbuf>> to memref<32x32xf16>
          %52 = bufferization.to_tensor %memspacecast restrict writable : memref<32x32xf16>
          %53 = tt.dot %52, %50, %cst : tensor<32x32xf16> * tensor<32x64xf16> -> tensor<32x64xf32>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 12
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%53 : tensor<32x64xf32>) outs(%alloc_4 : memref<32x64xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 4
          %54 = tt.trans %47 {order = array<i32: 1, 0>} : tensor<64x32xf16> -> tensor<32x64xf16>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
          %55 = hivm.hir.convert_layout %alloc {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<2x2x16x16xf16, #hivm.address_space<cbuf>>) -> memref<32x32xf16, #hivm.address_space<cbuf>>
          %memspacecast_9 = memref.memory_space_cast %55 : memref<32x32xf16, #hivm.address_space<cbuf>> to memref<32x32xf16>
          %56 = bufferization.to_tensor %memspacecast_9 restrict writable : memref<32x32xf16>
          %57 = tt.dot %56, %54, %cst : tensor<32x32xf16> * tensor<32x64xf16> -> tensor<32x64xf32>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 11
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 10
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%57 : tensor<32x64xf32>) outs(%alloc_8 : memref<32x64xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
          %58 = arith.addi %arg35, %c32_i32 : i32
          %59 = arith.addi %arg36, %c32_i32 : i32
          %60 = arith.addi %arg37, %c32_i32 : i32
          %61 = arith.addi %arg38, %c32_i32 : i32
          scf.yield %58, %59, %60, %61 : i32, i32, i32, i32
        }
      }
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 7
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 8
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 10
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    tt.return
  }
}
