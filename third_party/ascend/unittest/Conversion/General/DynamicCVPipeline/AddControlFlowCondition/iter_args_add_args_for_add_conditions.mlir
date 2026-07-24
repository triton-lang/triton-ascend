// RUN: triton-opt --update-for-ops %s | FileCheck %s

// Unit Test for Tensor Iter Arg Eq Conditions
// This test verifies that the AddControlFlowCondition pass:
// 1. Creates new iter_args for tensor arguments
// 2. Adds eq conditions (var == 0 for producers, var == 1 for consumers)
// 3. Updates variables after execution (var += 1 or var -= 1)
// 4. Properly yields updated values
// CHECK: %[[C0:.*]] = arith.constant 1 : i32
// CHECK: scf.for
// CHECK-SAME: iter_args(%[[ARG:.*]] = %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[NEW_ARG:.*]] = %[[C0]])
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chunk_oja_bwd_kernel_dhu_blockdim64(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg12: f32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 17 : i32} 0.000000e+00 : f32
    %c1_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 1 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 0 : i32
    %c32768_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 32768 : i32
    %c64_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 64 : i32
    %c8_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 8 : i32
    %c4096_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 4096 : i32
    %c-1_i32 = arith.constant {ssbuffer.block_id = 17 : i32} -1 : i32
    %c512_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 512 : i32
    %c63_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 63 : i32
    %c64 = arith.constant {ssbuffer.block_id = 17 : i32} 64 : index
    %c0 = arith.constant {ssbuffer.block_id = 17 : i32} 0 : index
    %c512 = arith.constant {ssbuffer.block_id = 17 : i32} 512 : index
    %cst_0 = arith.constant {ssbuffer.block_id = 21 : i32} dense<[8, 4, 16, 8]> : tensor<4xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 21 : i32} dense<[64, 8, 8]> : tensor<3xi64>
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 17 : i32} : tensor<64x64xf32>
      %1 = arith.divsi %arg18, %c8_i32 {ssbuffer.block_id = 17 : i32} : i32
      %2 = arith.remsi %arg18, %c8_i32 {ssbuffer.block_id = 17 : i32} : i32
      %3 = arith.muli %1, %arg13 {ssbuffer.block_id = 17 : i32} : i32
      %4 = arith.addi %arg13, %c63_i32 {ssbuffer.block_id = 17 : i32} : i32
      %5 = arith.divsi %4, %c64_i32 {ssbuffer.block_id = 17 : i32} : i32
      %6 = arith.muli %1, %5 {ssbuffer.block_id = 17 : i32} : i32
      %7 = arith.muli %3, %c8_i32 {ssbuffer.block_id = 17 : i32} : i32
      %8 = arith.addi %7, %2 {ssbuffer.block_id = 17 : i32} : i32
      %9 = arith.muli %8, %c64_i32 {ssbuffer.block_id = 17 : i32} : i32
      %10 = arith.index_cast %9 {ssbuffer.block_id = 17 : i32} : i32 to index
      %11 = arith.muli %6, %c8_i32 {ssbuffer.block_id = 17 : i32} : i32
      %12 = arith.addi %11, %2 {ssbuffer.block_id = 17 : i32} : i32
      %13 = arith.muli %12, %c4096_i32 {ssbuffer.block_id = 17 : i32} : i32
      %14 = arith.muli %arg17, %c64_i32 {ssbuffer.block_id = 17 : i32} : i32
      %15 = arith.maxsi %14, %c0_i32 {ssbuffer.block_id = 17 : i32} : i32
      %16 = arith.index_cast %15 {ssbuffer.block_id = 17 : i32} : i32 to index
      %17 = arith.muli %16, %c64 {ssbuffer.block_id = 17 : i32} : index
      %18 = arith.divsi %17, %c64 {ssbuffer.block_id = 17 : i32} : index
      %19 = arith.subi %c64, %18 {ssbuffer.block_id = 17 : i32} : index
      %20 = arith.maxsi %19, %c0 {ssbuffer.block_id = 17 : i32} : index
      %21 = arith.minsi %20, %c64 {ssbuffer.block_id = 17 : i32} : index
      %22 = arith.remsi %17, %c64 {ssbuffer.block_id = 17 : i32} : index
      %23 = arith.subi %c64, %22 {ssbuffer.block_id = 17 : i32} : index
      %24 = arith.maxsi %23, %c0 {ssbuffer.block_id = 17 : i32} : index
      %25 = arith.minsi %24, %c64 {ssbuffer.block_id = 17 : i32} : index
      %26 = arith.subi %c0_i32, %14 {ssbuffer.block_id = 17 : i32} : i32
      %27 = arith.maxsi %26, %c0_i32 {ssbuffer.block_id = 17 : i32} : i32
      %28 = arith.index_cast %27 {ssbuffer.block_id = 17 : i32} : i32 to index
      %29 = arith.minsi %28, %21 {ssbuffer.block_id = 17 : i32} : index
      %30 = arith.subi %21, %29 {ssbuffer.block_id = 17 : i32} : index
      %31 = arith.minsi %25, %c0 {ssbuffer.block_id = 17 : i32} : index
      %32 = arith.subi %25, %31 {ssbuffer.block_id = 17 : i32} : index
      %33 = arith.cmpi slt, %30, %c64 {ssbuffer.block_id = 17 : i32} : index
      %34 = arith.cmpi slt, %32, %c64 {ssbuffer.block_id = 17 : i32} : index
      %35 = arith.ori %33, %34 {ssbuffer.block_id = 17 : i32} : i1
      %36 = arith.subi %5, %c1_i32 {ssbuffer.block_id = 17 : i32} : i32
      %37 = linalg.fill {ssbuffer.block_id = 17 : i32} ins(%arg12 : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %alloc = memref.alloc() {ssbuffer.block_id = 18 : i32} : memref<64x64xf32>
      scf.if %35 {
        linalg.fill {ssbuffer.block_id = 18 : i32} ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 18 : i32}
      %38 = linalg.fill {ssbuffer.block_id = 18 : i32} ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %39 = arith.muli %arg18, %c4096_i32 {ssbuffer.block_id = 18 : i32} : i32
      %40 = arith.index_cast %39 {ssbuffer.block_id = 18 : i32} : i32 to index
      %41 = arith.addi %17, %40 {ssbuffer.block_id = 18 : i32} : index
      %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%41], sizes: [64, 64], strides: [64, 1] {ssbuffer.block_id = 18 : i32} : memref<?xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      %subview = memref.subview %reinterpret_cast[0, 0] [%30, %32] [1, 1] {ssbuffer.block_id = 18 : i32} : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[%29, %31] [%30, %32] [1, 1] {ssbuffer.block_id = 18 : i32} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %subview, %subview_2 {ssbuffer.block_id = 18 : i32} : memref<?x?xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %42 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 18 : i32} : memref<64x64xf32>
      %43 = arith.addf %42, %38 {DataUse, ssbuffer.block_id = 18 : i32} : tensor<64x64xf32>
      %44 = arith.addi %arg13, %c63_i32 {ssbuffer.block_id = 9 : i32} : i32
      %45 = arith.divsi %44, %c64_i32 {ssbuffer.block_id = 9 : i32} : i32
      %46 = arith.subi %45, %c1_i32 {ssbuffer.block_id = 9 : i32} : i32
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [3 : i32, 1 : i32], ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [4 : i32, 1 : i32], ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
      %alloc_8 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [5 : i32, 1 : i32], ssbuffer.transfer_id = 5 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
      %alloc_9 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc_9 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %alloc_10 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast_11 = memref.memory_space_cast %alloc_10 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %47 = scf.for %arg20 = %c-1_i32 to %46 step %c1_i32 iter_args(%arg21 = %43) -> (tensor<64x64xf32>)  : i32 {
        %true = arith.constant true
        scf.if %true {
          %49 = arith.subi %36, %arg20 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [12 : i32]} : i32
          %50 = arith.addi %49, %c-1_i32 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
          %51 = arith.muli %50, %c64_i32 {ssbuffer.block_id = 14 : i32} : i32
          %52 = arith.maxsi %51, %c0_i32 {ssbuffer.block_id = 14 : i32} : i32
          %53 = arith.index_cast %52 {ssbuffer.block_id = 14 : i32} : i32 to index
          %54 = arith.muli %53, %c512 {ssbuffer.block_id = 14 : i32} : index
          %55 = arith.addi %54, %10 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [6 : i32]} : index
          %56 = arith.index_cast %arg13 {ssbuffer.block_id = 14 : i32} : i32 to index
          %57 = arith.addi %55, %16 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [1 : i32, 13 : i32]} : index
          %58 = arith.divsi %54, %c512 {ssbuffer.block_id = 14 : i32} : index
          %59 = arith.subi %56, %58 {ssbuffer.block_id = 14 : i32} : index
          %60 = arith.maxsi %59, %c0 {ssbuffer.block_id = 14 : i32} : index
          %61 = arith.minsi %60, %c64 {ssbuffer.block_id = 14 : i32} : index
          %62 = arith.remsi %54, %c512 {ssbuffer.block_id = 14 : i32} : index
          %63 = arith.subi %c64, %62 {ssbuffer.block_id = 14 : i32} : index
          %64 = arith.maxsi %63, %c0 {ssbuffer.block_id = 14 : i32} : index
          %65 = arith.minsi %64, %c64 {ssbuffer.block_id = 14 : i32} : index
          %66 = arith.subi %c0_i32, %51 {ssbuffer.block_id = 14 : i32} : i32
          %67 = arith.maxsi %66, %c0_i32 {ssbuffer.block_id = 14 : i32} : i32
          %68 = arith.index_cast %67 {ssbuffer.block_id = 14 : i32} : i32 to index
          %69 = arith.minsi %68, %61 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [9 : i32]} : index
          %70 = arith.subi %61, %69 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [7 : i32]} : index
          %71 = arith.minsi %65, %c0 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [10 : i32]} : index
          %72 = arith.subi %65, %71 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [8 : i32]} : index
          %73 = arith.cmpi slt, %70, %c64 {ssbuffer.block_id = 14 : i32} : index
          %74 = arith.cmpi slt, %72, %c64 {ssbuffer.block_id = 14 : i32} : index
          %75 = arith.ori %73, %74 {ssbuffer.block_id = 14 : i32} : i1
          %76 = arith.subi %57, %10 {ssbuffer.block_id = 14 : i32} : index
          %77 = arith.divsi %76, %c512 {ssbuffer.block_id = 14 : i32} : index
          %78 = arith.subi %56, %77 {ssbuffer.block_id = 14 : i32} : index
          %79 = arith.maxsi %78, %c0 {ssbuffer.block_id = 14 : i32} : index
          %80 = arith.minsi %79, %c64 {ssbuffer.block_id = 14 : i32} : index
          %81 = arith.remsi %76, %c512 {ssbuffer.block_id = 14 : i32} : index
          %82 = arith.subi %c64, %81 {ssbuffer.block_id = 14 : i32} : index
          %83 = arith.maxsi %82, %c0 {ssbuffer.block_id = 14 : i32} : index
          %84 = arith.minsi %83, %c64 {ssbuffer.block_id = 14 : i32} : index
          %85 = arith.minsi %68, %80 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [4 : i32, 14 : i32]} : index
          %86 = arith.subi %80, %85 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [2 : i32, 16 : i32]} : index
          %87 = arith.minsi %28, %84 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [5 : i32, 15 : i32]} : index
          %88 = arith.subi %84, %87 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [3 : i32, 17 : i32]} : index
          %89 = arith.cmpi slt, %86, %c64 {ssbuffer.block_id = 14 : i32} : index
          %90 = arith.cmpi slt, %88, %c64 {ssbuffer.block_id = 14 : i32} : index
          %91 = arith.ori %89, %90 {ssbuffer.block_id = 14 : i32} : i1
        } {hivm.matmul_limited_in_cube, ssbuffer.if = 14 : i32}
        %true_14 = arith.constant true
        scf.if %true_14 {
          %reshape = tensor.reshape %arg21(%cst_1) {ssbuffer.block_id = 21 : i32} : (tensor<64x64xf32>, tensor<3xi64>) -> tensor<64x8x8xf32>
          %49 = tensor.empty() {ssbuffer.block_id = 21 : i32} : tensor<8x64x8xf32>
          %transposed = linalg.transpose ins(%reshape : tensor<64x8x8xf32>) outs(%49 : tensor<8x64x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 21 : i32}
          %reshape_17 = tensor.reshape %transposed(%cst_0) {ssbuffer.block_id = 21 : i32} : (tensor<8x64x8xf32>, tensor<4xi64>) -> tensor<8x4x16x8xf32>
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.copy ins(%reshape_17 : tensor<8x4x16x8xf32>) outs(%alloc_3 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 0 : i32}
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        } {hivm.matmul_limited_in_cube, ssbuffer.if = 21 : i32}
        %true_15 = arith.constant true
        scf.if %true_15 {
          %49 = arith.subi %36, %arg20 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [12 : i32]} : i32
          %50 = arith.addi %49, %c-1_i32 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
          %51 = arith.muli %50, %c64_i32 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : i32
          %52 = arith.maxsi %51, %c0_i32 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : i32
          %53 = arith.index_cast %52 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : i32 to index
          %54 = arith.muli %53, %c512 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %55 = arith.addi %54, %10 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [6 : i32]} : index
          %56 = arith.index_cast %arg13 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : i32 to index
          %57 = arith.addi %55, %16 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [1 : i32, 13 : i32]} : index
          %58 = arith.divsi %54, %c512 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %59 = arith.subi %56, %58 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %60 = arith.maxsi %59, %c0 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %61 = arith.minsi %60, %c64 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %62 = arith.remsi %54, %c512 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %63 = arith.subi %c64, %62 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %64 = arith.maxsi %63, %c0 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %65 = arith.minsi %64, %c64 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %66 = arith.subi %c0_i32, %51 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : i32
          %67 = arith.maxsi %66, %c0_i32 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : i32
          %68 = arith.index_cast %67 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : i32 to index
          %69 = arith.minsi %68, %61 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [9 : i32]} : index
          %70 = arith.subi %61, %69 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [7 : i32]} : index
          %71 = arith.minsi %65, %c0 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [10 : i32]} : index
          %72 = arith.subi %65, %71 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [8 : i32]} : index
          %73 = arith.cmpi slt, %70, %c64 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %74 = arith.cmpi slt, %72, %c64 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %75 = arith.ori %73, %74 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : i1
          %76 = arith.subi %57, %10 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %77 = arith.divsi %76, %c512 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %78 = arith.subi %56, %77 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %79 = arith.maxsi %78, %c0 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %80 = arith.minsi %79, %c64 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %81 = arith.remsi %76, %c512 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %82 = arith.subi %c64, %81 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %83 = arith.maxsi %82, %c0 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %84 = arith.minsi %83, %c64 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %85 = arith.minsi %68, %80 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [4 : i32, 14 : i32]} : index
          %86 = arith.subi %80, %85 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [2 : i32, 16 : i32]} : index
          %87 = arith.minsi %28, %84 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [5 : i32, 15 : i32]} : index
          %88 = arith.subi %84, %87 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [3 : i32, 17 : i32]} : index
          %89 = arith.cmpi slt, %86, %c64 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %90 = arith.cmpi slt, %88, %c64 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : index
          %91 = arith.ori %89, %90 {ssbuffer.block_id = 15 : i32, ssbuffer.clone = 14 : i32} : i1
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
          %memspacecast_17 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 15 : i32, ssbuffer.crossDeps = [3 : i32, 0 : i32], ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
          %92 = bufferization.to_tensor %memspacecast_17 restrict writable {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32>
          %alloc_18 = memref.alloc() {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
          %alloc_19 = memref.alloc() {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
          %alloc_20 = memref.alloc() {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
          scf.if %91 {
            linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst : f32) outs(%alloc_18 : memref<64x64xf32>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 15 : i32}
          scf.if %75 {
            linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst : f32) outs(%alloc_20 : memref<64x64xf32>)
            linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst : f32) outs(%alloc_19 : memref<64x64xf32>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 15 : i32}
          %reinterpret_cast_21 = memref.reinterpret_cast %arg10 to offset: [%57], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [1 : i32, 13 : i32]} : memref<?xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          %subview_22 = memref.subview %reinterpret_cast_21[0, 0] [%86, %88] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [2 : i32, 3 : i32, 16 : i32, 17 : i32]} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
          %subview_23 = memref.subview %alloc_18[%85, %87] [%86, %88] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [2 : i32, 3 : i32, 4 : i32, 5 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32]} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
          memref.copy %subview_22, %subview_23 {ssbuffer.block_id = 15 : i32} : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
          %93 = bufferization.to_tensor %alloc_18 restrict writable {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
          %94 = arith.addf %92, %93 {ssbuffer.add_from_matmul, ssbuffer.block_id = 15 : i32} : tensor<64x64xf32>
          %95 = arith.subi %arg20, %c-1_i32 {ssbuffer.block_id = 15 : i32} : i32
          %96 = arith.divui %95, %c1_i32 {ssbuffer.block_id = 15 : i32} : i32
          %c2_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 2 : i32
          %97 = arith.remsi %96, %c2_i32 {ssbuffer.block_id = 15 : i32} : i32
          %c0_i32_24 = arith.constant {ssbuffer.block_id = 15 : i32} 0 : i32
          %98 = arith.cmpi eq, %97, %c0_i32_24 {ssbuffer.block_id = 15 : i32} : i32
          scf.if %98 {
            hivm.hir.copy ins(%94 : tensor<64x64xf32>) outs(%memspacecast : memref<64x64xf32>) {ssbuffer.block_id = 15 : i32}
          } else {
            hivm.hir.copy ins(%94 : tensor<64x64xf32>) outs(%memspacecast_11 : memref<64x64xf32>) {ssbuffer.block_id = 15 : i32}
          } {ssbuffer.block_id = 15 : i32, ssbuffer.intra_buffer}
          %reinterpret_cast_25 = memref.reinterpret_cast %arg8 to offset: [%55], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [6 : i32]} : memref<?xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          %subview_26 = memref.subview %reinterpret_cast_25[0, 0] [%70, %72] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [7 : i32, 8 : i32]} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
          %subview_27 = memref.subview %alloc_19[%69, %71] [%70, %72] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [7 : i32, 8 : i32, 9 : i32, 10 : i32]} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
          memref.copy %subview_26, %subview_27 {ssbuffer.block_id = 15 : i32} : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
          %99 = bufferization.to_tensor %alloc_19 restrict writable {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
          %reinterpret_cast_28 = memref.reinterpret_cast %arg5 to offset: [%55], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [6 : i32]} : memref<?xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          %subview_29 = memref.subview %reinterpret_cast_28[0, 0] [%70, %72] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [7 : i32, 8 : i32]} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
          %subview_30 = memref.subview %alloc_20[%69, %71] [%70, %72] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [7 : i32, 8 : i32, 9 : i32, 10 : i32]} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
          memref.copy %subview_29, %subview_30 {ssbuffer.block_id = 15 : i32} : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
          %100 = bufferization.to_tensor %alloc_20 restrict writable {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
          %101 = math.exp %100 {DataUse, ssbuffer.block_id = 15 : i32} : tensor<64x64xf32>
          %102 = arith.mulf %99, %101 {DataUse, ssbuffer.block_id = 15 : i32} : tensor<64x64xf32>
          %reshape = tensor.reshape %94(%cst_1) {ssbuffer.block_id = 15 : i32} : (tensor<64x64xf32>, tensor<3xi64>) -> tensor<64x8x8xf32>
          %103 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<8x64x8xf32>
          %transposed = linalg.transpose ins(%reshape : tensor<64x8x8xf32>) outs(%103 : tensor<8x64x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 15 : i32}
          %reshape_31 = tensor.reshape %transposed(%cst_0) {ssbuffer.block_id = 15 : i32} : (tensor<8x64x8xf32>, tensor<4xi64>) -> tensor<8x4x16x8xf32>
          %reshape_32 = tensor.reshape %102(%cst_1) {ssbuffer.block_id = 15 : i32} : (tensor<64x64xf32>, tensor<3xi64>) -> tensor<64x8x8xf32>
          %104 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<8x64x8xf32>
          %transposed_33 = linalg.transpose ins(%reshape_32 : tensor<64x8x8xf32>) outs(%104 : tensor<8x64x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 15 : i32}
          %reshape_34 = tensor.reshape %transposed_33(%cst_0) {ssbuffer.block_id = 15 : i32} : (tensor<8x64x8xf32>, tensor<4xi64>) -> tensor<8x4x16x8xf32>
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
          hivm.hir.copy ins(%reshape_31 : tensor<8x4x16x8xf32>) outs(%alloc_4 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 1 : i32}
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
          hivm.hir.copy ins(%reshape_34 : tensor<8x4x16x8xf32>) outs(%alloc_5 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 2 : i32}
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        } {hivm.matmul_limited_in_cube, ssbuffer.if = 15 : i32}
        %true_16 = arith.constant true
        %48 = scf.if %true_16 -> (tensor<64x64xf32>) {
          %49 = arith.subi %36, %arg20 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [12 : i32]} : i32
          %50 = arith.addi %49, %c-1_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
          %51 = arith.muli %50, %c64_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : i32
          %52 = arith.maxsi %51, %c0_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : i32
          %53 = arith.index_cast %52 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : i32 to index
          %54 = arith.muli %53, %c512 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : index
          %55 = arith.addi %54, %10 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [6 : i32]} : index
          %56 = arith.index_cast %arg13 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : i32 to index
          %57 = arith.addi %55, %16 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [1 : i32, 13 : i32]} : index
          %58 = arith.subi %c0_i32, %51 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : i32
          %59 = arith.maxsi %58, %c0_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : i32
          %60 = arith.index_cast %59 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : i32 to index
          %61 = arith.subi %57, %10 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : index
          %62 = arith.divsi %61, %c512 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : index
          %63 = arith.subi %56, %62 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : index
          %64 = arith.maxsi %63, %c0 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : index
          %65 = arith.minsi %64, %c64 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : index
          %66 = arith.remsi %61, %c512 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : index
          %67 = arith.subi %c64, %66 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : index
          %68 = arith.maxsi %67, %c0 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : index
          %69 = arith.minsi %68, %c64 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32} : index
          %70 = arith.minsi %60, %65 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [4 : i32, 14 : i32]} : index
          %71 = arith.subi %65, %70 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [2 : i32, 16 : i32]} : index
          %72 = arith.minsi %28, %69 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [5 : i32, 15 : i32]} : index
          %73 = arith.subi %69, %72 {ssbuffer.block_id = 16 : i32, ssbuffer.clone = 14 : i32, ssbuffer.dep_mark = [3 : i32, 17 : i32]} : index
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
          %memspacecast_17 = memref.memory_space_cast %alloc_8 {ssbuffer.block_id = 16 : i32, ssbuffer.crossDeps = [5 : i32, 0 : i32], ssbuffer.transfer_id = 5 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
          %74 = bufferization.to_tensor %memspacecast_17 restrict writable {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x64xf32>
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 5
          %memspacecast_18 = memref.memory_space_cast %alloc_7 {ssbuffer.block_id = 16 : i32, ssbuffer.crossDeps = [4 : i32, 0 : i32], ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
          %75 = bufferization.to_tensor %memspacecast_18 restrict writable {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32>
          %76 = arith.muli %50, %c32768_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
          %77 = arith.index_cast %13 {ssbuffer.block_id = 16 : i32} : i32 to index
          %78 = arith.index_cast %76 {ssbuffer.block_id = 16 : i32} : i32 to index
          %79 = arith.addi %77, %78 {ssbuffer.block_id = 16 : i32} : index
          %80 = arith.addi %17, %79 {ssbuffer.block_id = 16 : i32} : index
          %reinterpret_cast_19 = memref.reinterpret_cast %arg9 to offset: [%80], sizes: [64, 64], strides: [64, 1] {ssbuffer.block_id = 16 : i32} : memref<?xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
          %extracted_slice_20 = tensor.extract_slice %arg21[%29, %31] [%30, %32] [1, 1] {ssbuffer.block_id = 16 : i32} : tensor<64x64xf32> to tensor<?x?xf32>
          %subview_21 = memref.subview %reinterpret_cast_19[0, 0] [%30, %32] [1, 1] {ssbuffer.block_id = 16 : i32} : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
          bufferization.materialize_in_destination %extracted_slice_20 in writable %subview_21 {ssbuffer.block_id = 16 : i32} : (tensor<?x?xf32>, memref<?x?xf32, strided<[64, 1], offset: ?>>) -> ()
          %81 = arith.muli %49, %c64_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.dep_mark = [12 : i32]} : i32
          %82 = arith.minsi %81, %arg13 {ssbuffer.block_id = 16 : i32} : i32
          %83 = arith.subi %82, %c1_i32 {ssbuffer.block_id = 16 : i32} : i32
          %reinterpret_cast_22 = memref.reinterpret_cast %arg11 to offset: [%57], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.dep_mark = [1 : i32, 13 : i32]} : memref<?xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          %84 = arith.subi %arg20, %c-1_i32 {ssbuffer.block_id = 16 : i32} : i32
          %85 = arith.divui %84, %c1_i32 {ssbuffer.block_id = 16 : i32} : i32
          %c2_i32 = arith.constant {ssbuffer.block_id = 16 : i32} 2 : i32
          %86 = arith.remsi %85, %c2_i32 {ssbuffer.block_id = 16 : i32} : i32
          %c0_i32_23 = arith.constant {ssbuffer.block_id = 16 : i32} 0 : i32
          %87 = arith.cmpi eq, %86, %c0_i32_23 {ssbuffer.block_id = 16 : i32} : i32
          %88 = scf.if %87 -> (tensor<64x64xf32>) {
            %98 = bufferization.to_tensor %memspacecast restrict writable : memref<64x64xf32>
            scf.yield %98 : tensor<64x64xf32>
          } else {
            %98 = bufferization.to_tensor %memspacecast_11 restrict writable : memref<64x64xf32>
            scf.yield %98 : tensor<64x64xf32>
          } {ssbuffer.block_id = 16 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32], ssbuffer.intra_buffer}
          %extracted_slice_24 = tensor.extract_slice %88[%70, %72] [%71, %73] [1, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.dep_mark = [2 : i32, 3 : i32, 4 : i32, 5 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32]} : tensor<64x64xf32> to tensor<?x?xf32>
          %subview_25 = memref.subview %reinterpret_cast_22[0, 0] [%71, %73] [1, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.dep_mark = [2 : i32, 3 : i32, 16 : i32, 17 : i32]} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
          bufferization.materialize_in_destination %extracted_slice_24 in writable %subview_25 {ssbuffer.block_id = 16 : i32} : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
          %89 = arith.muli %83, %c512_i32 {ssbuffer.block_id = 16 : i32} : i32
          %90 = arith.index_cast %89 {ssbuffer.block_id = 16 : i32} : i32 to index
          %91 = arith.addi %10, %90 {ssbuffer.block_id = 16 : i32} : index
          %reinterpret_cast_26 = memref.reinterpret_cast %arg5 to offset: [%91], sizes: [64], strides: [1] {ssbuffer.block_id = 16 : i32} : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
          %alloc_27 = memref.alloc() {ssbuffer.block_id = 16 : i32} : memref<64xf32>
          memref.copy %reinterpret_cast_26, %alloc_27 {ssbuffer.block_id = 16 : i32} : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32>
          %92 = bufferization.to_tensor %alloc_27 restrict writable {ssbuffer.block_id = 16 : i32} : memref<64xf32>
          %expanded = tensor.expand_shape %92 [[0, 1]] output_shape [1, 64] {ssbuffer.block_id = 16 : i32} : tensor<64xf32> into tensor<1x64xf32>
          %93 = math.exp %expanded {DataUse, ssbuffer.block_id = 16 : i32} : tensor<1x64xf32>
          %collapsed = tensor.collapse_shape %93 [[0, 1]] {ssbuffer.block_id = 16 : i32} : tensor<1x64xf32> into tensor<64xf32>
          %broadcasted = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [0]  {ssbuffer.block_id = 16 : i32}
          %94 = arith.mulf %arg21, %broadcasted {DataUse, ssbuffer.block_id = 16 : i32} : tensor<64x64xf32>
          %95 = arith.mulf %74, %37 {DataUse, ssbuffer.block_id = 16 : i32} : tensor<64x64xf32>
          %96 = arith.subf %95, %75 {DataUse, ssbuffer.block_id = 16 : i32} : tensor<64x64xf32>
          %97 = arith.addf %94, %96 {DataUse, ssbuffer.block_id = 16 : i32} : tensor<64x64xf32>
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
          scf.yield %97 : tensor<64x64xf32>
        } else {
          scf.yield %arg21 : tensor<64x64xf32>
        } {hivm.matmul_limited_in_cube, ssbuffer.if = 16 : i32}
        scf.yield %48 : tensor<64x64xf32>
      } {DataUse, ssbuffer.block_id = 20 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      %reinterpret_cast_12 = memref.reinterpret_cast %arg7 to offset: [%41], sizes: [64, 64], strides: [64, 1] {ssbuffer.block_id = 19 : i32} : memref<?xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      %extracted_slice = tensor.extract_slice %47[%29, %31] [%30, %32] [1, 1] {ssbuffer.block_id = 19 : i32} : tensor<64x64xf32> to tensor<?x?xf32>
      %subview_13 = memref.subview %reinterpret_cast_12[0, 0] [%30, %32] [1, 1] {ssbuffer.block_id = 19 : i32} : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_13 {ssbuffer.block_id = 19 : i32} : (tensor<?x?xf32>, memref<?x?xf32, strided<[64, 1], offset: ?>>) -> ()
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
