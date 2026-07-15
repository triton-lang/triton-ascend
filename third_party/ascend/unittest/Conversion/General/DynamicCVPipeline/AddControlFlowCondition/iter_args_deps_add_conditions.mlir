// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// CHECK: scf.for {{.*}} iter_args({{.*}}, %arg27 = %c1_i32{{.*}}) ->
// CHECK: {{.*}}
// CHECK: arith.cmpi eq, %arg27, %c1_i32{{.*}} : i32
// CHECK: arith.andi {{.*}}
// CHECK: scf.if {{.*}} {
// CHECK: {{.*}}
// CHECK: arith.subi %arg27, %c1_i32{{.*}} : i32
// CHECK: scf.yield {{.*}}
// CHECK: } {hivm.matmul_limited_in_cube, ssbuffer.if = 21 : i32}
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
        %48 = arith.subi %36, %arg20 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [12 : i32]} : i32
        %49 = arith.addi %48, %c-1_i32 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %50 = arith.muli %49, %c64_i32 {ssbuffer.block_id = 14 : i32} : i32
        %51 = arith.maxsi %50, %c0_i32 {ssbuffer.block_id = 14 : i32} : i32
        %52 = arith.index_cast %51 {ssbuffer.block_id = 14 : i32} : i32 to index
        %53 = arith.muli %52, %c512 {ssbuffer.block_id = 14 : i32} : index
        %54 = arith.addi %53, %10 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [6 : i32]} : index
        %55 = arith.index_cast %arg13 {ssbuffer.block_id = 14 : i32} : i32 to index
        %56 = arith.addi %54, %16 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [1 : i32, 13 : i32]} : index
        %57 = arith.divsi %53, %c512 {ssbuffer.block_id = 14 : i32} : index
        %58 = arith.subi %55, %57 {ssbuffer.block_id = 14 : i32} : index
        %59 = arith.maxsi %58, %c0 {ssbuffer.block_id = 14 : i32} : index
        %60 = arith.minsi %59, %c64 {ssbuffer.block_id = 14 : i32} : index
        %61 = arith.remsi %53, %c512 {ssbuffer.block_id = 14 : i32} : index
        %62 = arith.subi %c64, %61 {ssbuffer.block_id = 14 : i32} : index
        %63 = arith.maxsi %62, %c0 {ssbuffer.block_id = 14 : i32} : index
        %64 = arith.minsi %63, %c64 {ssbuffer.block_id = 14 : i32} : index
        %65 = arith.subi %c0_i32, %50 {ssbuffer.block_id = 14 : i32} : i32
        %66 = arith.maxsi %65, %c0_i32 {ssbuffer.block_id = 14 : i32} : i32
        %67 = arith.index_cast %66 {ssbuffer.block_id = 14 : i32} : i32 to index
        %68 = arith.minsi %67, %60 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [9 : i32]} : index
        %69 = arith.subi %60, %68 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [7 : i32]} : index
        %70 = arith.minsi %64, %c0 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [10 : i32]} : index
        %71 = arith.subi %64, %70 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [8 : i32]} : index
        %72 = arith.cmpi slt, %69, %c64 {ssbuffer.block_id = 14 : i32} : index
        %73 = arith.cmpi slt, %71, %c64 {ssbuffer.block_id = 14 : i32} : index
        %74 = arith.ori %72, %73 {ssbuffer.block_id = 14 : i32} : i1
        %75 = arith.subi %56, %10 {ssbuffer.block_id = 14 : i32} : index
        %76 = arith.divsi %75, %c512 {ssbuffer.block_id = 14 : i32} : index
        %77 = arith.subi %55, %76 {ssbuffer.block_id = 14 : i32} : index
        %78 = arith.maxsi %77, %c0 {ssbuffer.block_id = 14 : i32} : index
        %79 = arith.minsi %78, %c64 {ssbuffer.block_id = 14 : i32} : index
        %80 = arith.remsi %75, %c512 {ssbuffer.block_id = 14 : i32} : index
        %81 = arith.subi %c64, %80 {ssbuffer.block_id = 14 : i32} : index
        %82 = arith.maxsi %81, %c0 {ssbuffer.block_id = 14 : i32} : index
        %83 = arith.minsi %82, %c64 {ssbuffer.block_id = 14 : i32} : index
        %84 = arith.minsi %67, %79 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [4 : i32, 14 : i32]} : index
        %85 = arith.subi %79, %84 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [2 : i32, 16 : i32]} : index
        %86 = arith.minsi %28, %83 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [5 : i32, 15 : i32]} : index
        %87 = arith.subi %83, %86 {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [3 : i32, 17 : i32]} : index
        %88 = arith.cmpi slt, %85, %c64 {ssbuffer.block_id = 14 : i32} : index
        %89 = arith.cmpi slt, %87, %c64 {ssbuffer.block_id = 14 : i32} : index
        %90 = arith.ori %88, %89 {ssbuffer.block_id = 14 : i32} : i1
        %reshape = tensor.reshape %arg21(%cst_1) {ssbuffer.block_id = 21 : i32} : (tensor<64x64xf32>, tensor<3xi64>) -> tensor<64x8x8xf32>
        %91 = tensor.empty() {ssbuffer.block_id = 21 : i32} : tensor<8x64x8xf32>
        %transposed = linalg.transpose ins(%reshape : tensor<64x8x8xf32>) outs(%91 : tensor<8x64x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 21 : i32}
        %reshape_14 = tensor.reshape %transposed(%cst_0) {ssbuffer.block_id = 21 : i32} : (tensor<8x64x8xf32>, tensor<4xi64>) -> tensor<8x4x16x8xf32>
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.copy ins(%reshape_14 : tensor<8x4x16x8xf32>) outs(%alloc_3 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
        %memspacecast_15 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 15 : i32, ssbuffer.crossDeps = [3 : i32, 0 : i32], ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %92 = bufferization.to_tensor %memspacecast_15 restrict writable {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32>
        %alloc_16 = memref.alloc() {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
        %alloc_17 = memref.alloc() {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
        %alloc_18 = memref.alloc() {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
        scf.if %90 {
          linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst : f32) outs(%alloc_16 : memref<64x64xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 15 : i32}
        scf.if %74 {
          linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst : f32) outs(%alloc_18 : memref<64x64xf32>)
          linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst : f32) outs(%alloc_17 : memref<64x64xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 15 : i32}
        %reinterpret_cast_19 = memref.reinterpret_cast %arg10 to offset: [%56], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [1 : i32, 13 : i32]} : memref<?xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
        %subview_20 = memref.subview %reinterpret_cast_19[0, 0] [%85, %87] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [2 : i32, 3 : i32, 16 : i32, 17 : i32]} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %subview_21 = memref.subview %alloc_16[%84, %86] [%85, %87] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [2 : i32, 3 : i32, 4 : i32, 5 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32]} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_20, %subview_21 {ssbuffer.block_id = 15 : i32} : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        %93 = bufferization.to_tensor %alloc_16 restrict writable {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
        %94 = arith.addf %92, %93 {ssbuffer.add_from_matmul, ssbuffer.block_id = 15 : i32} : tensor<64x64xf32>
        %95 = arith.subi %arg20, %c-1_i32 {ssbuffer.block_id = 15 : i32} : i32
        %96 = arith.divui %95, %c1_i32 {ssbuffer.block_id = 15 : i32} : i32
        %c2_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 2 : i32
        %97 = arith.remsi %96, %c2_i32 {ssbuffer.block_id = 15 : i32} : i32
        %c0_i32_22 = arith.constant {ssbuffer.block_id = 15 : i32} 0 : i32
        %98 = arith.cmpi eq, %97, %c0_i32_22 {ssbuffer.block_id = 15 : i32} : i32
        scf.if %98 {
          hivm.hir.copy ins(%94 : tensor<64x64xf32>) outs(%memspacecast : memref<64x64xf32>) {ssbuffer.block_id = 15 : i32}
        } else {
          hivm.hir.copy ins(%94 : tensor<64x64xf32>) outs(%memspacecast_11 : memref<64x64xf32>) {ssbuffer.block_id = 15 : i32}
        } {ssbuffer.block_id = 15 : i32, ssbuffer.intra_buffer}
        %reinterpret_cast_23 = memref.reinterpret_cast %arg8 to offset: [%54], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [6 : i32]} : memref<?xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
        %subview_24 = memref.subview %reinterpret_cast_23[0, 0] [%69, %71] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [7 : i32, 8 : i32]} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %subview_25 = memref.subview %alloc_17[%68, %70] [%69, %71] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [7 : i32, 8 : i32, 9 : i32, 10 : i32]} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_24, %subview_25 {ssbuffer.block_id = 15 : i32} : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        %99 = bufferization.to_tensor %alloc_17 restrict writable {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
        %reinterpret_cast_26 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [6 : i32]} : memref<?xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
        %subview_27 = memref.subview %reinterpret_cast_26[0, 0] [%69, %71] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [7 : i32, 8 : i32]} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %subview_28 = memref.subview %alloc_18[%68, %70] [%69, %71] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.dep_mark = [7 : i32, 8 : i32, 9 : i32, 10 : i32]} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_27, %subview_28 {ssbuffer.block_id = 15 : i32} : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        %100 = bufferization.to_tensor %alloc_18 restrict writable {ssbuffer.block_id = 15 : i32} : memref<64x64xf32>
        %101 = math.exp %100 {DataUse, ssbuffer.block_id = 15 : i32} : tensor<64x64xf32>
        %102 = arith.mulf %99, %101 {DataUse, ssbuffer.block_id = 15 : i32} : tensor<64x64xf32>
        %reshape_29 = tensor.reshape %94(%cst_1) {ssbuffer.block_id = 15 : i32} : (tensor<64x64xf32>, tensor<3xi64>) -> tensor<64x8x8xf32>
        %103 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<8x64x8xf32>
        %transposed_30 = linalg.transpose ins(%reshape_29 : tensor<64x8x8xf32>) outs(%103 : tensor<8x64x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 15 : i32}
        %reshape_31 = tensor.reshape %transposed_30(%cst_0) {ssbuffer.block_id = 15 : i32} : (tensor<8x64x8xf32>, tensor<4xi64>) -> tensor<8x4x16x8xf32>
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
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
        %memspacecast_35 = memref.memory_space_cast %alloc_8 {ssbuffer.block_id = 16 : i32, ssbuffer.crossDeps = [5 : i32, 0 : i32], ssbuffer.transfer_id = 5 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %105 = bufferization.to_tensor %memspacecast_35 restrict writable {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 5
        %memspacecast_36 = memref.memory_space_cast %alloc_7 {ssbuffer.block_id = 16 : i32, ssbuffer.crossDeps = [4 : i32, 0 : i32], ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %106 = bufferization.to_tensor %memspacecast_36 restrict writable {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32>
        %107 = arith.muli %49, %c32768_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %108 = arith.index_cast %13 {ssbuffer.block_id = 16 : i32} : i32 to index
        %109 = arith.index_cast %107 {ssbuffer.block_id = 16 : i32} : i32 to index
        %110 = arith.addi %108, %109 {ssbuffer.block_id = 16 : i32} : index
        %111 = arith.addi %17, %110 {ssbuffer.block_id = 16 : i32} : index
        %reinterpret_cast_37 = memref.reinterpret_cast %arg9 to offset: [%111], sizes: [64, 64], strides: [64, 1] {ssbuffer.block_id = 16 : i32} : memref<?xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
        %extracted_slice_38 = tensor.extract_slice %arg21[%29, %31] [%30, %32] [1, 1] {ssbuffer.block_id = 16 : i32} : tensor<64x64xf32> to tensor<?x?xf32>
        %subview_39 = memref.subview %reinterpret_cast_37[0, 0] [%30, %32] [1, 1] {ssbuffer.block_id = 16 : i32} : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice_38 in writable %subview_39 {ssbuffer.block_id = 16 : i32} : (tensor<?x?xf32>, memref<?x?xf32, strided<[64, 1], offset: ?>>) -> ()
        %112 = arith.muli %48, %c64_i32 {ssbuffer.block_id = 16 : i32, ssbuffer.dep_mark = [12 : i32]} : i32
        %113 = arith.minsi %112, %arg13 {ssbuffer.block_id = 16 : i32} : i32
        %114 = arith.subi %113, %c1_i32 {ssbuffer.block_id = 16 : i32} : i32
        %reinterpret_cast_40 = memref.reinterpret_cast %arg11 to offset: [%56], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.dep_mark = [1 : i32, 13 : i32]} : memref<?xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
        %115 = arith.subi %arg20, %c-1_i32 {ssbuffer.block_id = 16 : i32} : i32
        %116 = arith.divui %115, %c1_i32 {ssbuffer.block_id = 16 : i32} : i32
        %c2_i32_41 = arith.constant {ssbuffer.block_id = 16 : i32} 2 : i32
        %117 = arith.remsi %116, %c2_i32_41 {ssbuffer.block_id = 16 : i32} : i32
        %c0_i32_42 = arith.constant {ssbuffer.block_id = 16 : i32} 0 : i32
        %118 = arith.cmpi eq, %117, %c0_i32_42 {ssbuffer.block_id = 16 : i32} : i32
        %119 = scf.if %118 -> (tensor<64x64xf32>) {
          %129 = bufferization.to_tensor %memspacecast restrict writable : memref<64x64xf32>
          scf.yield %129 : tensor<64x64xf32>
        } else {
          %129 = bufferization.to_tensor %memspacecast_11 restrict writable : memref<64x64xf32>
          scf.yield %129 : tensor<64x64xf32>
        } {ssbuffer.block_id = 16 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32], ssbuffer.intra_buffer}
        %extracted_slice_43 = tensor.extract_slice %119[%84, %86] [%85, %87] [1, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.dep_mark = [2 : i32, 3 : i32, 4 : i32, 5 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32]} : tensor<64x64xf32> to tensor<?x?xf32>
        %subview_44 = memref.subview %reinterpret_cast_40[0, 0] [%85, %87] [1, 1] {ssbuffer.block_id = 16 : i32, ssbuffer.dep_mark = [2 : i32, 3 : i32, 16 : i32, 17 : i32]} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice_43 in writable %subview_44 {ssbuffer.block_id = 16 : i32} : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
        %120 = arith.muli %114, %c512_i32 {ssbuffer.block_id = 16 : i32} : i32
        %121 = arith.index_cast %120 {ssbuffer.block_id = 16 : i32} : i32 to index
        %122 = arith.addi %10, %121 {ssbuffer.block_id = 16 : i32} : index
        %reinterpret_cast_45 = memref.reinterpret_cast %arg5 to offset: [%122], sizes: [64], strides: [1] {ssbuffer.block_id = 16 : i32} : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
        %alloc_46 = memref.alloc() {ssbuffer.block_id = 16 : i32} : memref<64xf32>
        memref.copy %reinterpret_cast_45, %alloc_46 {ssbuffer.block_id = 16 : i32} : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32>
        %123 = bufferization.to_tensor %alloc_46 restrict writable {ssbuffer.block_id = 16 : i32} : memref<64xf32>
        %expanded = tensor.expand_shape %123 [[0, 1]] output_shape [1, 64] {ssbuffer.block_id = 16 : i32} : tensor<64xf32> into tensor<1x64xf32>
        %124 = math.exp %expanded {DataUse, ssbuffer.block_id = 16 : i32} : tensor<1x64xf32>
        %collapsed = tensor.collapse_shape %124 [[0, 1]] {ssbuffer.block_id = 16 : i32} : tensor<1x64xf32> into tensor<64xf32>
        %broadcasted = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [0]  {ssbuffer.block_id = 16 : i32}
        %125 = arith.mulf %arg21, %broadcasted {DataUse, ssbuffer.block_id = 16 : i32} : tensor<64x64xf32>
        %126 = arith.mulf %105, %37 {DataUse, ssbuffer.block_id = 16 : i32} : tensor<64x64xf32>
        %127 = arith.subf %126, %106 {DataUse, ssbuffer.block_id = 16 : i32} : tensor<64x64xf32>
        %128 = arith.addf %125, %127 {DataUse, ssbuffer.block_id = 16 : i32} : tensor<64x64xf32>
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
        scf.yield %128 : tensor<64x64xf32>
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
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 9 : i32} : tensor<64x64xf32>
      %1 = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %2 = arith.divsi %arg18, %c8_i32 {ssbuffer.block_id = 9 : i32} : i32
      %3 = arith.remsi %arg18, %c8_i32 {ssbuffer.block_id = 9 : i32} : i32
      %4 = arith.muli %2, %arg13 {ssbuffer.block_id = 9 : i32} : i32
      %5 = arith.addi %arg13, %c63_i32 {ssbuffer.block_id = 9 : i32} : i32
      %6 = arith.divsi %5, %c64_i32 {ssbuffer.block_id = 9 : i32} : i32
      %7 = arith.muli %4, %c8_i32 {ssbuffer.block_id = 9 : i32} : i32
      %8 = arith.addi %7, %3 {ssbuffer.block_id = 9 : i32} : i32
      %9 = arith.muli %8, %c64_i32 {ssbuffer.block_id = 9 : i32} : i32
      %10 = arith.index_cast %9 {ssbuffer.block_id = 9 : i32} : i32 to index
      %11 = arith.muli %arg17, %c64_i32 {ssbuffer.block_id = 9 : i32} : i32
      %12 = arith.maxsi %11, %c0_i32 {ssbuffer.block_id = 9 : i32} : i32
      %13 = arith.index_cast %12 {ssbuffer.block_id = 9 : i32} : i32 to index
      %14 = arith.subi %c0_i32, %11 {ssbuffer.block_id = 9 : i32} : i32
      %15 = arith.maxsi %14, %c0_i32 {ssbuffer.block_id = 9 : i32} : i32
      %16 = arith.index_cast %15 {ssbuffer.block_id = 9 : i32} : i32 to index
      %17 = arith.subi %6, %c1_i32 {ssbuffer.block_id = 9 : i32} : i32
      %alloc = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      %alloc_2 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [2 : i32, 1 : i32], ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 3
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      scf.for %arg20 = %c-1_i32 to %17 step %c1_i32  : i32 {
        %18 = arith.subi %17, %arg20 {ssbuffer.block_id = 4 : i32} : i32
        %19 = arith.addi %18, %c-1_i32 {ssbuffer.block_id = 4 : i32} : i32
        %20 = arith.muli %19, %c64_i32 {ssbuffer.block_id = 4 : i32} : i32
        %21 = arith.maxsi %20, %c0_i32 {ssbuffer.block_id = 4 : i32} : i32
        %22 = arith.index_cast %21 {ssbuffer.block_id = 4 : i32} : i32 to index
        %23 = arith.muli %22, %c512 {ssbuffer.block_id = 4 : i32} : index
        %24 = arith.index_cast %arg13 {ssbuffer.block_id = 4 : i32} : i32 to index
        %alloc_7 = memref.alloc() {ssbuffer.block_id = 4 : i32} : memref<64x64xf32>
        %25 = arith.divsi %23, %c512 {ssbuffer.block_id = 4 : i32} : index
        %26 = arith.subi %24, %25 {ssbuffer.block_id = 4 : i32} : index
        %27 = arith.maxsi %26, %c0 {ssbuffer.block_id = 4 : i32} : index
        %28 = arith.minsi %27, %c64 {ssbuffer.block_id = 4 : i32} : index
        %29 = arith.remsi %23, %c512 {ssbuffer.block_id = 4 : i32} : index
        %30 = arith.subi %c64, %29 {ssbuffer.block_id = 4 : i32} : index
        %31 = arith.maxsi %30, %c0 {ssbuffer.block_id = 4 : i32} : index
        %32 = arith.minsi %31, %c64 {ssbuffer.block_id = 4 : i32} : index
        %33 = arith.subi %c0_i32, %20 {ssbuffer.block_id = 4 : i32} : i32
        %34 = arith.maxsi %33, %c0_i32 {ssbuffer.block_id = 4 : i32} : i32
        %35 = arith.index_cast %34 {ssbuffer.block_id = 4 : i32} : i32 to index
        %36 = arith.minsi %35, %28 {ssbuffer.block_id = 4 : i32} : index
        %37 = arith.subi %28, %36 {ssbuffer.block_id = 4 : i32} : index
        %38 = arith.minsi %32, %c0 {ssbuffer.block_id = 4 : i32} : index
        %39 = arith.subi %32, %38 {ssbuffer.block_id = 4 : i32} : index
        %40 = arith.cmpi slt, %37, %c64 {ssbuffer.block_id = 4 : i32} : index
        %41 = arith.cmpi slt, %39, %c64 {ssbuffer.block_id = 4 : i32} : index
        %42 = arith.ori %40, %41 {ssbuffer.block_id = 4 : i32} : i1
        scf.if %42 {
          linalg.fill {ssbuffer.block_id = 4 : i32} ins(%cst : f32) outs(%alloc_7 : memref<64x64xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
        %43 = arith.addi %23, %10 {ssbuffer.block_id = 4 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 4 : i32} : memref<?xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
        %subview = memref.subview %reinterpret_cast[0, 0] [%37, %39] [1, 1] {ssbuffer.block_id = 4 : i32} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %subview_8 = memref.subview %alloc_7[%36, %38] [%37, %39] [1, 1] {ssbuffer.block_id = 4 : i32} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview, %subview_8 {ssbuffer.block_id = 4 : i32} : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        %44 = bufferization.to_tensor %alloc_7 restrict writable {ssbuffer.block_id = 4 : i32} : memref<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        %45 = hivm.hir.convert_layout %alloc output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 4 : i32, ssbuffer.crossDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 0 : i32} : (memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<64x64xf32, #hivm.address_space<cbuf>>
        %memspacecast = memref.memory_space_cast %45 {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32} : memref<64x64xf32, #hivm.address_space<cbuf>> to memref<64x64xf32>
        %46 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32} : memref<64x64xf32>
        %transposed = linalg.transpose ins(%46 : tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) permutation = [1, 0]  {ssbuffer.block_id = 4 : i32}
        %47 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<64x64xf32>
        %48 = linalg.fill {ssbuffer.block_id = 4 : i32} ins(%cst : f32) outs(%47 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %49 = linalg.matmul {input_precision = "ieee", ssbuffer.bdep, ssbuffer.block_id = 4 : i32} ins(%44, %transposed : tensor<64x64xf32>, tensor<64x64xf32>) outs(%48 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32} ins(%49 : tensor<64x64xf32>) outs(%alloc_4 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 4
        %alloc_9 = memref.alloc() {ssbuffer.block_id = 6 : i32} : memref<64x64xf32>
        scf.if %42 {
          linalg.fill {ssbuffer.block_id = 6 : i32} ins(%cst : f32) outs(%alloc_9 : memref<64x64xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 6 : i32}
        %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 6 : i32} : memref<?xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
        %subview_11 = memref.subview %reinterpret_cast_10[0, 0] [%37, %39] [1, 1] {ssbuffer.block_id = 6 : i32} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %subview_12 = memref.subview %alloc_9[%36, %38] [%37, %39] [1, 1] {ssbuffer.block_id = 6 : i32} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_11, %subview_12 {ssbuffer.block_id = 6 : i32} : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        %50 = bufferization.to_tensor %alloc_9 restrict writable {ssbuffer.block_id = 6 : i32} : memref<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %51 = hivm.hir.convert_layout %alloc_2 output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 6 : i32, ssbuffer.crossDeps = [1 : i32, 0 : i32], ssbuffer.transfer_id = 1 : i32} : (memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<64x64xf32, #hivm.address_space<cbuf>>
        %memspacecast_13 = memref.memory_space_cast %51 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32, #hivm.address_space<cbuf>> to memref<64x64xf32>
        %52 = bufferization.to_tensor %memspacecast_13 restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32>
        %transposed_14 = linalg.transpose ins(%52 : tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) permutation = [1, 0]  {ssbuffer.block_id = 6 : i32}
        %53 = linalg.matmul {input_precision = "ieee", ssbuffer.adep, ssbuffer.block_id = 6 : i32} ins(%transposed_14, %50 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 4 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 4 : i32} ins(%53 : tensor<64x64xf32>) outs(%alloc_5 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 4 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 5
        %54 = arith.addi %43, %13 {ssbuffer.block_id = 8 : i32} : index
        %55 = arith.subi %54, %10 {ssbuffer.block_id = 8 : i32} : index
        %56 = arith.divsi %55, %c512 {ssbuffer.block_id = 8 : i32} : index
        %57 = arith.subi %24, %56 {ssbuffer.block_id = 8 : i32} : index
        %58 = arith.maxsi %57, %c0 {ssbuffer.block_id = 8 : i32} : index
        %59 = arith.minsi %58, %c64 {ssbuffer.block_id = 8 : i32} : index
        %60 = arith.remsi %55, %c512 {ssbuffer.block_id = 8 : i32} : index
        %61 = arith.subi %c64, %60 {ssbuffer.block_id = 8 : i32} : index
        %62 = arith.maxsi %61, %c0 {ssbuffer.block_id = 8 : i32} : index
        %63 = arith.minsi %62, %c64 {ssbuffer.block_id = 8 : i32} : index
        %64 = arith.minsi %35, %59 {ssbuffer.block_id = 8 : i32} : index
        %65 = arith.subi %59, %64 {ssbuffer.block_id = 8 : i32} : index
        %66 = arith.minsi %16, %63 {ssbuffer.block_id = 8 : i32} : index
        %67 = arith.subi %63, %66 {ssbuffer.block_id = 8 : i32} : index
        %68 = arith.cmpi slt, %65, %c64 {ssbuffer.block_id = 8 : i32} : index
        %69 = arith.cmpi slt, %67, %c64 {ssbuffer.block_id = 8 : i32} : index
        %70 = arith.ori %68, %69 {ssbuffer.block_id = 8 : i32} : i1
        %alloc_15 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<64x64xf32>
        scf.if %70 {
          linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%alloc_15 : memref<64x64xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
        %reinterpret_cast_16 = memref.reinterpret_cast %arg2 to offset: [%54], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 5 : i32} : memref<?xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
        %subview_17 = memref.subview %reinterpret_cast_16[0, 0] [%65, %67] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %subview_18 = memref.subview %alloc_15[%64, %66] [%65, %67] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_17, %subview_18 {ssbuffer.block_id = 5 : i32} : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
        %71 = bufferization.to_tensor %alloc_15 restrict writable {ssbuffer.block_id = 5 : i32} : memref<64x64xf32>
        %transposed_19 = linalg.transpose ins(%71 : tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) permutation = [1, 0]  {ssbuffer.block_id = 5 : i32}
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
        %72 = hivm.hir.convert_layout %alloc_3 output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 5 : i32, ssbuffer.crossDeps = [2 : i32, 0 : i32], ssbuffer.transfer_id = 2 : i32} : (memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<64x64xf32, #hivm.address_space<cbuf>>
        %memspacecast_20 = memref.memory_space_cast %72 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32, #hivm.address_space<cbuf>> to memref<64x64xf32>
        %73 = bufferization.to_tensor %memspacecast_20 restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32>
        %74 = linalg.matmul {input_precision = "ieee", ssbuffer.bdep, ssbuffer.block_id = 5 : i32} ins(%transposed_19, %73 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 3
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 5 : i32} ins(%74 : tensor<64x64xf32>) outs(%alloc_6 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
      } {DataUse, ssbuffer.block_id = 20 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 4 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
