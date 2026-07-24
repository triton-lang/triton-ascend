// RUN: triton-opt --add-control-flow-condition %s | FileCheck %s

// Unit Tests for AddControlFlowCondition Pass - Cross Core Dependencies Test
//
// This test verifies:
// the same producer will be used in different consumers

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chunk_mesa_net_fwd_kernel_h(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c16384_i64 = arith.constant {ssbuffer.block_id = 15 : i32} 16384 : i64
    %c0_i32 = arith.constant {Undefined, ssbuffer.block_id = 15 : i32} 0 : i32
    %c16384_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 16384 : i32
    %c63_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 63 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 128 : i32
    %c1_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 1 : i32
    %c64_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 64 : i32
    %c8_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 8 : i32
    %c0 = arith.constant {DataUse, ssbuffer.block_id = 15 : i32} 0 : index
    %c128 = arith.constant {ssbuffer.block_id = 15 : i32} 128 : index
    %c64 = arith.constant {ssbuffer.block_id = 15 : i32} 64 : index
    %cst = arith.constant {ssbuffer.block_id = 15 : i32} 0.000000e+00 : f32
    %c1024 = arith.constant {ssbuffer.block_id = 15 : i32} 1024 : index
    %c8 = arith.constant {ssbuffer.block_id = 15 : i32} 8 : index
    %cst_0 = arith.constant {ssbuffer.block_id = 15 : i32} 0.000000e+00 : f16
    %cst_1 = arith.constant {ssbuffer.block_id = 13 : i32} dense<[4, 4, 16, 16]> : tensor<4xi64>
    %cst_2 = arith.constant {ssbuffer.block_id = 13 : i32} dense<[64, 4, 16]> : tensor<3xi64>

    // CHECK:       %[[PTR0:.*]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr<11>
    // CHECK:       %[[PTR1:.*]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr<11>
    // CHECK:       %[[PTR2:.*]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr<11>
    // CHECK:       %[[PTR3:.*]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr<11>
    // CHECK：      scope.scope : () -> () {
    scope.scope : () -> () {
      %0 = arith.divsi %arg18, %c8_i32 {ssbuffer.block_id = 15 : i32} : i32
      %1 = arith.remsi %arg18, %c8_i32 {ssbuffer.block_id = 15 : i32} : i32
      %2 = arith.muli %0, %arg12 {ssbuffer.block_id = 15 : i32} : i32
      %3 = arith.addi %arg12, %c63_i32 {ssbuffer.block_id = 15 : i32} : i32
      %4 = arith.divsi %3, %c64_i32 {ssbuffer.block_id = 15 : i32} : i32
      %5 = arith.muli %0, %4 {ssbuffer.block_id = 15 : i32} : i32
      %6 = arith.muli %arg18, %c16384_i32 {ssbuffer.block_id = 15 : i32} : i32
      %7 = arith.index_cast %6 {ssbuffer.block_id = 15 : i32} : i32 to index
      %8 = arith.muli %arg16, %c64_i32 {ssbuffer.block_id = 15 : i32} : i32
      %9 = arith.muli %arg17, %c64_i32 {ssbuffer.block_id = 15 : i32} : i32
      %10 = arith.maxsi %8, %c0_i32 {ssbuffer.block_id = 15 : i32} : i32
      %11 = arith.index_cast %10 {ssbuffer.block_id = 15 : i32} : i32 to index
      %12 = arith.maxsi %9, %c0_i32 {ssbuffer.block_id = 15 : i32} : i32
      %13 = arith.index_cast %12 {ssbuffer.block_id = 15 : i32} : i32 to index
      %14 = arith.muli %11, %c128 {ssbuffer.block_id = 15 : i32} : index
      %15 = arith.addi %14, %7 {ssbuffer.block_id = 15 : i32} : index
      %16 = arith.addi %15, %13 {ssbuffer.block_id = 15 : i32} : index
      %17 = arith.subi %16, %7 {ssbuffer.block_id = 15 : i32} : index
      %18 = arith.divsi %17, %c128 {ssbuffer.block_id = 15 : i32} : index
      %19 = arith.subi %c128, %18 {ssbuffer.block_id = 15 : i32} : index
      %20 = arith.maxsi %19, %c0 {ssbuffer.block_id = 15 : i32} : index
      %21 = arith.minsi %20, %c64 {ssbuffer.block_id = 15 : i32} : index
      %22 = arith.remsi %17, %c128 {ssbuffer.block_id = 15 : i32} : index
      %23 = arith.subi %c128, %22 {ssbuffer.block_id = 15 : i32} : index
      %24 = arith.maxsi %23, %c0 {ssbuffer.block_id = 15 : i32} : index
      %25 = arith.minsi %24, %c64 {ssbuffer.block_id = 15 : i32} : index
      %26 = arith.subi %c0_i32, %8 {ssbuffer.block_id = 15 : i32} : i32
      %27 = arith.maxsi %26, %c0_i32 {ssbuffer.block_id = 15 : i32} : i32
      %28 = arith.index_cast %27 {ssbuffer.block_id = 15 : i32} : i32 to index
      %29 = arith.minsi %28, %21 {ssbuffer.block_id = 15 : i32} : index
      %30 = arith.subi %21, %29 {ssbuffer.block_id = 15 : i32} : index
      %31 = arith.subi %c0_i32, %9 {ssbuffer.block_id = 15 : i32} : i32
      %32 = arith.maxsi %31, %c0_i32 {ssbuffer.block_id = 15 : i32} : i32
      %33 = arith.index_cast %32 {ssbuffer.block_id = 15 : i32} : i32 to index
      %34 = arith.minsi %33, %25 {ssbuffer.block_id = 15 : i32} : index
      %35 = arith.subi %25, %34 {ssbuffer.block_id = 15 : i32} : index
      %36 = arith.cmpi slt, %30, %c64 {ssbuffer.block_id = 15 : i32} : index
      %37 = arith.cmpi slt, %35, %c64 {ssbuffer.block_id = 15 : i32} : index
      %38 = arith.ori %36, %37 {ssbuffer.block_id = 15 : i32} : i1
      %39 = arith.muli %2, %c8_i32 {ssbuffer.block_id = 15 : i32} : i32
      %40 = arith.addi %39, %1 {ssbuffer.block_id = 15 : i32} : i32
      %41 = arith.muli %40, %c128_i32 {ssbuffer.block_id = 15 : i32} : i32
      %42 = arith.index_cast %41 {ssbuffer.block_id = 15 : i32} : i32 to index
      %43 = arith.index_cast %40 {ssbuffer.block_id = 15 : i32} : i32 to index
      %alloc = memref.alloc() {ssbuffer.block_id = 16 : i32} : memref<64x64xf32>
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 16 : i32} : memref<64x64xf32>
      scf.if %38 {
        linalg.fill {ssbuffer.block_id = 16 : i32} ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
        linalg.fill {ssbuffer.block_id = 16 : i32} ins(%cst : f32) outs(%alloc_3 : memref<64x64xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 16 : i32}
      %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%16], sizes: [64, 64], strides: [128, 1] {ssbuffer.block_id = 16 : i32} : memref<?xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
      %subview = memref.subview %reinterpret_cast[0, 0] [%30, %35] [1, 1] {ssbuffer.block_id = 16 : i32} : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
      %subview_4 = memref.subview %alloc[%29, %34] [%30, %35] [1, 1] {ssbuffer.block_id = 16 : i32} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %subview, %subview_4 {ssbuffer.block_id = 16 : i32} : memref<?x?xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %44 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 16 : i32} : memref<64x64xf32>
      %reinterpret_cast_5 = memref.reinterpret_cast %arg9 to offset: [%16], sizes: [64, 64], strides: [128, 1] {ssbuffer.block_id = 16 : i32} : memref<?xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
      %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%30, %35] [1, 1] {ssbuffer.block_id = 16 : i32} : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
      %subview_7 = memref.subview %alloc_3[%29, %34] [%30, %35] [1, 1] {ssbuffer.block_id = 16 : i32} : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %subview_6, %subview_7 {ssbuffer.block_id = 16 : i32} : memref<?x?xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %45 = bufferization.to_tensor %alloc_3 restrict writable {ssbuffer.block_id = 16 : i32} : memref<64x64xf32>
      %alloc_8 = memref.alloc() {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_9 = memref.alloc() {ssbuffer.block_id = 18 : i32, ssbuffer.crossDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
      %alloc_10 = memref.alloc() {ssbuffer.block_id = 18 : i32, ssbuffer.crossDeps = [2 : i32, 1 : i32], ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_10 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      %46:2 = scf.for %arg19 = %c0_i32 to %4 step %c1_i32 iter_args(%arg20 = %44, %arg21 = %45) -> (tensor<64x64xf32>, tensor<64x64xf32>)  : i32 {
        // CHECK:       scf.if
        %47 = arith.muli %arg19, %c64_i32 {ssbuffer.block_id = 12 : i32} : i32
        %48 = arith.maxsi %47, %c0_i32 {ssbuffer.block_id = 12 : i32} : i32
        %49 = arith.index_cast %48 {ssbuffer.block_id = 12 : i32} : i32 to index
        %50 = arith.muli %49, %c1024 {ssbuffer.block_id = 12 : i32} : index
        %51 = arith.addi %50, %42 {ssbuffer.block_id = 12 : i32} : index
        %52 = arith.index_cast %arg12 {ssbuffer.block_id = 12 : i32} : i32 to index
        %53 = arith.addi %51, %11 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [1 : i32]} : index
        %54 = arith.muli %49, %c8 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [2 : i32]} : index
        %55 = arith.divsi %54, %c8 {ssbuffer.block_id = 12 : i32} : index
        %56 = arith.subi %52, %55 {ssbuffer.block_id = 12 : i32} : index
        %57 = arith.maxsi %56, %c0 {ssbuffer.block_id = 12 : i32} : index
        %58 = arith.minsi %57, %c64 {ssbuffer.block_id = 12 : i32} : index
        %59 = arith.subi %c0_i32, %47 {ssbuffer.block_id = 12 : i32} : i32
        %60 = arith.maxsi %59, %c0_i32 {ssbuffer.block_id = 12 : i32} : i32
        %61 = arith.index_cast %60 {ssbuffer.block_id = 12 : i32} : i32 to index
        %62 = arith.minsi %61, %58 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [4 : i32]} : index
        %63 = arith.subi %58, %62 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [3 : i32]} : index
        %64 = arith.cmpi slt, %63, %c64 {ssbuffer.block_id = 12 : i32} : index
        %65 = arith.subi %53, %42 {ssbuffer.block_id = 12 : i32} : index
        %66 = arith.divsi %65, %c1024 {ssbuffer.block_id = 12 : i32} : index
        %67 = arith.subi %52, %66 {ssbuffer.block_id = 12 : i32} : index
        %68 = arith.maxsi %67, %c0 {ssbuffer.block_id = 12 : i32} : index
        %69 = arith.minsi %68, %c64 {ssbuffer.block_id = 12 : i32} : index
        %70 = arith.remsi %65, %c1024 {ssbuffer.block_id = 12 : i32} : index
        %71 = arith.subi %c128, %70 {ssbuffer.block_id = 12 : i32} : index
        %72 = arith.maxsi %71, %c0 {ssbuffer.block_id = 12 : i32} : index
        %73 = arith.minsi %72, %c64 {ssbuffer.block_id = 12 : i32} : index
        %74 = arith.minsi %61, %69 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [7 : i32]} : index
        %75 = arith.subi %69, %74 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [5 : i32]} : index
        %76 = arith.minsi %28, %73 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [8 : i32]} : index
        %77 = arith.subi %73, %76 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [6 : i32]} : index
        %78 = arith.cmpi slt, %75, %c64 {ssbuffer.block_id = 12 : i32} : index
        %79 = arith.cmpi slt, %77, %c64 {ssbuffer.block_id = 12 : i32} : index
        %80 = arith.ori %78, %79 {ssbuffer.block_id = 12 : i32} : i1
        %81 = arith.index_cast %47 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [9 : i32]} : i32 to index
        %82 = arith.addi %81, %c64 {ssbuffer.block_id = 12 : i32} : index
        %83 = arith.index_cast %arg12 {ssbuffer.block_id = 12 : i32} : i32 to index
        %84 = arith.maxsi %81, %83 {ssbuffer.block_id = 12 : i32} : index
        %85 = arith.minsi %82, %84 {ssbuffer.block_id = 12 : i32} : index
        %86 = arith.subi %85, %81 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [10 : i32]} : index
        %87 = arith.cmpi slt, %86, %c64 {ssbuffer.block_id = 12 : i32} : index
        // there is only one producer, but it contains 2 cross dependency
        // CHECK:       llvm.load volatile
        // CHECK:       arith.cmpi slt
        // CHECK:       llvm.load volatile
        // CHECK:       arith.cmpi slt
        // CHECK:       arith.cmpi slt
        // CHECK:       scf.if
        %alloc_16 = memref.alloc() {ssbuffer.block_id = 13 : i32} : memref<64xf16>
        %alloc_17 = memref.alloc() {ssbuffer.block_id = 13 : i32} : memref<64x64xf16>
        %alloc_18 = memref.alloc() {ssbuffer.block_id = 13 : i32} : memref<64xf32>
        scf.if %87 {
          linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%alloc_18 : memref<64xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 13 : i32}
        scf.if %80 {
          linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst_0 : f16) outs(%alloc_17 : memref<64x64xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 13 : i32}
        scf.if %64 {
          linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst_0 : f16) outs(%alloc_16 : memref<64xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 13 : i32}
        %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%53], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [1 : i32]} : memref<?xf16> to memref<64x64xf16, strided<[1024, 1], offset: ?>>
        %88 = arith.addi %54, %43 {ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [2 : i32]} : index
        %reinterpret_cast_20 = memref.reinterpret_cast %arg4 to offset: [%88], sizes: [64], strides: [8] {ssbuffer.block_id = 13 : i32} : memref<?xf16> to memref<64xf16, strided<[8], offset: ?>>
        %subview_21 = memref.subview %reinterpret_cast_20[0] [%63] [1] {ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [3 : i32]} : memref<64xf16, strided<[8], offset: ?>> to memref<?xf16, strided<[8], offset: ?>>
        %subview_22 = memref.subview %alloc_16[%62] [%63] [1] {ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [3 : i32, 4 : i32]} : memref<64xf16> to memref<?xf16, strided<[1], offset: ?>>
        memref.copy %subview_21, %subview_22 {ssbuffer.block_id = 13 : i32} : memref<?xf16, strided<[8], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %89 = bufferization.to_tensor %alloc_16 restrict writable {ssbuffer.block_id = 13 : i32} : memref<64xf16>
        %subview_23 = memref.subview %reinterpret_cast_19[0, 0] [%75, %77] [1, 1] {ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [5 : i32, 6 : i32]} : memref<64x64xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %subview_24 = memref.subview %alloc_17[%74, %76] [%75, %77] [1, 1] {ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [5 : i32, 6 : i32, 7 : i32, 8 : i32]} : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
        memref.copy %subview_23, %subview_24 {ssbuffer.block_id = 13 : i32} : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1], offset: ?>>
        %90 = bufferization.to_tensor %alloc_17 restrict writable {ssbuffer.block_id = 13 : i32} : memref<64x64xf16>
        %91 = arith.addi %arg19, %c1_i32 {ssbuffer.block_id = 13 : i32} : i32
        %92 = arith.muli %91, %c64_i32 {ssbuffer.block_id = 13 : i32} : i32
        %93 = arith.minsi %92, %arg12 {ssbuffer.block_id = 13 : i32} : i32
        %94 = arith.subi %93, %c1_i32 {ssbuffer.block_id = 13 : i32} : i32
        %95 = arith.muli %94, %c8_i32 {ssbuffer.block_id = 13 : i32} : i32
        %96 = arith.index_cast %39 {ssbuffer.block_id = 13 : i32} : i32 to index
        %97 = arith.index_cast %95 {ssbuffer.block_id = 13 : i32} : i32 to index
        %98 = arith.addi %96, %97 {ssbuffer.block_id = 13 : i32} : index
        %99 = arith.index_cast %1 {ssbuffer.block_id = 13 : i32} : i32 to index
        %100 = arith.addi %98, %99 {ssbuffer.block_id = 13 : i32} : index
        %reinterpret_cast_25 = memref.reinterpret_cast %arg5 to offset: [%100], sizes: [1], strides: [1] {ssbuffer.block_id = 13 : i32} : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
        %101 = memref.load %reinterpret_cast_25[%c0] {ssbuffer.block_id = 13 : i32} : memref<1xf32, strided<[1], offset: ?>>
        %102 = arith.muli %81, %c8 {ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [9 : i32]} : index
        %103 = arith.addi %96, %102 {ssbuffer.block_id = 13 : i32} : index
        %104 = arith.addi %103, %99 {ssbuffer.block_id = 13 : i32} : index
        %reinterpret_cast_26 = memref.reinterpret_cast %arg5 to offset: [%104], sizes: [64], strides: [8] {ssbuffer.block_id = 13 : i32} : memref<?xf32> to memref<64xf32, strided<[8], offset: ?>>
        %105 = tensor.empty() {ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [12 : i32]} : tensor<64x64xf32>
        %subview_27 = memref.subview %reinterpret_cast_26[0] [%86] [1] {ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [10 : i32]} : memref<64xf32, strided<[8], offset: ?>> to memref<?xf32, strided<[8], offset: ?>>
        %subview_28 = memref.subview %alloc_18[0] [%86] [1] {ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [10 : i32]} : memref<64xf32> to memref<?xf32, strided<[1]>>
        memref.copy %subview_27, %subview_28 {ssbuffer.block_id = 13 : i32} : memref<?xf32, strided<[8], offset: ?>> to memref<?xf32, strided<[1]>>
        %106 = bufferization.to_tensor %alloc_18 restrict writable {ssbuffer.block_id = 13 : i32} : memref<64xf32>
        %107 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<64xf32>
        %108 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%101 : f32) outs(%107 : tensor<64xf32>) -> tensor<64xf32>
        %109 = arith.subf %108, %106 {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64xf32>
        %110 = math.exp %109 {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64xf32>
        %111 = arith.extf %90 {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64x64xf16> to tensor<64x64xf32>
        %broadcasted = linalg.broadcast ins(%110 : tensor<64xf32>) outs(%105 : tensor<64x64xf32>) dimensions = [1]  {ssbuffer.block_id = 13 : i32}
        %112 = arith.mulf %111, %broadcasted {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64x64xf32>
        %expanded = tensor.expand_shape %89 [[0, 1]] output_shape [64, 1] {ssbuffer.block_id = 13 : i32} : tensor<64xf16> into tensor<64x1xf16>
        %113 = arith.extf %expanded {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64x1xf16> to tensor<64x1xf32>
        %collapsed = tensor.collapse_shape %113 [[0, 1]] {ssbuffer.block_id = 13 : i32} : tensor<64x1xf32> into tensor<64xf32>
        %broadcasted_29 = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%105 : tensor<64x64xf32>) dimensions = [1]  {ssbuffer.block_id = 13 : i32}
        %114 = arith.mulf %112, %broadcasted_29 {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64x64xf32>
        %115 = arith.truncf %114 {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64x64xf32> to tensor<64x64xf16>
        %reshape = tensor.reshape %115(%cst_2) {ssbuffer.block_id = 13 : i32} : (tensor<64x64xf16>, tensor<3xi64>) -> tensor<64x4x16xf16>
        %116 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<4x64x16xf16>
        %transposed = linalg.transpose ins(%reshape : tensor<64x4x16xf16>) outs(%116 : tensor<4x64x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 13 : i32}
        %reshape_30 = tensor.reshape %transposed(%cst_1) {ssbuffer.block_id = 13 : i32} : (tensor<4x64x16xf16>, tensor<4xi64>) -> tensor<4x4x16x16xf16>
        %117 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<1xf32>
        %inserted = tensor.insert %101 into %117[%c0] {ssbuffer.block_id = 13 : i32} : tensor<1xf32>
        %118 = math.exp %inserted {DataUse, ssbuffer.block_id = 13 : i32} : tensor<1xf32>
        %extracted = tensor.extract %118[%c0] {DataUse, ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [11 : i32]} : tensor<1xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.copy ins(%reshape_30 : tensor<4x4x16x16xf16>) outs(%alloc_8 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        // CHECK:       scf.if
        hivm.hir.sync_block_wait {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
        %memspacecast = memref.memory_space_cast %alloc_10 {ssbuffer.block_id = 14 : i32, ssbuffer.crossDeps = [2 : i32, 0 : i32], ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %119 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
        %memspacecast_31 = memref.memory_space_cast %alloc_9 {ssbuffer.block_id = 14 : i32, ssbuffer.crossDeps = [1 : i32, 0 : i32], ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %120 = bufferization.to_tensor %memspacecast_31 restrict writable {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32>
        %121 = arith.addi %5, %arg19 {ssbuffer.block_id = 14 : i32} : i32
        %122 = arith.muli %121, %c8_i32 {ssbuffer.block_id = 14 : i32} : i32
        %123 = arith.addi %122, %1 {ssbuffer.block_id = 14 : i32} : i32
        %124 = arith.extsi %123 {ssbuffer.block_id = 14 : i32} : i32 to i64
        %125 = arith.muli %124, %c16384_i64 {ssbuffer.block_id = 14 : i32} : i64
        %126 = arith.index_cast %125 {ssbuffer.block_id = 14 : i32} : i64 to index
        %127 = arith.addi %14, %126 {ssbuffer.block_id = 14 : i32} : index
        %128 = arith.addi %127, %13 {ssbuffer.block_id = 14 : i32} : index
        %reinterpret_cast_32 = memref.reinterpret_cast %arg6 to offset: [%128], sizes: [64, 64], strides: [128, 1] {ssbuffer.block_id = 14 : i32} : memref<?xf16> to memref<64x64xf16, strided<[128, 1], offset: ?>>
        %reinterpret_cast_33 = memref.reinterpret_cast %arg7 to offset: [%128], sizes: [64, 64], strides: [128, 1] {ssbuffer.block_id = 14 : i32} : memref<?xf16> to memref<64x64xf16, strided<[128, 1], offset: ?>>
        %129 = arith.truncf %arg20 {DataUse, ssbuffer.block_id = 14 : i32} : tensor<64x64xf32> to tensor<64x64xf16>
        %130 = arith.subi %128, %126 {ssbuffer.block_id = 14 : i32} : index
        %131 = arith.divsi %130, %c128 {ssbuffer.block_id = 14 : i32} : index
        %132 = arith.subi %c128, %131 {ssbuffer.block_id = 14 : i32} : index
        %133 = arith.maxsi %132, %c0 {ssbuffer.block_id = 14 : i32} : index
        %134 = arith.minsi %133, %c64 {ssbuffer.block_id = 14 : i32} : index
        %135 = arith.remsi %130, %c128 {ssbuffer.block_id = 14 : i32} : index
        %136 = arith.subi %c128, %135 {ssbuffer.block_id = 14 : i32} : index
        %137 = arith.maxsi %136, %c0 {ssbuffer.block_id = 14 : i32} : index
        %138 = arith.minsi %137, %c64 {ssbuffer.block_id = 14 : i32} : index
        %139 = arith.minsi %28, %134 {ssbuffer.block_id = 14 : i32} : index
        %140 = arith.subi %134, %139 {ssbuffer.block_id = 14 : i32} : index
        %141 = arith.minsi %33, %138 {ssbuffer.block_id = 14 : i32} : index
        %142 = arith.subi %138, %141 {ssbuffer.block_id = 14 : i32} : index
        %extracted_slice_34 = tensor.extract_slice %129[%139, %141] [%140, %142] [1, 1] {ssbuffer.block_id = 14 : i32} : tensor<64x64xf16> to tensor<?x?xf16>
        %subview_35 = memref.subview %reinterpret_cast_32[0, 0] [%140, %142] [1, 1] {ssbuffer.block_id = 14 : i32} : memref<64x64xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice_34 in writable %subview_35 {ssbuffer.block_id = 14 : i32} : (tensor<?x?xf16>, memref<?x?xf16, strided<[128, 1], offset: ?>>) -> ()
        %143 = arith.truncf %arg21 {DataUse, ssbuffer.block_id = 14 : i32} : tensor<64x64xf32> to tensor<64x64xf16>
        %extracted_slice_36 = tensor.extract_slice %143[%139, %141] [%140, %142] [1, 1] {ssbuffer.block_id = 14 : i32} : tensor<64x64xf16> to tensor<?x?xf16>
        %subview_37 = memref.subview %reinterpret_cast_33[0, 0] [%140, %142] [1, 1] {ssbuffer.block_id = 14 : i32} : memref<64x64xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice_36 in writable %subview_37 {ssbuffer.block_id = 14 : i32} : (tensor<?x?xf16>, memref<?x?xf16, strided<[128, 1], offset: ?>>) -> ()
        %144 = linalg.fill {ssbuffer.block_id = 14 : i32, ssbuffer.dep_mark = [11 : i32, 12 : i32]} ins(%extracted : f32) outs(%105 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %145 = arith.mulf %arg20, %144 {DataUse, ssbuffer.block_id = 14 : i32} : tensor<64x64xf32>
        %146 = arith.mulf %arg21, %144 {DataUse, ssbuffer.block_id = 14 : i32} : tensor<64x64xf32>
        %147 = arith.addf %119, %145 {ssbuffer.add_from_matmul, ssbuffer.block_id = 14 : i32} : tensor<64x64xf32>
        %148 = arith.addf %120, %146 {ssbuffer.add_from_matmul, ssbuffer.block_id = 14 : i32} : tensor<64x64xf32>
        hivm.hir.sync_block_set {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.sync_block_set {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        scf.yield %147, %148 : tensor<64x64xf32>, tensor<64x64xf32>
      } {DataUse, ssbuffer.block_id = 18 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      %reinterpret_cast_11 = memref.reinterpret_cast %arg10 to offset: [%16], sizes: [64, 64], strides: [128, 1] {ssbuffer.block_id = 17 : i32} : memref<?xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
      %extracted_slice = tensor.extract_slice %46#0[%29, %34] [%30, %35] [1, 1] {ssbuffer.block_id = 17 : i32} : tensor<64x64xf32> to tensor<?x?xf32>
      %subview_12 = memref.subview %reinterpret_cast_11[0, 0] [%30, %35] [1, 1] {ssbuffer.block_id = 17 : i32} : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_12 {ssbuffer.block_id = 17 : i32} : (tensor<?x?xf32>, memref<?x?xf32, strided<[128, 1], offset: ?>>) -> ()
      %reinterpret_cast_13 = memref.reinterpret_cast %arg11 to offset: [%16], sizes: [64, 64], strides: [128, 1] {ssbuffer.block_id = 17 : i32} : memref<?xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
      %extracted_slice_14 = tensor.extract_slice %46#1[%29, %34] [%30, %35] [1, 1] {ssbuffer.block_id = 17 : i32} : tensor<64x64xf32> to tensor<?x?xf32>
      %subview_15 = memref.subview %reinterpret_cast_13[0, 0] [%30, %35] [1, 1] {ssbuffer.block_id = 17 : i32} : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_14 in writable %subview_15 {ssbuffer.block_id = 17 : i32} : (tensor<?x?xf32>, memref<?x?xf32, strided<[128, 1], offset: ?>>) -> ()
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %0 = arith.addi %arg12, %c63_i32 {ssbuffer.block_id = 15 : i32} : i32
      %1 = arith.divsi %0, %c64_i32 {ssbuffer.block_id = 15 : i32} : i32
      %2 = arith.divsi %arg18, %c8_i32 {ssbuffer.block_id = 6 : i32} : i32
      %3 = arith.remsi %arg18, %c8_i32 {ssbuffer.block_id = 6 : i32} : i32
      %4 = arith.muli %2, %arg12 {ssbuffer.block_id = 6 : i32} : i32
      %5 = arith.muli %arg17, %c64_i32 {ssbuffer.block_id = 6 : i32} : i32
      %6 = arith.maxsi %5, %c0_i32 {ssbuffer.block_id = 6 : i32} : i32
      %7 = arith.index_cast %6 {ssbuffer.block_id = 6 : i32} : i32 to index
      %8 = arith.subi %c0_i32, %5 {ssbuffer.block_id = 6 : i32} : i32
      %9 = arith.maxsi %8, %c0_i32 {ssbuffer.block_id = 6 : i32} : i32
      %10 = arith.index_cast %9 {ssbuffer.block_id = 6 : i32} : i32 to index
      %11 = arith.muli %4, %c8_i32 {ssbuffer.block_id = 6 : i32} : i32
      %12 = arith.addi %11, %3 {ssbuffer.block_id = 6 : i32} : i32
      %13 = arith.muli %12, %c128_i32 {ssbuffer.block_id = 6 : i32} : i32
      %14 = arith.index_cast %13 {ssbuffer.block_id = 6 : i32} : i32 to index
      %alloc = memref.alloc() {ssbuffer.block_id = 18 : i32, ssbuffer.crossDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      scf.for %arg19 = %c0_i32 to %1 step %c1_i32  : i32 {
        // verify different consumers user different ptrs
        // CHECK:       llvm.load volatile
        // CHECK:       llvm.load volatile
        // CHECK:       arith.cmpi sgt
        // CHECK:       arith.cmpi sgt
        // CHECK:       llvm.load volatile
        // CHECK:       llvm.load volatile
        // CHECK:       arith.cmpi slt
        // CHECK:       arith.cmpi slt
        // CHECK:       arith.cmpi slt
        // CHECK:       scf.if
        hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        %15 = hivm.hir.convert_layout %alloc output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 3 : i32, ssbuffer.crossDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 0 : i32} : (memref<4x4x16x16xf16, #hivm.address_space<cbuf>>) -> memref<64x64xf16, #hivm.address_space<cbuf>>
        %memspacecast = memref.memory_space_cast %15 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32} : memref<64x64xf16, #hivm.address_space<cbuf>> to memref<64x64xf16>
        %16 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32} : memref<64x64xf16>
        %17 = arith.muli %arg19, %c64_i32 {ssbuffer.block_id = 3 : i32} : i32
        %18 = arith.maxsi %17, %c0_i32 {ssbuffer.block_id = 3 : i32} : i32
        %19 = arith.index_cast %18 {ssbuffer.block_id = 3 : i32} : i32 to index
        %20 = arith.muli %19, %c1024 {ssbuffer.block_id = 3 : i32} : index
        %21 = arith.addi %20, %14 {ssbuffer.block_id = 3 : i32} : index
        %22 = arith.index_cast %arg12 {ssbuffer.block_id = 3 : i32} : i32 to index
        %23 = arith.addi %21, %7 {ssbuffer.block_id = 3 : i32} : index
        %24 = arith.subi %c0_i32, %17 {ssbuffer.block_id = 3 : i32} : i32
        %25 = arith.maxsi %24, %c0_i32 {ssbuffer.block_id = 3 : i32} : i32
        %26 = arith.index_cast %25 {ssbuffer.block_id = 3 : i32} : i32 to index
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 3 : i32} : memref<64x64xf16>
        %27 = arith.subi %23, %14 {ssbuffer.block_id = 3 : i32} : index
        %28 = arith.divsi %27, %c1024 {ssbuffer.block_id = 3 : i32} : index
        %29 = arith.subi %22, %28 {ssbuffer.block_id = 3 : i32} : index
        %30 = arith.maxsi %29, %c0 {ssbuffer.block_id = 3 : i32} : index
        %31 = arith.minsi %30, %c64 {ssbuffer.block_id = 3 : i32} : index
        %32 = arith.remsi %27, %c1024 {ssbuffer.block_id = 3 : i32} : index
        %33 = arith.subi %c128, %32 {ssbuffer.block_id = 3 : i32} : index
        %34 = arith.maxsi %33, %c0 {ssbuffer.block_id = 3 : i32} : index
        %35 = arith.minsi %34, %c64 {ssbuffer.block_id = 3 : i32} : index
        %36 = arith.minsi %26, %31 {ssbuffer.block_id = 3 : i32} : index
        %37 = arith.subi %31, %36 {ssbuffer.block_id = 3 : i32} : index
        %38 = arith.minsi %10, %35 {ssbuffer.block_id = 3 : i32} : index
        %39 = arith.subi %35, %38 {ssbuffer.block_id = 3 : i32} : index
        %40 = arith.cmpi slt, %37, %c64 {ssbuffer.block_id = 3 : i32} : index
        %41 = arith.cmpi slt, %39, %c64 {ssbuffer.block_id = 3 : i32} : index
        %42 = arith.ori %40, %41 {ssbuffer.block_id = 3 : i32} : i1
        scf.if %42 {
          linalg.fill {ssbuffer.block_id = 3 : i32} ins(%cst_0 : f16) outs(%alloc_5 : memref<64x64xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 3 : i32}
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 3 : i32} : memref<?xf16> to memref<64x64xf16, strided<[1024, 1], offset: ?>>
        %subview = memref.subview %reinterpret_cast[0, 0] [%37, %39] [1, 1] {ssbuffer.block_id = 3 : i32} : memref<64x64xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %subview_6 = memref.subview %alloc_5[%36, %38] [%37, %39] [1, 1] {ssbuffer.block_id = 3 : i32} : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
        memref.copy %subview, %subview_6 {ssbuffer.block_id = 3 : i32} : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1], offset: ?>>
        %43 = bufferization.to_tensor %alloc_5 restrict writable {ssbuffer.block_id = 3 : i32} : memref<64x64xf16>
        %44 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<64x64xf16>
        %transposed = linalg.transpose ins(%16 : tensor<64x64xf16>) outs(%44 : tensor<64x64xf16>) permutation = [1, 0]  {ssbuffer.block_id = 3 : i32}
        %45 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<64x64xf32>
        %46 = linalg.fill {ssbuffer.block_id = 3 : i32} ins(%cst : f32) outs(%45 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %47 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32} ins(%transposed, %43 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%46 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32} ins(%47 : tensor<64x64xf32>) outs(%alloc_4 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
        // CHECK:       llvm.load volatile
        // CHECK:       llvm.load volatile
        // CHECK:       arith.cmpi sgt
        // CHECK:       arith.cmpi sgt
        // CHECK:       llvm.load volatile
        // CHECK:       llvm.load volatile
        // CHECK:       arith.cmpi slt
        // CHECK:       arith.cmpi slt
        // CHECK:       arith.cmpi slt
        // CHECK:       scf.if
        %alloc_7 = memref.alloc() {ssbuffer.block_id = 4 : i32} : memref<64x64xf16>
        scf.if %42 {
          linalg.fill {ssbuffer.block_id = 4 : i32} ins(%cst_0 : f16) outs(%alloc_7 : memref<64x64xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
        %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 4 : i32} : memref<?xf16> to memref<64x64xf16, strided<[1024, 1], offset: ?>>
        %subview_9 = memref.subview %reinterpret_cast_8[0, 0] [%37, %39] [1, 1] {ssbuffer.block_id = 4 : i32} : memref<64x64xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %subview_10 = memref.subview %alloc_7[%36, %38] [%37, %39] [1, 1] {ssbuffer.block_id = 4 : i32} : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
        memref.copy %subview_9, %subview_10 {ssbuffer.block_id = 4 : i32} : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1], offset: ?>>
        %48 = bufferization.to_tensor %alloc_7 restrict writable {ssbuffer.block_id = 4 : i32} : memref<64x64xf16>
        %49 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<64x64xf32>
        %50 = linalg.fill {ssbuffer.block_id = 4 : i32} ins(%cst : f32) outs(%49 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %51 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32} ins(%transposed, %48 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%50 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32} ins(%51 : tensor<64x64xf32>) outs(%alloc_3 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
      } {DataUse, ssbuffer.block_id = 18 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
      hivm.hir.sync_block_wait {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
