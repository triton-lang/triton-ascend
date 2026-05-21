// RUN: triton-opt --update-for-ops %s --allow-unregistered-dialect | FileCheck %s

// CHECK: func.func @test_pipes_insert
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_pipes_insert(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32}, %arg9: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: f32, %arg11: f32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} dense<[4, 16, 16, 16]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 13 : i32} dense<[256, 4, 16]> : tensor<3xi64>
    %c8_i64 = arith.constant {ssbuffer.block_id = 18 : i32} 8 : i64
    %cst_1 = arith.constant {ssbuffer.block_id = 17 : i32} 0.000000e+00 : f32
    %c0_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 0 : i32
    %c3_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 3 : i32
    %c1_i64 = arith.constant {ssbuffer.block_id = 17 : i32} 1 : i64
    %c63_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 63 : i32
    %c256_i64 = arith.constant {ssbuffer.block_id = 17 : i32} 256 : i64
    %c512_i64 = arith.constant {ssbuffer.block_id = 17 : i32} 512 : i64
    %c64_i64 = arith.constant {ssbuffer.block_id = 17 : i32} 64 : i64
    %c2_i64 = arith.constant {ssbuffer.block_id = 17 : i32} 2 : i64
    %c1_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 1 : i32
    %c64_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 64 : i32
    %c2_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 2 : i32
    %c256_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 256 : i32
    %c0_i64 = arith.constant {ssbuffer.block_id = 17 : i32} 0 : i64
    %cst_2 = arith.constant {ssbuffer.block_id = 17 : i32} 1.000000e+00 : f32
    %c0 = arith.constant {ssbuffer.block_id = 17 : i32} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 17 : i32} 1 : index
    %c512 = arith.constant {ssbuffer.block_id = 17 : i32} 512 : index
    %c256 = arith.constant {ssbuffer.block_id = 17 : i32} 256 : index
    %cst_3 = arith.constant {ssbuffer.block_id = 17 : i32} 0.000000e+00 : f16
    %c64 = arith.constant {ssbuffer.block_id = 17 : i32} 64 : index
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 17 : i32} : tensor<256x64xf32>
      %1 = linalg.fill {ssbuffer.block_id = 17 : i32} ins(%cst_1 : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
      %2 = linalg.fill {ssbuffer.block_id = 17 : i32} ins(%cst_2 : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
      %3 = arith.cmpi sle, %arg12, %c256_i32 {ssbuffer.block_id = 17 : i32} : i32
      %4 = scf.if %3 -> (i64) {
        scf.yield %c2_i64 : i64
      } else {
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [2], sizes: [1], strides: [1] {ssbuffer.block_id = 7 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: 2>>
        %24 = memref.load %reinterpret_cast[%c0] {ssbuffer.block_id = 7 : i32} : memref<1xi32, strided<[1], offset: 2>>
        %25 = arith.extsi %24 {ssbuffer.block_id = 7 : i32} : i32 to i64
        scf.yield %25 : i64
      } {ssbuffer.block_id = 19 : i32}
      %5 = arith.muli %4, %c8_i64 {ssbuffer.block_id = 18 : i32} : i64
      %6 = arith.extsi %arg15 {ssbuffer.block_id = 18 : i32} : i32 to i64
      %7 = arith.minsi %6, %5 {ssbuffer.block_id = 18 : i32} : i64
      %8 = arith.divsi %5, %7 {ssbuffer.block_id = 18 : i32} : i64
      %9 = arith.addi %8, %c1_i64 {ssbuffer.block_id = 18 : i32} : i64
      %10 = arith.remsi %5, %7 {ssbuffer.block_id = 18 : i32} : i64
      %11 = arith.extsi %arg18 {ssbuffer.block_id = 18 : i32} : i32 to i64
      %12 = arith.cmpi slt, %11, %7 {ssbuffer.block_id = 18 : i32} : i64
      %13 = arith.cmpi slt, %11, %10 {ssbuffer.block_id = 18 : i32} : i64
      %14 = arith.muli %11, %9 {ssbuffer.block_id = 18 : i32} : i64
      %15 = arith.muli %10, %9 {ssbuffer.block_id = 18 : i32} : i64
      %16 = arith.subi %11, %10 {ssbuffer.block_id = 18 : i32} : i64
      %17 = arith.muli %16, %8 {ssbuffer.block_id = 18 : i32} : i64
      %18 = arith.addi %15, %17 {ssbuffer.block_id = 18 : i32} : i64
      %19 = arith.select %13, %14, %18 {ssbuffer.block_id = 18 : i32} : i64
      %20 = arith.select %12, %19, %c0_i64 {ssbuffer.block_id = 18 : i32} : i64
      %21 = arith.select %13, %9, %8 {ssbuffer.block_id = 18 : i32} : i64
      %22 = arith.select %12, %21, %c0_i64 {ssbuffer.block_id = 18 : i32} : i64
      %23 = arith.cmpi sge, %11, %7 {ssbuffer.block_id = 18 : i32} : i64
      scf.if %23 {
      } else {
        %24 = arith.cmpi sle, %22, %c0_i64 {ssbuffer.block_id = 16 : i32} : i64
        scf.if %24 {
        } else {
          %25 = arith.addi %arg13, %c63_i32 {ssbuffer.block_id = 15 : i32} : i32
          %26 = arith.divsi %25, %c64_i32 {ssbuffer.block_id = 15 : i32} : i32
          %27 = arith.extsi %26 {ssbuffer.block_id = 15 : i32} : i32 to i64
          %28 = arith.muli %22, %27 {ssbuffer.block_id = 15 : i32} : i64
          %29 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%arg10 : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
          %30 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%arg11 : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
          %alloc = memref.alloc() {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          %alloc_4 = memref.alloc() {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_set {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
          %alloc_5 = memref.alloc() {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 2 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 2 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_set {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
          scf.for %arg21 = %c0_i64 to %28 step %c1_i64  : i64 {
            // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
            %31 = arith.divsi %arg21, %27 {ssbuffer.block_id = 11 : i32} : i64
            %32 = arith.addi %20, %31 {ssbuffer.block_id = 11 : i32, ssbuffer.dep_mark = [5 : i32]} : i64
            %33 = arith.remsi %32, %4 {ssbuffer.block_id = 11 : i32, ssbuffer.dep_mark = [3 : i32, 4 : i32]} : i64
            // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
          } {ssbuffer.block_id = 26 : i32, ssbuffer.main_loop = 0 : i32, ssbuffer.vector_first}
          // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
          hivm.hir.sync_block_wait {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        } {ssbuffer.block_id = 27 : i32}
      } {ssbuffer.block_id = 28 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %0 = arith.cmpi sle, %arg12, %c256_i32 {ssbuffer.block_id = 17 : i32} : i32
      %1 = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<256x64xf32>
      %2 = linalg.fill {ssbuffer.block_id = 6 : i32} ins(%cst_1 : f32) outs(%1 : tensor<256x64xf32>) -> tensor<256x64xf32>
      %3 = scf.if %0 -> (i64) {
        scf.yield %c2_i64 : i64
      } else {
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [2], sizes: [1], strides: [1] {ssbuffer.block_id = 7 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: 2>>
        %23 = memref.load %reinterpret_cast[%c0] {ssbuffer.block_id = 7 : i32} : memref<1xi32, strided<[1], offset: 2>>
        %24 = arith.extsi %23 {ssbuffer.block_id = 7 : i32} : i32 to i64
        scf.yield %24 : i64
      } {ssbuffer.block_id = 19 : i32}
      %4 = arith.muli %3, %c8_i64 {ssbuffer.block_id = 18 : i32} : i64
      %5 = arith.extsi %arg15 {ssbuffer.block_id = 18 : i32} : i32 to i64
      %6 = arith.minsi %5, %4 {ssbuffer.block_id = 18 : i32} : i64
      %7 = arith.divsi %4, %6 {ssbuffer.block_id = 18 : i32} : i64
      %8 = arith.addi %7, %c1_i64 {ssbuffer.block_id = 18 : i32} : i64
      %9 = arith.remsi %4, %6 {ssbuffer.block_id = 18 : i32} : i64
      %10 = arith.extsi %arg18 {ssbuffer.block_id = 18 : i32} : i32 to i64
      %11 = arith.cmpi slt, %10, %6 {ssbuffer.block_id = 18 : i32} : i64
      %12 = arith.cmpi slt, %10, %9 {ssbuffer.block_id = 18 : i32} : i64
      %13 = arith.muli %10, %8 {ssbuffer.block_id = 18 : i32} : i64
      %14 = arith.muli %9, %8 {ssbuffer.block_id = 18 : i32} : i64
      %15 = arith.subi %10, %9 {ssbuffer.block_id = 18 : i32} : i64
      %16 = arith.muli %15, %7 {ssbuffer.block_id = 18 : i32} : i64
      %17 = arith.addi %14, %16 {ssbuffer.block_id = 18 : i32} : i64
      %18 = arith.select %12, %13, %17 {ssbuffer.block_id = 18 : i32} : i64
      %19 = arith.select %11, %18, %c0_i64 {ssbuffer.block_id = 18 : i32} : i64
      %20 = arith.select %12, %8, %7 {ssbuffer.block_id = 18 : i32} : i64
      %21 = arith.select %11, %20, %c0_i64 {ssbuffer.block_id = 18 : i32} : i64
      %22 = arith.cmpi sge, %10, %6 {ssbuffer.block_id = 18 : i32} : i64
      scf.if %22 {
      } else {
        %23 = arith.cmpi sle, %21, %c0_i64 {ssbuffer.block_id = 16 : i32} : i64
        scf.if %23 {
        } else {
          %24 = arith.addi %arg13, %c63_i32 {ssbuffer.block_id = 15 : i32} : i32
          %25 = arith.divsi %24, %c64_i32 {ssbuffer.block_id = 15 : i32} : i32
          %26 = arith.extsi %25 {ssbuffer.block_id = 15 : i32} : i32 to i64
          %27 = arith.muli %21, %26 {ssbuffer.block_id = 15 : i32} : i64
          %alloc = memref.alloc() {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.sync_block_set {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          %alloc_4 = memref.alloc() {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
          %alloc_5 = memref.alloc() {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 2 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 2 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
          // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
          scf.for %arg21 = %c0_i64 to %27 step %c1_i64  : i64 {
            // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
            %28 = arith.divsi %arg21, %26 {ssbuffer.block_id = 11 : i32} : i64
            %29 = arith.addi %19, %28 {ssbuffer.block_id = 11 : i32} : i64
            %30 = arith.remsi %29, %3 {ssbuffer.block_id = 11 : i32} : i64
            %alloc_6 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<256x64xf16>
            %alloc_7 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<64x64xf16>
            %alloc_8 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<64x64xf16>
            // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
          } {ssbuffer.block_id = 26 : i32, ssbuffer.main_loop = 0 : i32, ssbuffer.vector_first}
          hivm.hir.sync_block_wait {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
          hivm.hir.sync_block_wait {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        } {ssbuffer.block_id = 27 : i32}
      } {ssbuffer.block_id = 28 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}