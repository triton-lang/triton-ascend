// RUN: triton-opt --analyze-data-flow --verify-diagnostics %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} { // expected-error {{[analyze-data-flow] Pass failed!}}
  func.func @test_tensor_args_exist_dependency(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c16384_i32 = arith.constant {ssbuffer.block_id = 16 : i32} 16384 : i32
    %cst = arith.constant {ssbuffer.block_id = 12 : i32} dense<[2, 4, 16, 16]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 12 : i32} dense<[64, 2, 16]> : tensor<3xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 15 : i32} 0.000000e+00 : f32
    %c0_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 0 : i32
    %c128_i64 = arith.constant {ssbuffer.block_id = 15 : i32} 128 : i64
    %c1_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 1 : i32
    %c64_i32 = arith.constant {MixUse, ssbuffer.block_id = 15 : i32} 64 : i32
    %c32_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 32 : i32
    %c64 = arith.constant {DataUse, ssbuffer.block_id = 15 : i32} 64 : index
    %c0 = arith.constant {DataUse, ssbuffer.block_id = 15 : i32} 0 : index
    %c63_i32 = arith.constant {ssbuffer.block_id = 15 : i32} 63 : i32
    %c16384_i64 = arith.constant {ssbuffer.block_id = 15 : i32} 16384 : i64
    %c524288_i64 = arith.constant {ssbuffer.block_id = 15 : i32} 524288 : i64
    %c128 = arith.constant {ssbuffer.block_id = 15 : i32} 128 : index
    %c32 = arith.constant {ssbuffer.block_id = 15 : i32} 32 : index
    %c4096 = arith.constant {ssbuffer.block_id = 15 : i32} 4096 : index
    %cst_2 = arith.constant {ssbuffer.block_id = 15 : i32} 0.000000e+00 : bf16
    %c8192 = arith.constant {ssbuffer.block_id = 15 : i32} 8192 : index
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<64x32xf32>
      %1 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst_1 : f32) outs(%0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %2 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<64xf32>
      %3 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%cst_1 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
      %4 = arith.divsi %arg14, %c32_i32 {ssbuffer.block_id = 15 : i32} : i32
      %5 = arith.remsi %arg14, %c32_i32 {ssbuffer.block_id = 15 : i32} : i32
      %6 = arith.muli %4, %arg9 {ssbuffer.block_id = 15 : i32} : i32
      %7 = arith.addi %arg9, %c63_i32 {ssbuffer.block_id = 15 : i32} : i32
      %8 = arith.divsi %7, %c64_i32 {ssbuffer.block_id = 15 : i32} : i32
      %9 = arith.muli %4, %8 {ssbuffer.block_id = 15 : i32} : i32
      %10 = arith.muli %9, %c32_i32 {ssbuffer.block_id = 15 : i32} : i32
      %11 = arith.addi %10, %5 {ssbuffer.block_id = 15 : i32} : i32
      %12 = arith.extsi %11 {ssbuffer.block_id = 15 : i32} : i32 to i64
      %13 = arith.muli %12, %c16384_i64 {ssbuffer.block_id = 15 : i32} : i64
      %14 = arith.muli %6, %c32_i32 {ssbuffer.block_id = 15 : i32} : i32
      %15 = arith.addi %14, %5 {ssbuffer.block_id = 15 : i32} : i32
      %16 = arith.extsi %15 {ssbuffer.block_id = 15 : i32} : i32 to i64
      %17 = arith.muli %16, %c128_i64 {ssbuffer.block_id = 15 : i32} : i64
      %18 = arith.index_cast %17 {ssbuffer.block_id = 15 : i32} : i64 to index
      %19 = arith.muli %arg13, %c32_i32 {ssbuffer.block_id = 15 : i32} : i32
      %20 = arith.index_cast %15 {ssbuffer.block_id = 15 : i32} : i32 to index
      %alloc = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 1 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 1 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 3 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 3 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
      %alloc_8 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
      %alloc_9 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
      %21:2 = scf.for %arg16 = %c0_i32 to %8 step %c1_i32 iter_args(%arg17 = %1, %arg18 = %1) -> (tensor<64x32xf32>, tensor<64x32xf32>)  : i32 {
        %57 = arith.maxsi %19, %c0_i32 {ssbuffer.block_id = 12 : i32} : i32
        %58 = arith.index_cast %57 {ssbuffer.block_id = 12 : i32} : i32 to index
        %59 = arith.truncf %arg17 {DataUse, ssbuffer.block_id = 12 : i32} : tensor<64x32xf32> to tensor<64x32xbf16>
        %reshape = tensor.reshape %59(%cst_0) {ssbuffer.block_id = 12 : i32} : (tensor<64x32xbf16>, tensor<3xi64>) -> tensor<64x2x16xbf16>
        %60 = tensor.empty() {ssbuffer.block_id = 12 : i32} : tensor<2x64x16xbf16>
        %transposed = linalg.transpose ins(%reshape : tensor<64x2x16xbf16>) outs(%60 : tensor<2x64x16xbf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 12 : i32}
        %reshape_13 = tensor.reshape %transposed(%cst) {ssbuffer.block_id = 12 : i32} : (tensor<2x64x16xbf16>, tensor<4xi64>) -> tensor<2x4x16x16xbf16>
        %61 = arith.subi %c0_i32, %19 {ssbuffer.block_id = 12 : i32} : i32
        %62 = arith.maxsi %61, %c0_i32 {ssbuffer.block_id = 12 : i32} : i32
        %63 = arith.index_cast %62 {ssbuffer.block_id = 12 : i32} : i32 to index
        %64 = arith.truncf %arg18 {DataUse, ssbuffer.block_id = 12 : i32} : tensor<64x32xf32> to tensor<64x32xbf16>
        %reshape_14 = tensor.reshape %64(%cst_0) {ssbuffer.block_id = 12 : i32} : (tensor<64x32xbf16>, tensor<3xi64>) -> tensor<64x2x16xbf16>
        %167 = arith.mulf %arg17, %1 {DataUse, ssbuffer.block_id = 14 : i32} : tensor<64x32xf32>
        %168 = arith.mulf %arg18, %1 {DataUse, ssbuffer.block_id = 14 : i32} : tensor<64x32xf32>
        %169 = arith.addf %1, %167 {ssbuffer.block_id = 14 : i32} : tensor<64x32xf32>
        %170 = arith.extf %64 {DataUse, ssbuffer.block_id = 14 : i32} : tensor<64x32xbf16> to tensor<64x32xf32>
        hivm.hir.sync_block_set {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
        hivm.hir.sync_block_set {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
        scf.yield %169, %170 : tensor<64x32xf32>, tensor<64x32xf32>
      } {DataUse, ssbuffer.block_id = 17 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
