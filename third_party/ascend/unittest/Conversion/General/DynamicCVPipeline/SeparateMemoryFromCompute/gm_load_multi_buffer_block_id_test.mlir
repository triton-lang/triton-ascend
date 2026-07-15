// RUN: triton-opt --gm-load-multi-buffer %s | FileCheck %s

// ============================================================================
// Test Group B1: DTS1 VECTOR - two independent marked loads with shared address
//
// Based on dts1_result_v1_multibuffer_ir_issues.zh.md:
// - Problem 1: Two marked loads must each have independent multi-buffer
//   state (buffer slots, flags, counters, slot selection).
// - Problem 3: Top-level block_id runs must be consecutive.
//   Old pass produced 7 -> 8 -> 7 -> 8 -> 9; fix must produce 7 -> 8 -> 9.
//
// Key pattern: block 7 has remaining body ops (%24=muli, %25=index_cast)
// used by block 9 store address (%46=muli %25). These must NOT be placed
// between two block 8 runs.
// ============================================================================

// CHECK-LABEL: func.func @_attn_bwd
func.func @_attn_bwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: f32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
  %cst = arith.constant {ssbuffer.block_id = 7 : i32} dense<[4, 4, 16, 16]> : tensor<4xi64>
  %cst_0 = arith.constant {ssbuffer.block_id = 7 : i32} dense<[64, 4, 16]> : tensor<3xi64>
  %c64_i32 = arith.constant {MixUse, ssbuffer.block_id = 12 : i32} 64 : i32
  %c8388608_i32 = arith.constant {MixUse, ssbuffer.block_id = 12 : i32} 8388608 : i32
  %c1048576_i32 = arith.constant {MixUse, ssbuffer.block_id = 12 : i32} 1048576 : i32
  %c8_i32 = arith.constant {MixUse, ssbuffer.block_id = 12 : i32} 8 : i32
  %c8192_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 8192 : i32
  %c128_i32 = arith.constant {MixUse, ssbuffer.block_id = 12 : i32} 128 : i32
  %c131072_i32 = arith.constant {Undefined, ssbuffer.block_id = 12 : i32} 131072 : i32
  %c28_i32 = arith.constant {Undefined, ssbuffer.block_id = 12 : i32} 28 : i32
  %c0_i32 = arith.constant {Undefined, ssbuffer.block_id = 12 : i32} 0 : i32
  %c1_i32 = arith.constant {Undefined, ssbuffer.block_id = 12 : i32} 1 : i32
  %cst_1 = arith.constant {ssbuffer.block_id = 12 : i32} 0.000000e+00 : f32
  %c128 = arith.constant {ssbuffer.block_id = 12 : i32} 128 : index

  // R1: Each load gets 2 independent buffer slots (4 total)
  // CHECK: memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<64xf32>
  // CHECK: memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<64xf32>
  // CHECK: memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<64xf32>
  // CHECK: memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<64xf32>

  // R1: 2 original iter_args + 8 control (4 per load) = 10 total
  // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<64x128xf32>, tensor<64x128xf32>, i1, i1, index, index, i1, i1, index, index)

  scope.scope : () -> () {
    %0 = tensor.empty() {ssbuffer.block_id = 12 : i32} : tensor<64x128xf32>
    %1 = linalg.fill {ssbuffer.block_id = 12 : i32} ins(%cst_1 : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = tensor.empty() {ssbuffer.block_id = 12 : i32} : tensor<64x64xf32>
    %3 = linalg.fill {ssbuffer.block_id = 12 : i32} ins(%cst_1 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %4 = linalg.fill {ssbuffer.block_id = 12 : i32} ins(%arg11 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.for %arg18 = %arg15 to %c131072_i32 step %c28_i32  : i32 {
      %5 = arith.divsi %arg18, %c128_i32 {MixUse, ssbuffer.block_id = 10 : i32} : i32
      %6 = arith.muli %5, %c8192_i32 {ssbuffer.block_id = 10 : i32} : i32
      %7 = arith.remsi %5, %c8_i32 {MixUse, ssbuffer.block_id = 10 : i32} : i32
      %8 = arith.muli %7, %c1048576_i32 {MixUse, ssbuffer.block_id = 10 : i32} : i32
      %9 = arith.divsi %5, %c8_i32 {MixUse, ssbuffer.block_id = 10 : i32} : i32
      %10 = arith.muli %9, %c8388608_i32 {MixUse, ssbuffer.block_id = 10 : i32} : i32
      %11 = arith.addi %8, %10 {MixUse, ssbuffer.block_id = 10 : i32} : i32
      %12 = arith.index_cast %11 {ssbuffer.block_id = 10 : i32} : i32 to index
      %13 = arith.index_cast %6 {ssbuffer.block_id = 10 : i32} : i32 to index
      %alloc = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_2 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
      %alloc_8 = memref.alloc() {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8

      // ── Inner loop (main_loop) ────────────────────────────────────────

      // R2: Block 7 remaining ops (%24=muli %arg19, %25=index_cast %24)
      // must stay in block 7 run, BEFORE block 8 starts.
      // This produces: 7 -> 8 -> 9 (not 7 -> 8 -> 7 -> 8 -> 9).

      // Block 7: first producer slot
      // CHECK: scf.if {{.*}} -> (i1, index) {
      // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 7
      // CHECK: } {ssbuffer.block_id = 7 : i32, ssbuffer.load_store}

      // Block 7: second producer slot (uses first slot's updated index)
      // CHECK: scf.if {{.*}} -> (i1, index) {
      // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 7
      // CHECK: } {ssbuffer.block_id = 7 : i32, ssbuffer.load_store}

      // Block 7: consumer select
      // CHECK: arith.remui {{.*}} {ssbuffer.block_id = 7
      // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 7
      // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 7
      // CHECK: arith.select {{.*}} {ssbuffer.block_id = 7

      // Block 7: flag clear (now before compute)
      // CHECK: scf.if {{.*}} -> (i1) {
      // CHECK: } {ssbuffer.block_id = 7 : i32, ssbuffer.load_store}
      // CHECK: scf.if {{.*}} -> (i1) {
      // CHECK: } {ssbuffer.block_id = 7 : i32, ssbuffer.load_store}

      // Block 7: compute using selected tensor
      // CHECK: arith.mulf {{.*ssbuffer.block_id = 7}}
      // CHECK: linalg.broadcast ins({{.*}} : tensor<64xf32>) {{.*}} {ssbuffer.block_id = 7
      // CHECK: math.exp {{.*}} ssbuffer.block_id = 7

      // Block 7: sync + copy to cbuf
      // CHECK: hivm.hir.sync_block_wait {{.*ssbuffer.block_id = 7}}
      // CHECK: hivm.hir.copy {{.*ssbuffer.block_id = 7}}
      // CHECK: hivm.hir.sync_block_set {{.*ssbuffer.block_id = 7}}

      // Block 8: first producer slot (independent from block 7)
      // CHECK: scf.if {{.*}} -> (i1, index) {
      // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 8
      // CHECK: } {ssbuffer.block_id = 8 : i32, ssbuffer.load_store}

      // Block 8: second producer slot
      // CHECK: scf.if {{.*}} -> (i1, index) {
      // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 8
      // CHECK: } {ssbuffer.block_id = 8 : i32, ssbuffer.load_store}

      // Block 8: consumer select (independent from block 7)
      // CHECK: arith.remui {{.*}} {ssbuffer.block_id = 8
      // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 8
      // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 8
      // CHECK: arith.select {{.*}} {ssbuffer.block_id = 8

      // Block 8: flag clear (now before compute)
      // CHECK: scf.if {{.*}} -> (i1) {
      // CHECK: } {ssbuffer.block_id = 8 : i32, ssbuffer.load_store}
      // CHECK: scf.if {{.*}} -> (i1) {
      // CHECK: } {ssbuffer.block_id = 8 : i32, ssbuffer.load_store}

      // Block 8: compute using selected tensor
      // CHECK: arith.extf {{.*ssbuffer.block_id = 8}}
      // CHECK: arith.mulf {{.*ssbuffer.block_id = 8}}

      // Block 8: sync + copy to cbuf
      // CHECK: hivm.hir.sync_block_wait {{.*ssbuffer.block_id = 8}}
      // CHECK: hivm.hir.copy {{.*ssbuffer.block_id = 8}}
      // CHECK: hivm.hir.sync_block_set {{.*ssbuffer.block_id = 8}}

      // Block 9: tail (uses %25 from block 7 for store address)
      // CHECK: hivm.hir.sync_block_wait {{.*ssbuffer.block_id = 9}}
      // CHECK: arith.muli {{.*ssbuffer.block_id = 9}}
      // CHECK: arith.addi {{.*ssbuffer.block_id = 9}}
      // CHECK: hivm.hir.store {{.*ssbuffer.block_id = 9}}
      // CHECK: hivm.hir.sync_block_set {{.*ssbuffer.block_id = 9}}

      // scf.yield with 10 values
      // CHECK: scf.yield {{.*}} : tensor<64x128xf32>, tensor<64x128xf32>, i1, i1, index, index, i1, i1, index, index

      %14:2 = scf.for %arg19 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg20 = %1, %arg21 = %1) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 4
        %memspacecast = memref.memory_space_cast %alloc_4 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %23 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xf32>
        %24 = arith.muli %arg19, %c64_i32 {MixUse, ssbuffer.block_id = 7 : i32} : i32
        %25 = arith.index_cast %24 {ssbuffer.block_id = 7 : i32} : i32 to index
        %26 = arith.addi %13, %25 {ssbuffer.block_id = 7 : i32} : index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg9 to offset: [%26], sizes: [64], strides: [1] {ssbuffer.block_id = 7 : i32} : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
        %alloc_11 = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<64xf32>
        memref.copy %reinterpret_cast_10, %alloc_11 {ssbuffer.block_id = 7 : i32} : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32>
        %27 = bufferization.to_tensor %alloc_11 restrict writable {gm_load_bufferable, ssbuffer.block_id = 7 : i32} : memref<64xf32>
        %28 = arith.addf %23, %3 {ssbuffer.block_id = 7 : i32} : tensor<64x64xf32>
        %29 = arith.mulf %28, %4 {DataUse, ssbuffer.block_id = 7 : i32} : tensor<64x64xf32>
        %broadcasted = linalg.broadcast ins(%27 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1]  {ssbuffer.block_id = 7 : i32}
        %30 = arith.subf %29, %broadcasted {DataUse, ssbuffer.block_id = 7 : i32} : tensor<64x64xf32>
        %31 = math.exp %30 {DataUse, ssbuffer.block_id = 7 : i32} : tensor<64x64xf32>
        %32 = arith.truncf %31 {DataUse, ssbuffer.block_id = 7 : i32} : tensor<64x64xf32> to tensor<64x64xf16>
        %reshape = tensor.reshape %32(%cst_0) {ssbuffer.block_id = 7 : i32} : (tensor<64x64xf16>, tensor<3xi64>) -> tensor<64x4x16xf16>
        %33 = tensor.empty() {ssbuffer.block_id = 7 : i32} : tensor<4x64x16xf16>
        %transposed = linalg.transpose ins(%reshape : tensor<64x4x16xf16>) outs(%33 : tensor<4x64x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 7 : i32}
        %reshape_12 = tensor.reshape %transposed(%cst) {ssbuffer.block_id = 7 : i32} : (tensor<4x64x16xf16>, tensor<4xi64>) -> tensor<4x4x16x16xf16>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        hivm.hir.copy ins(%reshape_12 : tensor<4x4x16x16xf16>) outs(%alloc_2 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 1 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        hivm.hir.sync_block_wait {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 5
        %memspacecast_13 = memref.memory_space_cast %alloc_5 {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %34 = bufferization.to_tensor %memspacecast_13 restrict writable {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x64xf32>
        %reinterpret_cast_14 = memref.reinterpret_cast %arg10 to offset: [%26], sizes: [64], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
        %alloc_15 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<64xf32>
        memref.copy %reinterpret_cast_14, %alloc_15 {ssbuffer.block_id = 8 : i32} : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32>
        %35 = bufferization.to_tensor %alloc_15 restrict writable {gm_load_bufferable, ssbuffer.block_id = 8 : i32} : memref<64xf32>
        %36 = arith.addf %34, %3 {ssbuffer.block_id = 8 : i32} : tensor<64x64xf32>
        %broadcasted_16 = linalg.broadcast ins(%35 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1]  {ssbuffer.block_id = 8 : i32}
        %37 = arith.subf %36, %broadcasted_16 {DataUse, ssbuffer.block_id = 8 : i32} : tensor<64x64xf32>
        %38 = arith.extf %32 {DataUse, ssbuffer.block_id = 8 : i32} : tensor<64x64xf16> to tensor<64x64xf32>
        %39 = arith.mulf %38, %37 {DataUse, ssbuffer.block_id = 8 : i32} : tensor<64x64xf32>
        %40 = arith.mulf %39, %4 {DataUse, ssbuffer.block_id = 8 : i32} : tensor<64x64xf32>
        %41 = arith.truncf %40 {DataUse, ssbuffer.block_id = 8 : i32} : tensor<64x64xf32> to tensor<64x64xf16>
        %reshape_17 = tensor.reshape %41(%cst_0) {ssbuffer.block_id = 8 : i32} : (tensor<64x64xf16>, tensor<3xi64>) -> tensor<64x4x16xf16>
        %42 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<4x64x16xf16>
        %transposed_18 = linalg.transpose ins(%reshape_17 : tensor<64x4x16xf16>) outs(%42 : tensor<4x64x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 8 : i32}
        %reshape_19 = tensor.reshape %transposed_18(%cst) {ssbuffer.block_id = 8 : i32} : (tensor<4x64x16xf16>, tensor<4xi64>) -> tensor<4x4x16x16xf16>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.copy ins(%reshape_19 : tensor<4x4x16x16xf16>) outs(%alloc : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
        hivm.hir.copy ins(%reshape_19 : tensor<4x4x16x16xf16>) outs(%alloc_3 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
        hivm.hir.sync_block_set {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
        hivm.hir.sync_block_wait {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 8
        %memspacecast_20 = memref.memory_space_cast %alloc_8 {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %43 = bufferization.to_tensor %memspacecast_20 restrict writable {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x128xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
        %memspacecast_21 = memref.memory_space_cast %alloc_7 {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %44 = bufferization.to_tensor %memspacecast_21 restrict writable {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x128xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
        %memspacecast_22 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %45 = bufferization.to_tensor %memspacecast_22 restrict writable {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x128xf32>
        %46 = arith.muli %25, %c128 {ssbuffer.block_id = 9 : i32} : index
        %47 = arith.addi %12, %46 {ssbuffer.block_id = 9 : i32} : index
        %48 = arith.addf %44, %arg21 {ssbuffer.block_id = 9 : i32} : tensor<64x128xf32>
        %49 = arith.addf %43, %arg20 {ssbuffer.block_id = 9 : i32} : tensor<64x128xf32>
        %50 = arith.addf %45, %1 {ssbuffer.block_id = 9 : i32} : tensor<64x128xf32>
        %reinterpret_cast_23 = memref.reinterpret_cast %arg6 to offset: [%47], sizes: [64, 128], strides: [128, 1] {ssbuffer.block_id = 9 : i32} : memref<?xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
        %51 = arith.truncf %50 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<64x128xf32> to tensor<64x128xf16>
        hivm.hir.store ins(%51 : tensor<64x128xf16>) outs(%reinterpret_cast_23 : memref<64x128xf16, strided<[128, 1], offset: ?>>) {ssbuffer.block_id = 9 : i32} atomic = <add>
        hivm.hir.sync_block_set {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
        hivm.hir.sync_block_set {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
        hivm.hir.sync_block_set {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
        scf.yield %49, %48 : tensor<64x128xf32>, tensor<64x128xf32>
      } {DataUse, ssbuffer.block_id = 13 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
      hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
      hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      %15 = arith.muli %5, %c128_i32 {ssbuffer.block_id = 11 : i32} : i32
      %16 = arith.subi %arg18, %15 {ssbuffer.block_id = 11 : i32} : i32
      %17 = arith.muli %16, %c64_i32 {ssbuffer.block_id = 11 : i32} : i32
      %18 = arith.index_cast %17 {ssbuffer.block_id = 11 : i32} : i32 to index
      %19 = arith.muli %18, %c128 {ssbuffer.block_id = 11 : i32} : index
      %20 = arith.addi %12, %19 {ssbuffer.block_id = 11 : i32} : index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%20], sizes: [64, 128], strides: [128, 1] {ssbuffer.block_id = 11 : i32} : memref<?xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
      %21 = arith.truncf %14#0 {DataUse, ssbuffer.block_id = 11 : i32} : tensor<64x128xf32> to tensor<64x128xf16>
      bufferization.materialize_in_destination %21 in writable %reinterpret_cast {ssbuffer.block_id = 11 : i32} : (tensor<64x128xf16>, memref<64x128xf16, strided<[128, 1], offset: ?>>) -> ()
      %reinterpret_cast_9 = memref.reinterpret_cast %arg8 to offset: [%20], sizes: [64, 128], strides: [128, 1] {ssbuffer.block_id = 11 : i32} : memref<?xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
      %22 = arith.truncf %14#1 {DataUse, ssbuffer.block_id = 11 : i32} : tensor<64x128xf32> to tensor<64x128xf16>
      bufferization.materialize_in_destination %22 in writable %reinterpret_cast_9 {ssbuffer.block_id = 11 : i32} : (tensor<64x128xf16>, memref<64x128xf16, strided<[128, 1], offset: ?>>) -> ()
    } {Undefined, ssbuffer.block_id = 14 : i32}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// ============================================================================
// Test Group B2: Complex multi-buffer - two gm_load_bufferable with inner
// alloc + fill + copy pattern, block_id propagation through blocks 12/13/14
//
// Two marked loads share the same loop but use different buffer types:
// - Load 1: memref<64x32xbf16> (alloc inside loop, fill + copy then read)
// - Load 2: memref<64xf32> (alloc inside loop, fill + copy then read)
//
// Each gets 2 buffer slots (double buffer), 4 control iter_args each.
// ============================================================================

// CHECK-LABEL: func.func @_attn_bwd_complex
func.func @_attn_bwd_complex(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16>, %arg3: memref<?xbf16>, %arg4: memref<?xbf16>, %arg5: memref<?xbf16>, %arg6: memref<?xf32>, %arg7: memref<?xbf16>, %arg8: memref<?xf32>, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
  %cst = arith.constant dense<[2, 4, 16, 16]> : tensor<4xi64>
  %cst_0 = arith.constant dense<[64, 2, 16]> : tensor<3xi64>
  %cst_1 = arith.constant 0.000000e+00 : f32
  %cst_2 = arith.constant 0.000000e+00 : bf16
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %c63_i32 = arith.constant 63 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i64 = arith.constant 128 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c16384_i32 = arith.constant 16384 : i32
  %c524288_i64 = arith.constant 524288 : i64
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c8192 = arith.constant 8192 : index
  %c4096 = arith.constant 4096 : index

  // R1: Each load gets 2 buffer slots (4 total: 2 bf16 + 2 f32)
  // CHECK: memref.alloc() {ssbuffer.block_id = 12 : i32} : memref<64x32xbf16>
  // CHECK: memref.alloc() {ssbuffer.block_id = 12 : i32} : memref<64x32xbf16>
  // CHECK: memref.alloc() {ssbuffer.block_id = 12 : i32} : memref<64xf32>
  // CHECK: memref.alloc() {ssbuffer.block_id = 12 : i32} : memref<64xf32>

  // R1: 2 original iter_args + 8 control (4 per load) = 10 total
  // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<64x32xf32>, tensor<64x32xf32>, i1, i1, index, index, i1, i1, index, index)

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
      %alloc = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 1 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 1 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 3 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 3 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
      %alloc_8 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
      %alloc_9 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
      %21:2 = scf.for %arg16 = %c0_i32 to %8 step %c1_i32 iter_args(%arg17 = %1, %arg18 = %1) -> (tensor<64x32xf32>, tensor<64x32xf32>)  : i32 {
        %57 = arith.extsi %arg16 {ssbuffer.block_id = 12 : i32} : i32 to i64
        %58 = arith.muli %57, %c524288_i64 {ssbuffer.block_id = 12 : i32} : i64
        %59 = arith.addi %13, %58 {ssbuffer.block_id = 12 : i32} : i64
        %60 = arith.index_cast %59 {ssbuffer.block_id = 12 : i32} : i64 to index
        %61 = arith.maxsi %19, %c0_i32 {ssbuffer.block_id = 12 : i32} : i32
        %62 = arith.index_cast %61 {ssbuffer.block_id = 12 : i32} : i32 to index
        %63 = arith.addi %60, %62 {ssbuffer.block_id = 12 : i32} : index
        %reinterpret_cast_13 = memref.reinterpret_cast %arg7 to offset: [%63], sizes: [64, 32], strides: [128, 1] {ssbuffer.block_id = 12 : i32} : memref<?xbf16> to memref<64x32xbf16, strided<[128, 1], offset: ?>>
        %64 = arith.truncf %arg17 {DataUse, ssbuffer.block_id = 12 : i32} : tensor<64x32xf32> to tensor<64x32xbf16>
        %reshape = tensor.reshape %64(%cst_0) {ssbuffer.block_id = 12 : i32} : (tensor<64x32xbf16>, tensor<3xi64>) -> tensor<64x2x16xbf16>
        %65 = tensor.empty() {ssbuffer.block_id = 12 : i32} : tensor<2x64x16xbf16>
        %transposed = linalg.transpose ins(%reshape : tensor<64x2x16xbf16>) outs(%65 : tensor<2x64x16xbf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 12 : i32}
        %reshape_14 = tensor.reshape %transposed(%cst) {ssbuffer.block_id = 12 : i32} : (tensor<2x64x16xbf16>, tensor<4xi64>) -> tensor<2x4x16x16xbf16>
        %66 = arith.divsi %62, %c128 {ssbuffer.block_id = 12 : i32} : index
        %67 = arith.subi %c128, %66 {ssbuffer.block_id = 12 : i32} : index
        %68 = arith.maxsi %67, %c0 {ssbuffer.block_id = 12 : i32} : index
        %69 = arith.minsi %68, %c64 {ssbuffer.block_id = 12 : i32} : index
        %70 = arith.remsi %62, %c128 {ssbuffer.block_id = 12 : i32} : index
        %71 = arith.subi %c128, %70 {ssbuffer.block_id = 12 : i32} : index
        %72 = arith.maxsi %71, %c0 {ssbuffer.block_id = 12 : i32} : index
        %73 = arith.minsi %72, %c32 {ssbuffer.block_id = 12 : i32} : index
        %74 = arith.minsi %69, %c0 {ssbuffer.block_id = 12 : i32} : index
        %75 = arith.subi %69, %74 {ssbuffer.block_id = 12 : i32} : index
        %76 = arith.subi %c0_i32, %19 {ssbuffer.block_id = 12 : i32} : i32
        %77 = arith.maxsi %76, %c0_i32 {ssbuffer.block_id = 12 : i32} : i32
        %78 = arith.index_cast %77 {ssbuffer.block_id = 12 : i32} : i32 to index
        %79 = arith.minsi %78, %73 {ssbuffer.block_id = 12 : i32} : index
        %80 = arith.subi %73, %79 {ssbuffer.block_id = 12 : i32} : index
        %extracted_slice_15 = tensor.extract_slice %64[%74, %79] [%75, %80] [1, 1] {ssbuffer.block_id = 12 : i32} : tensor<64x32xbf16> to tensor<?x?xbf16>
        %subview_16 = memref.subview %reinterpret_cast_13[0, 0] [%75, %80] [1, 1] {ssbuffer.block_id = 12 : i32} : memref<64x32xbf16, strided<[128, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice_15 in writable %subview_16 {ssbuffer.block_id = 12 : i32} : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[128, 1], offset: ?>>) -> ()
        %81 = arith.addi %60, %c8192 {ssbuffer.block_id = 12 : i32} : index
        %82 = arith.addi %81, %62 {ssbuffer.block_id = 12 : i32} : index
        %reinterpret_cast_17 = memref.reinterpret_cast %arg7 to offset: [%82], sizes: [64, 32], strides: [128, 1] {ssbuffer.block_id = 12 : i32} : memref<?xbf16> to memref<64x32xbf16, strided<[128, 1], offset: ?>>
        %83 = arith.truncf %arg18 {DataUse, ssbuffer.block_id = 12 : i32} : tensor<64x32xf32> to tensor<64x32xbf16>
        %reshape_18 = tensor.reshape %83(%cst_0) {ssbuffer.block_id = 12 : i32} : (tensor<64x32xbf16>, tensor<3xi64>) -> tensor<64x2x16xbf16>
        %84 = tensor.empty() {ssbuffer.block_id = 12 : i32} : tensor<2x64x16xbf16>
        %transposed_19 = linalg.transpose ins(%reshape_18 : tensor<64x2x16xbf16>) outs(%84 : tensor<2x64x16xbf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 12 : i32}
        %reshape_20 = tensor.reshape %transposed_19(%cst) {ssbuffer.block_id = 12 : i32} : (tensor<2x64x16xbf16>, tensor<4xi64>) -> tensor<2x4x16x16xbf16>
        %85 = arith.subi %82, %60 {ssbuffer.block_id = 12 : i32} : index
        %86 = arith.divsi %85, %c128 {ssbuffer.block_id = 12 : i32} : index
        %87 = arith.subi %c128, %86 {ssbuffer.block_id = 12 : i32} : index
        %88 = arith.maxsi %87, %c0 {ssbuffer.block_id = 12 : i32} : index
        %89 = arith.minsi %88, %c64 {ssbuffer.block_id = 12 : i32} : index
        %90 = arith.remsi %85, %c128 {ssbuffer.block_id = 12 : i32} : index
        %91 = arith.subi %c128, %90 {ssbuffer.block_id = 12 : i32} : index
        %92 = arith.maxsi %91, %c0 {ssbuffer.block_id = 12 : i32} : index
        %93 = arith.minsi %92, %c32 {ssbuffer.block_id = 12 : i32} : index
        %94 = arith.minsi %89, %c0 {ssbuffer.block_id = 12 : i32} : index
        %95 = arith.subi %89, %94 {ssbuffer.block_id = 12 : i32} : index
        %96 = arith.minsi %78, %93 {ssbuffer.block_id = 12 : i32} : index
        %97 = arith.subi %93, %96 {ssbuffer.block_id = 12 : i32} : index
        %extracted_slice_21 = tensor.extract_slice %83[%94, %96] [%95, %97] [1, 1] {ssbuffer.block_id = 12 : i32} : tensor<64x32xbf16> to tensor<?x?xbf16>
        %subview_22 = memref.subview %reinterpret_cast_17[0, 0] [%95, %97] [1, 1] {ssbuffer.block_id = 12 : i32} : memref<64x32xbf16, strided<[128, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice_21 in writable %subview_22 {ssbuffer.block_id = 12 : i32} : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[128, 1], offset: ?>>) -> ()
        %98 = arith.muli %arg16, %c64_i32 {MixUse, ssbuffer.block_id = 12 : i32} : i32
        %99 = arith.maxsi %98, %c0_i32 {ssbuffer.block_id = 12 : i32} : i32
        %100 = arith.index_cast %99 {ssbuffer.block_id = 12 : i32} : i32 to index
        %101 = arith.muli %100, %c4096 {ssbuffer.block_id = 12 : i32} : index
        %102 = arith.addi %101, %18 {ssbuffer.block_id = 12 : i32} : index
        %103 = arith.index_cast %arg9 {ssbuffer.block_id = 12 : i32} : i32 to index
        %104 = arith.subi %c0_i32, %98 {ssbuffer.block_id = 12 : i32} : i32
        %105 = arith.maxsi %104, %c0_i32 {ssbuffer.block_id = 12 : i32} : i32
        %106 = arith.index_cast %105 {ssbuffer.block_id = 12 : i32} : i32 to index
        %107 = arith.addi %102, %62 {ssbuffer.block_id = 12 : i32} : index
        %alloc_23 = memref.alloc() {ssbuffer.block_id = 12 : i32} : memref<64x32xbf16>
        %108 = arith.subi %107, %18 {ssbuffer.block_id = 12 : i32} : index
        %109 = arith.divsi %108, %c4096 {ssbuffer.block_id = 12 : i32} : index
        %110 = arith.subi %103, %109 {ssbuffer.block_id = 12 : i32} : index
        %111 = arith.maxsi %110, %c0 {ssbuffer.block_id = 12 : i32} : index
        %112 = arith.minsi %111, %c64 {ssbuffer.block_id = 12 : i32} : index
        %113 = arith.remsi %108, %c4096 {ssbuffer.block_id = 12 : i32} : index
        %114 = arith.subi %c128, %113 {ssbuffer.block_id = 12 : i32} : index
        %115 = arith.maxsi %114, %c0 {ssbuffer.block_id = 12 : i32} : index
        %116 = arith.minsi %115, %c32 {ssbuffer.block_id = 12 : i32} : index
        %117 = arith.minsi %106, %112 {ssbuffer.block_id = 12 : i32} : index
        %118 = arith.subi %112, %117 {ssbuffer.block_id = 12 : i32} : index
        %119 = arith.minsi %78, %116 {ssbuffer.block_id = 12 : i32} : index
        %120 = arith.subi %116, %119 {ssbuffer.block_id = 12 : i32} : index
        %121 = arith.cmpi slt, %118, %c64 {ssbuffer.block_id = 12 : i32} : index
        %122 = arith.cmpi slt, %120, %c32 {ssbuffer.block_id = 12 : i32} : index
        %123 = arith.ori %121, %122 {ssbuffer.block_id = 12 : i32} : i1
        %124 = arith.muli %100, %c32 {ssbuffer.block_id = 12 : i32} : index
        %alloc_24 = memref.alloc() {ssbuffer.block_id = 12 : i32} : memref<64xf32>
        %125 = arith.divsi %124, %c32 {ssbuffer.block_id = 12 : i32} : index
        %126 = arith.subi %103, %125 {ssbuffer.block_id = 12 : i32} : index
        %127 = arith.maxsi %126, %c0 {ssbuffer.block_id = 12 : i32} : index
        %128 = arith.minsi %127, %c64 {ssbuffer.block_id = 12 : i32} : index
        %129 = arith.minsi %106, %128 {ssbuffer.block_id = 12 : i32} : index
        %130 = arith.subi %128, %129 {ssbuffer.block_id = 12 : i32} : index
        %131 = arith.cmpi slt, %130, %c64 {ssbuffer.block_id = 12 : i32} : index
        hivm.hir.sync_block_wait {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        hivm.hir.copy ins(%reshape_14 : tensor<2x4x16x16xbf16>) outs(%alloc_3 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        hivm.hir.sync_block_wait {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
        hivm.hir.copy ins(%reshape_20 : tensor<2x4x16x16xbf16>) outs(%alloc_5 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 3 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
        scf.if %131 {
          linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst_1 : f32) outs(%alloc_24 : memref<64xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 19 : i32}
        scf.if %123 {
          linalg.fill {ssbuffer.block_id = 10 : i32} ins(%cst_2 : bf16) outs(%alloc_23 : memref<64x32xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 20 : i32}

        // ── Producer: Load 1 (bf16) ──────────────────────────────────
        // R2: First producer slot for bf16 load
        // CHECK: scf.if {{.*}} -> (i1, index) {
        // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 13
        // CHECK: } {ssbuffer.block_id = 13 : i32, ssbuffer.load_store}

        // R2: Second producer slot for bf16 load
        // CHECK: scf.if {{.*}} -> (i1, index) {
        // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 13
        // CHECK: } {ssbuffer.block_id = 13 : i32, ssbuffer.load_store}

        // R2: Load 1 consumer select (bf16)
        // CHECK: arith.remui {{.*}} {ssbuffer.block_id = 13
        // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 13
        // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 13
        // CHECK: arith.select {{.*}} {ssbuffer.block_id = 13

        // R2: Load 1 flag clear (now before compute)
        // CHECK: scf.if {{.*}} -> (i1) {
        // CHECK: } {ssbuffer.block_id = 13 : i32, ssbuffer.load_store}
        // CHECK: scf.if {{.*}} -> (i1) {
        // CHECK: } {ssbuffer.block_id = 13 : i32, ssbuffer.load_store}

        // ── Producer: Load 2 (f32) ───────────────────────────────────
        // R3: First producer slot for f32 load
        // CHECK: scf.if {{.*}} -> (i1, index) {
        // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 13
        // CHECK: } {ssbuffer.block_id = 13 : i32, ssbuffer.load_store}

        // R3: Second producer slot for f32 load
        // CHECK: scf.if {{.*}} -> (i1, index) {
        // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 13
        // CHECK: } {ssbuffer.block_id = 13 : i32, ssbuffer.load_store}

        // R3: Load 2 consumer select (f32)
        // CHECK: arith.remui {{.*}} {ssbuffer.block_id = 13
        // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 13
        // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 13
        // CHECK: arith.select {{.*}} {ssbuffer.block_id = 13

        // R3: Load 2 flag clear (now before compute)
        // CHECK: scf.if {{.*}} -> (i1) {
        // CHECK: } {ssbuffer.block_id = 13 : i32, ssbuffer.load_store}
        // CHECK: scf.if {{.*}} -> (i1) {
        // CHECK: } {ssbuffer.block_id = 13 : i32, ssbuffer.load_store}

        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 8
        %memspacecast = memref.memory_space_cast %alloc_9 {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
        %132 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x32xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
        %memspacecast_25 = memref.memory_space_cast %alloc_7 {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
        %133 = bufferization.to_tensor %memspacecast_25 restrict writable {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x32xf32>
        %134 = arith.addf %133, %1 {ssbuffer.block_id = 13 : i32} : tensor<64x32xf32>
        %135 = arith.addf %132, %134 {ssbuffer.block_id = 13 : i32} : tensor<64x32xf32>
        %reinterpret_cast_26 = memref.reinterpret_cast %arg3 to offset: [%107], sizes: [64, 32], strides: [4096, 1] {ssbuffer.block_id = 13 : i32} : memref<?xbf16> to memref<64x32xbf16, strided<[4096, 1], offset: ?>>
        %subview_27 = memref.subview %reinterpret_cast_26[0, 0] [%118, %120] [1, 1] {ssbuffer.block_id = 13 : i32} : memref<64x32xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
        %subview_28 = memref.subview %alloc_23[%117, %119] [%118, %120] [1, 1] {ssbuffer.block_id = 13 : i32} : memref<64x32xbf16> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
        memref.copy %subview_27, %subview_28 {ssbuffer.block_id = 13 : i32} : memref<?x?xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
        %136 = bufferization.to_tensor %alloc_23 restrict writable {gm_load_bufferable, ssbuffer.block_id = 13 : i32} : memref<64x32xbf16>
        %137 = arith.extf %136 {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64x32xbf16> to tensor<64x32xf32>
        %138 = arith.subf %137, %135 {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64x32xf32>
        %reinterpret_cast_29 = memref.reinterpret_cast %arg5 to offset: [%107], sizes: [64, 32], strides: [4096, 1] {ssbuffer.block_id = 13 : i32} : memref<?xbf16> to memref<64x32xbf16, strided<[4096, 1], offset: ?>>
        %139 = arith.truncf %138 {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64x32xf32> to tensor<64x32xbf16>
        %extracted_slice_30 = tensor.extract_slice %139[%117, %119] [%118, %120] [1, 1] {ssbuffer.block_id = 13 : i32} : tensor<64x32xbf16> to tensor<?x?xbf16>
        %subview_31 = memref.subview %reinterpret_cast_29[0, 0] [%118, %120] [1, 1] {ssbuffer.block_id = 13 : i32} : memref<64x32xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice_30 in writable %subview_31 {ssbuffer.block_id = 13 : i32} : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[4096, 1], offset: ?>>) -> ()
        %140 = arith.addi %arg16, %c1_i32 {ssbuffer.block_id = 13 : i32} : i32
        %141 = arith.muli %140, %c64_i32 {ssbuffer.block_id = 13 : i32} : i32
        %142 = arith.minsi %141, %arg9 {ssbuffer.block_id = 13 : i32} : i32
        %143 = arith.subi %142, %c1_i32 {ssbuffer.block_id = 13 : i32} : i32
        %144 = arith.muli %143, %c32_i32 {ssbuffer.block_id = 13 : i32} : i32
        %145 = arith.addi %14, %144 {ssbuffer.block_id = 13 : i32} : i32
        %146 = arith.addi %145, %5 {ssbuffer.block_id = 13 : i32} : i32
        %147 = arith.index_cast %146 {ssbuffer.block_id = 13 : i32} : i32 to index
        %reinterpret_cast_32 = memref.reinterpret_cast %arg6 to offset: [%147], sizes: [1], strides: [1] {ssbuffer.block_id = 13 : i32} : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
        %148 = memref.load %reinterpret_cast_32[%c0] {ssbuffer.block_id = 13 : i32} : memref<1xf32, strided<[1], offset: ?>>
        %149 = arith.addi %124, %20 {ssbuffer.block_id = 13 : i32} : index
        %reinterpret_cast_33 = memref.reinterpret_cast %arg6 to offset: [%149], sizes: [64], strides: [32] {ssbuffer.block_id = 13 : i32} : memref<?xf32> to memref<64xf32, strided<[32], offset: ?>>
        %subview_34 = memref.subview %reinterpret_cast_33[0] [%130] [1] {ssbuffer.block_id = 13 : i32} : memref<64xf32, strided<[32], offset: ?>> to memref<?xf32, strided<[32], offset: ?>>
        %subview_35 = memref.subview %alloc_24[%129] [%130] [1] {ssbuffer.block_id = 13 : i32} : memref<64xf32> to memref<?xf32, strided<[1], offset: ?>>
        memref.copy %subview_34, %subview_35 {ssbuffer.block_id = 13 : i32} : memref<?xf32, strided<[32], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %150 = bufferization.to_tensor %alloc_24 restrict writable {gm_load_bufferable, ssbuffer.block_id = 13 : i32} : memref<64xf32>
        %151 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%148 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
        %152 = arith.subf %151, %150 {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64xf32>
        %153 = math.exp2 %152 {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64xf32>
        %154 = arith.index_cast %98 {DataUse, ssbuffer.block_id = 13 : i32} : i32 to index
        %155 = arith.addi %154, %c64 {DataUse, ssbuffer.block_id = 13 : i32} : index
        %156 = arith.index_cast %arg9 {DataUse, ssbuffer.block_id = 13 : i32} : i32 to index
        %157 = arith.maxsi %154, %156 {DataUse, ssbuffer.block_id = 13 : i32} : index
        %158 = arith.minsi %155, %157 {DataUse, ssbuffer.block_id = 13 : i32} : index
        %159 = arith.subi %158, %154 {DataUse, ssbuffer.block_id = 13 : i32} : index
        %extracted_slice_36 = tensor.extract_slice %153[0] [%159] [1] {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64xf32> to tensor<?xf32>
        %inserted_slice = tensor.insert_slice %extracted_slice_36 into %3[0] [%159] [1] {DataUse, ssbuffer.block_id = 13 : i32} : tensor<?xf32> into tensor<64xf32>
        %broadcasted = linalg.broadcast ins(%inserted_slice : tensor<64xf32>) outs(%0 : tensor<64x32xf32>) dimensions = [1]  {ssbuffer.block_id = 13 : i32}
        %160 = arith.mulf %138, %broadcasted {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64x32xf32>
        %161 = arith.truncf %160 {DataUse, ssbuffer.block_id = 13 : i32} : tensor<64x32xf32> to tensor<64x32xbf16>
        %reshape_37 = tensor.reshape %161(%cst_0) {ssbuffer.block_id = 13 : i32} : (tensor<64x32xbf16>, tensor<3xi64>) -> tensor<64x2x16xbf16>
        %162 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<2x64x16xbf16>
        %transposed_38 = linalg.transpose ins(%reshape_37 : tensor<64x2x16xbf16>) outs(%162 : tensor<2x64x16xbf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 13 : i32}
        %reshape_39 = tensor.reshape %transposed_38(%cst) {ssbuffer.block_id = 13 : i32} : (tensor<2x64x16xbf16>, tensor<4xi64>) -> tensor<2x4x16x16xbf16>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.copy ins(%reshape_39 : tensor<2x4x16x16xbf16>) outs(%alloc : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
        hivm.hir.copy ins(%reshape_39 : tensor<2x4x16x16xbf16>) outs(%alloc_4 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8

        // R6: Block 14 compute
        // CHECK: hivm.hir.sync_block_wait {{.*ssbuffer.block_id = 14}}
        // CHECK: arith.mulf {{.*ssbuffer.block_id = 14}}
        // CHECK: arith.addf {{.*ssbuffer.block_id = 14}}

        // R7: scf.yield with 10 values
        // CHECK: scf.yield {{.*}} : tensor<64x32xf32>, tensor<64x32xf32>, i1, i1, index, index, i1, i1, index, index

        hivm.hir.sync_block_wait {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
        %memspacecast_40 = memref.memory_space_cast %alloc_8 {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
        %163 = bufferization.to_tensor %memspacecast_40 restrict writable {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x32xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 5
        %memspacecast_41 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
        %164 = bufferization.to_tensor %memspacecast_41 restrict writable {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x32xf32>
        %165 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<1xf32>
        %inserted = tensor.insert %148 into %165[%c0] {ssbuffer.block_id = 14 : i32} : tensor<1xf32>
        %166 = math.exp2 %inserted {DataUse, ssbuffer.block_id = 14 : i32} : tensor<1xf32>
        %extracted = tensor.extract %166[%c0] {DataUse, ssbuffer.block_id = 14 : i32} : tensor<1xf32>
        %167 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%extracted : f32) outs(%0 : tensor<64x32xf32>) -> tensor<64x32xf32>
        %168 = arith.mulf %arg17, %167 {DataUse, ssbuffer.block_id = 14 : i32} : tensor<64x32xf32>
        %169 = arith.mulf %arg18, %167 {DataUse, ssbuffer.block_id = 14 : i32} : tensor<64x32xf32>
        %170 = arith.addf %163, %168 {ssbuffer.block_id = 14 : i32} : tensor<64x32xf32>
        %171 = arith.addf %164, %169 {ssbuffer.block_id = 14 : i32} : tensor<64x32xf32>
        hivm.hir.sync_block_set {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
        hivm.hir.sync_block_set {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
        scf.yield %170, %171 : tensor<64x32xf32>, tensor<64x32xf32>
      } {DataUse, ssbuffer.block_id = 23 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
      hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
      hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
      hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      %22 = arith.muli %arg14, %c16384_i32 {ssbuffer.block_id = 16 : i32} : i32
      %23 = arith.index_cast %22 {ssbuffer.block_id = 16 : i32} : i32 to index
      %24 = arith.maxsi %19, %c0_i32 {ssbuffer.block_id = 16 : i32} : i32
      %25 = arith.index_cast %24 {ssbuffer.block_id = 16 : i32} : i32 to index
      %26 = arith.addi %23, %25 {ssbuffer.block_id = 16 : i32} : index
      %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%26], sizes: [64, 32], strides: [128, 1] {ssbuffer.block_id = 16 : i32} : memref<?xf32> to memref<64x32xf32, strided<[128, 1], offset: ?>>
      %27 = arith.divsi %25, %c128 {ssbuffer.block_id = 16 : i32} : index
      %28 = arith.subi %c128, %27 {ssbuffer.block_id = 16 : i32} : index
      %29 = arith.maxsi %28, %c0 {ssbuffer.block_id = 16 : i32} : index
      %30 = arith.minsi %29, %c64 {ssbuffer.block_id = 16 : i32} : index
      %31 = arith.remsi %25, %c128 {ssbuffer.block_id = 16 : i32} : index
      %32 = arith.subi %c128, %31 {ssbuffer.block_id = 16 : i32} : index
      %33 = arith.maxsi %32, %c0 {ssbuffer.block_id = 16 : i32} : index
      %34 = arith.minsi %33, %c32 {ssbuffer.block_id = 16 : i32} : index
      %35 = arith.minsi %30, %c0 {ssbuffer.block_id = 16 : i32} : index
      %36 = arith.subi %30, %35 {ssbuffer.block_id = 16 : i32} : index
      %37 = arith.subi %c0_i32, %19 {ssbuffer.block_id = 16 : i32} : i32
      %38 = arith.maxsi %37, %c0_i32 {ssbuffer.block_id = 16 : i32} : i32
      %39 = arith.index_cast %38 {ssbuffer.block_id = 16 : i32} : i32 to index
      %40 = arith.minsi %39, %34 {ssbuffer.block_id = 16 : i32} : index
      %41 = arith.subi %34, %40 {ssbuffer.block_id = 16 : i32} : index
      %extracted_slice = tensor.extract_slice %21#0[%35, %40] [%36, %41] [1, 1] {ssbuffer.block_id = 16 : i32} : tensor<64x32xf32> to tensor<?x?xf32>
      %subview = memref.subview %reinterpret_cast[0, 0] [%36, %41] [1, 1] {ssbuffer.block_id = 16 : i32} : memref<64x32xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview {ssbuffer.block_id = 16 : i32} : (tensor<?x?xf32>, memref<?x?xf32, strided<[128, 1], offset: ?>>) -> ()
      %42 = arith.addi %23, %c8192 {ssbuffer.block_id = 16 : i32} : index
      %43 = arith.addi %42, %25 {ssbuffer.block_id = 16 : i32} : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg8 to offset: [%43], sizes: [64, 32], strides: [128, 1] {ssbuffer.block_id = 16 : i32} : memref<?xf32> to memref<64x32xf32, strided<[128, 1], offset: ?>>
      %44 = arith.subi %43, %23 {ssbuffer.block_id = 16 : i32} : index
      %45 = arith.divsi %44, %c128 {ssbuffer.block_id = 16 : i32} : index
      %46 = arith.subi %c128, %45 {ssbuffer.block_id = 16 : i32} : index
      %47 = arith.maxsi %46, %c0 {ssbuffer.block_id = 16 : i32} : index
      %48 = arith.minsi %47, %c64 {ssbuffer.block_id = 16 : i32} : index
      %49 = arith.remsi %44, %c128 {ssbuffer.block_id = 16 : i32} : index
      %50 = arith.subi %c128, %49 {ssbuffer.block_id = 16 : i32} : index
      %51 = arith.maxsi %50, %c0 {ssbuffer.block_id = 16 : i32} : index
      %52 = arith.minsi %51, %c32 {ssbuffer.block_id = 16 : i32} : index
      %53 = arith.minsi %48, %c0 {ssbuffer.block_id = 16 : i32} : index
      %54 = arith.subi %48, %53 {ssbuffer.block_id = 16 : i32} : index
      %55 = arith.minsi %39, %52 {ssbuffer.block_id = 16 : i32} : index
      %56 = arith.subi %52, %55 {ssbuffer.block_id = 16 : i32} : index
      %extracted_slice_11 = tensor.extract_slice %21#1[%53, %55] [%54, %56] [1, 1] {ssbuffer.block_id = 16 : i32} : tensor<64x32xf32> to tensor<?x?xf32>
      %subview_12 = memref.subview %reinterpret_cast_10[0, 0] [%54, %56] [1, 1] {ssbuffer.block_id = 16 : i32} : memref<64x32xf32, strided<[128, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_11 in writable %subview_12 {ssbuffer.block_id = 16 : i32} : (tensor<?x?xf32>, memref<?x?xf32, strided<[128, 1], offset: ?>>) -> ()
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// ============================================================================
// Test Group B3: attn_fwd - mixed VECTOR + CUBE scopes, 3 gm_load_bufferable
//
// Scope 1 (VECTOR): No gm_load_bufferable - should remain unchanged.
// Scope 2 (CUBE): 3 marked loads across outer+inner loops:
//   - Load A (outer loop, block_id=2): memref<128x128xf16> → double-buffered
//   - Load B (inner loop, block_id=0): memref<128x128xf16> → double-buffered
//   - Load C (inner loop, block_id=1): memref<128x128xf16> → double-buffered
//
// Outer loop gets 4 control iter_args (for Load A).
// Inner loop gets 8 control iter_args (4 for Load B + 4 for Load C).
// ============================================================================

// CHECK-LABEL: func.func @_attn_fwd
func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 5 : i32} dense<[8, 8, 16, 16]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 5 : i32} dense<[128, 8, 16]> : tensor<3xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 10 : i32} 0.000000e+00 : f32
    %c28_i32 = arith.constant {ssbuffer.block_id = 10 : i32} 28 : i32
    %c65536_i32 = arith.constant {ssbuffer.block_id = 10 : i32} 65536 : i32
    %c8192_i32 = arith.constant {ssbuffer.block_id = 10 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 10 : i32} 128 : i32
    %c1048576_i64 = arith.constant {ssbuffer.block_id = 10 : i32} 1048576 : i64
    %c8388608_i64 = arith.constant {ssbuffer.block_id = 10 : i32} 8388608 : i64
    %c8_i32 = arith.constant {ssbuffer.block_id = 10 : i32} 8 : i32
    %c64_i32 = arith.constant {ssbuffer.block_id = 10 : i32} 64 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 10 : i32} 0 : i32
    %cst_2 = arith.constant {ssbuffer.block_id = 10 : i32} 5.000000e-01 : f32
    %cst_3 = arith.constant {ssbuffer.block_id = 10 : i32} 0xFF800000 : f32
    %cst_4 = arith.constant {ssbuffer.block_id = 10 : i32} 1.000000e+00 : f32
    %c128 = arith.constant {ssbuffer.block_id = 10 : i32} 128 : index

    // ── Scope 1 (VECTOR): no gm_load_bufferable → unchanged ─────────

    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 10 : i32} : tensor<128x128xf32>
      %1 = linalg.fill {ssbuffer.block_id = 10 : i32} ins(%cst_1 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %2 = linalg.fill {ssbuffer.block_id = 10 : i32} ins(%cst_2 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %3 = tensor.empty() {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
      %4 = linalg.fill {ssbuffer.block_id = 10 : i32} ins(%cst_3 : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
      %5 = linalg.fill {ssbuffer.block_id = 10 : i32} ins(%cst_4 : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
      scf.for %arg16 = %arg13 to %c65536_i32 step %c28_i32  : i32 {
        %alloc = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        %alloc_6 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        %6:5 = scf.for %arg17 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg18 = %1, %arg19 = %5, %arg20 = %4, %arg21 = %c0_i32, %arg22 = %c0_i32) -> (tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32)  : i32 {
          hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
          %memspacecast = memref.memory_space_cast %alloc_5 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %30 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32>
          %31 = arith.mulf %30, %2 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %reduced = linalg.reduce ins(%31 : tensor<128x128xf32>) outs(%4 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
            (%in: f32, %init: f32) {
              %47 = arith.maximumf %in, %init : f32
              linalg.yield %47 : f32
            }
          %32 = arith.maximumf %arg20, %reduced {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
          %broadcasted_8 = linalg.broadcast ins(%32 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
          %33 = arith.subf %31, %broadcasted_8 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %34 = math.exp %33 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %35 = arith.truncf %34 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
          %reshape = tensor.reshape %35(%cst_0) {ssbuffer.block_id = 5 : i32} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
          annotation.mark %reshape {ssbuffer.block_id = 5 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<128x8x16xf16>
          %36 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x128x16xf16>
          %transposed = linalg.transpose ins(%reshape : tensor<128x8x16xf16>) outs(%36 : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 5 : i32}
          %reshape_9 = tensor.reshape %transposed(%cst) {ssbuffer.block_id = 5 : i32} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
          annotation.mark %reshape_9 {ssbuffer.block_id = 5 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x8x16x16xf16>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.copy ins(%reshape_9 : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
          hivm.hir.sync_block_wait {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %memspacecast_10 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %37 = bufferization.to_tensor %memspacecast_10 restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32>
          %38 = linalg.fill {ssbuffer.block_id = 7 : i32} ins(%cst_1 : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
          %reduced_11 = linalg.reduce ins(%34 : tensor<128x128xf32>) outs(%38 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 7 : i32}
            (%in: f32, %init: f32) {
              %47 = arith.addf %in, %init : f32
              linalg.yield %47 : f32
            }
          %39 = arith.subf %arg20, %32 {ssbuffer.block_id = 7 : i32} : tensor<128xf32>
          %40 = math.exp %39 {ssbuffer.block_id = 7 : i32} : tensor<128xf32>
          %41 = arith.mulf %arg19, %40 {ssbuffer.block_id = 7 : i32} : tensor<128xf32>
          %42 = arith.addf %41, %reduced_11 {ssbuffer.block_id = 7 : i32} : tensor<128xf32>
          %broadcasted_12 = linalg.broadcast ins(%40 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 7 : i32}
          %43 = arith.mulf %arg18, %broadcasted_12 {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32>
          %44 = arith.addf %37, %43 {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32>
          %45 = arith.addi %arg21, %c128_i32 {ssbuffer.block_id = 7 : i32} : i32
          %46 = arith.addi %arg22, %c128_i32 {ssbuffer.block_id = 7 : i32} : i32
          hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
          scf.yield %44, %42, %32, %45, %46 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32
        } {ssbuffer.block_id = 11 : i32, ssbuffer.main_loop = 0 : i32}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 9 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 4
        %7 = arith.divsi %arg16, %c64_i32 {ssbuffer.block_id = 9 : i32} : i32
        %8 = arith.remsi %arg16, %c64_i32 {ssbuffer.block_id = 9 : i32} : i32
        %9 = arith.divsi %7, %c8_i32 {ssbuffer.block_id = 9 : i32} : i32
        %10 = arith.remsi %7, %c8_i32 {ssbuffer.block_id = 9 : i32} : i32
        %11 = arith.extsi %9 {ssbuffer.block_id = 9 : i32} : i32 to i64
        %12 = arith.muli %11, %c8388608_i64 {ssbuffer.block_id = 9 : i32} : i64
        %13 = arith.extsi %10 {ssbuffer.block_id = 9 : i32} : i32 to i64
        %14 = arith.muli %13, %c1048576_i64 {ssbuffer.block_id = 9 : i32} : i64
        %15 = arith.addi %12, %14 {ssbuffer.block_id = 9 : i32} : i64
        %16 = arith.index_cast %15 {ssbuffer.block_id = 9 : i32} : i64 to index
        %17 = arith.muli %8, %c128_i32 {ssbuffer.block_id = 9 : i32} : i32
        %18 = arith.maxsi %17, %c0_i32 {ssbuffer.block_id = 9 : i32} : i32
        %19 = arith.index_cast %18 {ssbuffer.block_id = 9 : i32} : i32 to index
        %20 = arith.muli %19, %c128 {ssbuffer.block_id = 9 : i32} : index
        %21 = arith.addi %20, %16 {ssbuffer.block_id = 9 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%21], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 9 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %22 = math.log %6#1 {ssbuffer.block_id = 9 : i32} : tensor<128xf32>
        %23 = arith.addf %6#2, %22 {ssbuffer.block_id = 9 : i32} : tensor<128xf32>
        %broadcasted = linalg.broadcast ins(%6#1 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 9 : i32}
        %24 = arith.divf %6#0, %broadcasted {ssbuffer.block_id = 9 : i32} : tensor<128x128xf32>
        %25 = arith.muli %7, %c8192_i32 {ssbuffer.block_id = 9 : i32} : i32
        %26 = arith.index_cast %25 {ssbuffer.block_id = 9 : i32} : i32 to index
        %27 = arith.index_cast %17 {ssbuffer.block_id = 9 : i32} : i32 to index
        %28 = arith.addi %26, %27 {ssbuffer.block_id = 9 : i32} : index
        %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%28], sizes: [128], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
        bufferization.materialize_in_destination %23 in writable %reinterpret_cast_7 {ssbuffer.block_id = 9 : i32} : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
        %29 = arith.truncf %24 {ssbuffer.block_id = 9 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
        bufferization.materialize_in_destination %29 in writable %reinterpret_cast {ssbuffer.block_id = 9 : i32} : (tensor<128x128xf16>, memref<128x128xf16, strided<[128, 1], offset: ?>>) -> ()
      } {ssbuffer.block_id = 12 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    // ── Scope 2 (CUBE): 3 gm_load_bufferable → multi-buffered ───────
    // R1: Load A (outer loop, block_id=2) gets 2 buffer slots
    // CHECK: memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<128x128xf16>
    // CHECK: memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<128x128xf16>

    // R1: Outer loop gets 4 control iter_args (for Load A)
    // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (i1, i1, index, index)

    // R2: Load A producer + consumer
    // CHECK: scf.if {{.*}} -> (i1, index) {
    // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 2
    // CHECK: } {ssbuffer.block_id = 2 : i32, ssbuffer.load_store}
    // CHECK: scf.if {{.*}} -> (i1, index) {
    // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 2
    // CHECK: } {ssbuffer.block_id = 2 : i32, ssbuffer.load_store}

    // R2: Load A consumer select
    // CHECK: arith.remui {{.*}} {ssbuffer.block_id = 2
    // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 2
    // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 2
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 2

    // R2: Load A flag clear
    // CHECK: scf.if {{.*}} -> (i1) {
    // CHECK: } {ssbuffer.block_id = 2 : i32, ssbuffer.load_store}
    // CHECK: scf.if {{.*}} -> (i1) {
    // CHECK: } {ssbuffer.block_id = 2 : i32, ssbuffer.load_store}

    // R3: Loads B & C (inner loop, block_id=0 and block_id=1) each get 2 slots
    // CHECK: memref.alloc() {ssbuffer.block_id = 0 : i32} : memref<128x128xf16>
    // CHECK: memref.alloc() {ssbuffer.block_id = 0 : i32} : memref<128x128xf16>
    // CHECK: memref.alloc() {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
    // CHECK: memref.alloc() {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>

    // R3: Inner loop gets 8 control iter_args (4 for Load B + 4 for Load C)
    // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (i1, i1, index, index, i1, i1, index, index)

    // R4: Load B (block_id=0) producer + consumer
    // CHECK: scf.if {{.*}} -> (i1, index) {
    // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 0
    // CHECK: } {ssbuffer.block_id = 0 : i32, ssbuffer.load_store}
    // CHECK: arith.remui {{.*}} {ssbuffer.block_id = 0
    // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 0
    // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 0
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 0

    // R5: Load C (block_id=1) producer + consumer
    // CHECK: scf.if {{.*}} -> (i1, index) {
    // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 1
    // CHECK: } {ssbuffer.block_id = 1 : i32, ssbuffer.load_store}
    // CHECK: arith.remui {{.*}} {ssbuffer.block_id = 1
    // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 1
    // CHECK: bufferization.to_tensor {{.*}} {ssbuffer.block_id = 1
    // CHECK: arith.select {{.*}} {ssbuffer.block_id = 1

    // R6: Inner loop yield with 8 values
    // CHECK: scf.yield {{.*}} : i1, i1, index, index, i1, i1, index, index

    // R7: Outer loop yield with 4 values
    // CHECK: scf.yield {{.*}} : i1, i1, index, index

    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<128x128xf32>
      %1 = linalg.fill {ssbuffer.block_id = 3 : i32} ins(%cst_1 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.for %arg16 = %arg13 to %c65536_i32 step %c28_i32  : i32 {
        %2 = arith.divsi %arg16, %c64_i32 {ssbuffer.block_id = 2 : i32} : i32
        %3 = arith.remsi %arg16, %c64_i32 {ssbuffer.block_id = 2 : i32} : i32
        %4 = arith.divsi %2, %c8_i32 {ssbuffer.block_id = 2 : i32} : i32
        %5 = arith.remsi %2, %c8_i32 {ssbuffer.block_id = 2 : i32} : i32
        %6 = arith.muli %5, %arg9 {ssbuffer.block_id = 2 : i32} : i32
        %7 = arith.divsi %6, %c8_i32 {ssbuffer.block_id = 2 : i32} : i32
        %8 = arith.extsi %4 {ssbuffer.block_id = 2 : i32} : i32 to i64
        %9 = arith.muli %8, %c8388608_i64 {ssbuffer.block_id = 2 : i32} : i64
        %10 = arith.extsi %5 {ssbuffer.block_id = 2 : i32} : i32 to i64
        %11 = arith.muli %10, %c1048576_i64 {ssbuffer.block_id = 2 : i32} : i64
        %12 = arith.addi %9, %11 {ssbuffer.block_id = 2 : i32} : i64
        %13 = arith.extsi %7 {ssbuffer.block_id = 2 : i32} : i32 to i64
        %14 = arith.muli %13, %c1048576_i64 {ssbuffer.block_id = 2 : i32} : i64
        %15 = arith.addi %9, %14 {ssbuffer.block_id = 2 : i32} : i64
        %16 = arith.index_cast %12 {ssbuffer.block_id = 2 : i32} : i64 to index
        %17 = arith.muli %3, %c128_i32 {ssbuffer.block_id = 2 : i32} : i32
        %18 = arith.maxsi %17, %c0_i32 {ssbuffer.block_id = 2 : i32} : i32
        %19 = arith.index_cast %18 {ssbuffer.block_id = 2 : i32} : i32 to index
        %20 = arith.muli %19, %c128 {ssbuffer.block_id = 2 : i32} : index
        %21 = arith.addi %20, %16 {ssbuffer.block_id = 2 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 2 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %22 = arith.index_cast %15 {ssbuffer.block_id = 2 : i32} : i64 to index
        %alloc = memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<128x128xf16>
        memref.copy %reinterpret_cast, %alloc {ssbuffer.block_id = 2 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
        %23 = bufferization.to_tensor %alloc restrict writable {gm_load_bufferable, ssbuffer.block_id = 2 : i32} : memref<128x128xf16>
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        %alloc_6 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        %alloc_7 = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        %24:2 = scf.for %arg17 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg18 = %c0_i32, %arg19 = %c0_i32) -> (i32, i32)  : i32 {
          %25 = arith.index_cast %arg18 {ssbuffer.block_id = 0 : i32} : i32 to index
          %26 = arith.muli %25, %c128 {ssbuffer.block_id = 0 : i32} : index
          %27 = arith.addi %26, %22 {ssbuffer.block_id = 0 : i32} : index
          %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 0 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %alloc_9 = memref.alloc() {ssbuffer.block_id = 0 : i32} : memref<128x128xf16>
          memref.copy %reinterpret_cast_8, %alloc_9 {ssbuffer.block_id = 0 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
          %28 = bufferization.to_tensor %alloc_9 restrict writable {gm_load_bufferable, ssbuffer.block_id = 0 : i32} : memref<128x128xf16>
          %29 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf16>
          %transposed = linalg.transpose ins(%28 : tensor<128x128xf16>) outs(%29 : tensor<128x128xf16>) permutation = [1, 0]  {ssbuffer.block_id = 0 : i32}
          %30 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 0 : i32} ins(%23, %transposed : tensor<128x128xf16>, tensor<128x128xf16>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32} ins(%30 : tensor<128x128xf32>) outs(%alloc_6 : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
          hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          %31 = hivm.hir.convert_layout %alloc_5 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %31 {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16>
          %32 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x128xf16>
          %33 = arith.index_cast %arg19 {ssbuffer.block_id = 1 : i32} : i32 to index
          %34 = arith.muli %33, %c128 {ssbuffer.block_id = 1 : i32} : index
          %35 = arith.addi %34, %22 {ssbuffer.block_id = 1 : i32} : index
          %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 1 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %alloc_11 = memref.alloc() {ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
          memref.copy %reinterpret_cast_10, %alloc_11 {ssbuffer.block_id = 1 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
          %36 = bufferization.to_tensor %alloc_11 restrict writable {gm_load_bufferable, ssbuffer.block_id = 1 : i32} : memref<128x128xf16>
          %37 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf32>
          %38 = linalg.fill {ssbuffer.block_id = 1 : i32} ins(%cst_1 : f32) outs(%37 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %39 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32} ins(%32, %36 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%38 : tensor<128x128xf32>) -> tensor<128x128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32} ins(%39 : tensor<128x128xf32>) outs(%alloc_7 : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %40 = arith.addi %arg18, %c128_i32 {ssbuffer.block_id = 7 : i32} : i32
          %41 = arith.addi %arg19, %c128_i32 {ssbuffer.block_id = 7 : i32} : i32
          scf.yield %40, %41 : i32, i32
        } {ssbuffer.block_id = 11 : i32, ssbuffer.main_loop = 0 : i32}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
        hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 4
      } {ssbuffer.block_id = 12 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
