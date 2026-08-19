// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" | FileCheck %s
//
// Size-variant regression for the FA CV split path.
// BM=64 BN=128 N=4096 unroll-factor=4 pass output.

// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock, ssbuffer.inter_core_buf_count = 2 : i32, triton_ascend.cv_split_scheduling.applied = 1 : i32} {
// CHECK-LABEL: func.func @_attn_fwd

// Invariant accumulator templates are materialized once, outside the physical
// program loop, instead of being rebuilt by every program.
// CHECK: %{{.*}} = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<64x128xf32>) -> tensor<64x128xf32>
// CHECK: %{{.*}} = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<32x128xf32>) -> tensor<32x128xf32>
// CHECK: %{{.*}} = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %{{.*}} = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK: scf.for

// CHECK: %reinterpret_cast = memref.reinterpret_cast %arg2
// CHECK-NEXT: %{{.*}} = memref.alloc() : memref<64x64xf16>
// CHECK-NEXT: memref.copy %{{.*}}, %{{.*}} : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
// CHECK-NEXT: %{{.*}} = bufferization.to_tensor %{{.*}} restrict writable : memref<64x64xf16>
// CHECK-NEXT: %{{.*}} = arith.constant 512 : i32
// The two physical GM addresses are loop-carried below.  Their initial values
// are formed here, before the persistent UB/CBUF allocations.
// The two CUBE->VECTOR pools keep the inter-core buffer count of two: the
// score slots, then the L1 slots the P tiles pack into, then the product
// slots.  The VECTOR->CUBE pool gets one slot per lane instead -- it is in
// L1, so the extra slots are free against the UB budget, and with nothing
// reused there the schedule has nothing to pipeline against.
// CHECK: %{{.*}} = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
// CHECK-NEXT: annotation.mark %{{.*}} {effects = ["write", "read"]} : memref<32x128xf32, #hivm.address_space<ub>>
// CHECK-NEXT: %{{.*}} = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
// CHECK-NEXT: annotation.mark %{{.*}} {effects = ["write", "read"]} : memref<32x128xf32, #hivm.address_space<ub>>
// CHECK-NEXT: %{{.*}} = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: annotation.mark %{{.*}} {effects = ["write", "read"]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: %{{.*}} = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: annotation.mark %{{.*}} {effects = ["write", "read"]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: %{{.*}} = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: annotation.mark %{{.*}} {effects = ["write", "read"]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: %{{.*}} = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: annotation.mark %{{.*}} {effects = ["write", "read"]} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: %{{.*}} = memref.alloc() : memref<32x64xf32, #hivm.address_space<ub>>
// CHECK-NEXT: annotation.mark %{{.*}} {effects = ["write", "read"]} : memref<32x64xf32, #hivm.address_space<ub>>
// CHECK-NEXT: %{{.*}} = memref.alloc() : memref<32x64xf32, #hivm.address_space<ub>>
// CHECK-NEXT: annotation.mark %{{.*}} {effects = ["write", "read"]} : memref<32x64xf32, #hivm.address_space<ub>>

// CHECK: scope.scope : () -> () {
// One ND view per physical L1 slot, and one slot per lane.
// CHECK-NEXT: %{{.*}} = hivm.hir.convert_layout %{{.*}} output_shape [64, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) -> memref<64x128xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: %{{.*}} = hivm.hir.convert_layout %{{.*}} output_shape [64, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) -> memref<64x128xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: %{{.*}} = hivm.hir.convert_layout %{{.*}} output_shape [64, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) -> memref<64x128xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: %{{.*}} = hivm.hir.convert_layout %{{.*}} output_shape [64, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) -> memref<64x128xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: %{{.*}} = memref.memory_space_cast %{{.*}} : memref<64x128xf16, #hivm.address_space<cbuf>> to memref<64x128xf16>
// CHECK-NEXT: %{{.*}} = memref.memory_space_cast %{{.*}} : memref<64x128xf16, #hivm.address_space<cbuf>> to memref<64x128xf16>
// CHECK-NEXT: %{{.*}} = memref.memory_space_cast %{{.*}} : memref<64x128xf16, #hivm.address_space<cbuf>> to memref<64x128xf16>
// CHECK-NEXT: %{{.*}} = memref.memory_space_cast %{{.*}} : memref<64x128xf16, #hivm.address_space<cbuf>> to memref<64x128xf16>
// CHECK-NEXT: %{{.*}}:4 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (i32, i32, index, index) : i32 {

// The two CUBE->VECTOR pools still rotate over two slots, and the schedule
// orders that reuse itself rather than with a release flag: consuming work is
// pipelined behind by the VECTOR->CUBE pool's slot count, which leaves an
// existing cross-core handoff between a slot's read and the write reusing it.
// With one slot per lane there, that distance is the lane count, so nothing is
// emitted early and CUBE issues every score matmul before its first wait.
// See cv_split_scheduling_fa.mlir for the full description.
//
// All four score tiles, back to back.
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode{{<}}nz2nd{{>}}} ins(%{{.*}} : tensor<64x128xf32>) outs(%{{.*}} : memref<32x128xf32, #hivm.address_space<ub>>) dual_dst_mode = {{<}}ROW_SPLIT{{>}}
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK: hivm.hir.fixpipe {{.*}} dual_dst_mode = {{<}}ROW_SPLIT{{>}}
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: hivm.hir.fixpipe {{.*}} dual_dst_mode = {{<}}ROW_SPLIT{{>}}
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK: hivm.hir.fixpipe {{.*}} dual_dst_mode = {{<}}ROW_SPLIT{{>}}
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1

// Then the product tiles, each behind its own lane's P hand-off.
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
// CHECK: hivm.hir.fixpipe {{.*}} ins(%{{.*}} : tensor<64x64xf32>) outs(%{{.*}} : memref<32x64xf32, #hivm.address_space<ub>>) dual_dst_mode = {{<}}ROW_SPLIT{{>}}
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
// CHECK: hivm.hir.fixpipe {{.*}} dual_dst_mode = {{<}}ROW_SPLIT{{>}}
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 7
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
// CHECK: hivm.hir.fixpipe {{.*}} dual_dst_mode = {{<}}ROW_SPLIT{{>}}
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
// CHECK: hivm.hir.fixpipe {{.*}} dual_dst_mode = {{<}}ROW_SPLIT{{>}}
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 7

// CHECK: scope.return
// CHECK-NEXT: } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}

// Reusable output templates remain outside the vector loop.
// CHECK: %{{.*}} = tensor.empty() : tensor<32x128xf32>
// CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<32xf32>
// CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<32x64xf32>

// CHECK: scope.scope : () -> () {
// CHECK-NEXT: %{{.*}} = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT: %{{.*}} = arith.index_cast %{{.*}} : i64 to index
// CHECK-NEXT: %{{.*}}:7 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<32xf32>, tensor<32x64xf32>, tensor<32xf32>, i32, i32, index, index) : i32 {

// Consumer side, a plain two-phase order.  With a slot per lane in the
// VECTOR->CUBE pool there is no reuse to interleave against: all four softmaxes
// run first, each behind its own score tile, then the four accumulations drain.
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK: hivm.hir.copy ins(%{{.*}} : memref<8x2x16x16xf16, #hivm.address_space<ub>>) outs(%{{.*}} : memref<8x2x16x16xf16, strided<{{.*}}, offset: ?>, #hivm.address_space<cbuf>>)
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: hivm.hir.copy ins(%{{.*}} : memref<8x2x16x16xf16, #hivm.address_space<ub>>) outs(%{{.*}} : memref<8x2x16x16xf16, strided<{{.*}}, offset: ?>, #hivm.address_space<cbuf>>)
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK: hivm.hir.copy ins(%{{.*}} : memref<8x2x16x16xf16, #hivm.address_space<ub>>) outs(%{{.*}} : memref<8x2x16x16xf16, strided<{{.*}}, offset: ?>, #hivm.address_space<cbuf>>)
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: hivm.hir.copy ins(%{{.*}} : memref<8x2x16x16xf16, #hivm.address_space<ub>>) outs(%{{.*}} : memref<8x2x16x16xf16, strided<{{.*}}, offset: ?>, #hivm.address_space<cbuf>>)
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5

// Then the four product tiles, over the two PV slots.
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7

// No flag runs the other way.
// CHECK-NOT: sync_block{{.*}}<PIPE_V>, <PIPE_FIX>
// CHECK-NOT: sync_block{{.*}}<PIPE_MTE1>, <PIPE_MTE3>

// CHECK: scope.return
// CHECK-NEXT: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, ssbuffer.inter_core_buf_count = 2 : i32} {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %c_block_m = arith.constant 64 : i32
    %c_block_n = arith.constant 128 : i32
    %c_num_m_blocks = arith.constant 64 : i32
    %c_batch_stride = arith.constant 262144 : i64
    %c0_i32 = arith.constant 0 : i32
    %c_sequence = arith.constant 4096 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = tensor.empty() : tensor<64x128xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %4 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %5 = tensor.empty() : tensor<64xf32>
    %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<64xf32>) -> tensor<64xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%5 : tensor<64xf32>) -> tensor<64xf32>
    scf.for %arg13 = %arg10 to %c1_i32 step %c_block_m  : i32 {
      %8 = arith.divsi %arg13, %c_num_m_blocks : i32
      %9 = arith.remsi %arg13, %c_num_m_blocks : i32
      %10 = arith.extsi %8 : i32 to i64
      %11 = arith.muli %10, %c_batch_stride : i64
      %12 = arith.index_cast %11 : i64 to index
      %13 = arith.muli %9, %c_block_m : i32
      %14 = arith.maxsi %13, %c0_i32 : i32
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.muli %15, %c64 : index
      %17 = arith.addi %16, %12 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%17], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%17], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() : memref<64x64xf16>
      memref.copy %reinterpret_cast, %alloc : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
      %18 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
      %19:5 = scf.for %arg14 = %c0_i32 to %c_sequence step %c_block_n iter_args(%arg15 = %7, %arg16 = %1, %arg17 = %6, %arg18 = %c0_i32, %arg19 = %c0_i32) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, i32, i32)  : i32 {
        %28 = arith.maxsi %arg18, %c0_i32 : i32
        %29 = arith.index_cast %28 : i32 to index
        %30 = arith.muli %29, %c64 : index
        %31 = arith.addi %30, %12 : index
        %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %32 = arith.maxsi %arg19, %c0_i32 : i32
        %33 = arith.index_cast %32 : i32 to index
        %34 = arith.muli %33, %c64 : index
        %35 = arith.addi %34, %12 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %alloc_7 = memref.alloc() : memref<128x64xf16>
        memref.copy %reinterpret_cast_6, %alloc_7 : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16>
        %36 = bufferization.to_tensor %alloc_7 restrict writable : memref<128x64xf16>
        %37 = tensor.empty() : tensor<64x128xf16>
        %transposed = linalg.transpose ins(%36 : tensor<128x64xf16>) outs(%37 : tensor<64x128xf16>) permutation = [1, 0]
        %38 = linalg.matmul {input_precision = "ieee"} ins(%18, %transposed : tensor<64x64xf16>, tensor<64x128xf16>) outs(%4 : tensor<64x128xf32>) -> tensor<64x128xf32>
        %39 = arith.mulf %38, %3 : tensor<64x128xf32>
        %reduced = linalg.reduce ins(%39 : tensor<64x128xf32>) outs(%6 : tensor<64xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %54 = arith.maximumf %in, %init : f32
            linalg.yield %54 : f32
          }
        %40 = arith.maximumf %arg17, %reduced : tensor<64xf32>
        %broadcasted_8 = linalg.broadcast ins(%40 : tensor<64xf32>) outs(%2 : tensor<64x128xf32>) dimensions = [1]
        %41 = arith.subf %39, %broadcasted_8 : tensor<64x128xf32>
        %42 = math.exp %41 : tensor<64x128xf32>
        %43 = arith.truncf %42 : tensor<64x128xf32> to tensor<64x128xf16>
        %alloc_9 = memref.alloc() : memref<128x64xf16>
        memref.copy %reinterpret_cast_5, %alloc_9 : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16>
        %44 = bufferization.to_tensor %alloc_9 restrict writable : memref<128x64xf16>
        %45 = linalg.fill ins(%cst_2 : f32) outs(%5 : tensor<64xf32>) -> tensor<64xf32>
        %reduced_10 = linalg.reduce ins(%42 : tensor<64x128xf32>) outs(%45 : tensor<64xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %54 = arith.addf %in, %init : f32
            linalg.yield %54 : f32
          }
        %46 = arith.subf %arg17, %40 : tensor<64xf32>
        %47 = math.exp %46 : tensor<64xf32>
        %48 = arith.mulf %arg15, %47 : tensor<64xf32>
        %49 = arith.addf %48, %reduced_10 : tensor<64xf32>
        %broadcasted_11 = linalg.broadcast ins(%47 : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [1]
        %50 = arith.mulf %arg16, %broadcasted_11 : tensor<64x64xf32>
        %51 = linalg.matmul {input_precision = "ieee"} ins(%43, %44 : tensor<64x128xf16>, tensor<128x64xf16>) outs(%50 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %52 = arith.addi %arg18, %c_block_n : i32
        %53 = arith.addi %arg19, %c_block_n : i32
        scf.yield %49, %51, %40, %52, %53 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, i32, i32
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %broadcasted = linalg.broadcast ins(%19#0 : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [1]
      %20 = arith.divf %19#1, %broadcasted : tensor<64x64xf32>
      %21 = math.log %19#0 : tensor<64xf32>
      %22 = arith.addf %19#2, %21 : tensor<64xf32>
      %23 = arith.muli %8, %c_sequence : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.index_cast %13 : i32 to index
      %26 = arith.addi %24, %25 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%26], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %22 in writable %reinterpret_cast_4 : (tensor<64xf32>, memref<64xf32, strided<[1], offset: ?>>) -> ()
      %27 = arith.truncf %20 : tensor<64x64xf32> to tensor<64x64xf16>
      bufferization.materialize_in_destination %27 in writable %reinterpret_cast_3 : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    }
    return
  }
}
