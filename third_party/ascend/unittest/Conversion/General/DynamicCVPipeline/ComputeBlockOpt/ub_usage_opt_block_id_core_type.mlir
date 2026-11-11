// RUN: triton-opt --ub-usage-opt %s | FileCheck %s

// A block_id identifies one compute block and a compute block belongs to one
// core, so a block_id must never be used by both a CUBE and a VECTOR op.
//
// The kernel below has CUBE blocks 1/3/4 and VECTOR blocks 5/6/7/8/9/10.
// UB usage opt moves VECTOR ops between VECTOR blocks to shrink the amount
// of data crossing a block boundary, and it used to drag a CUBE dependency
// along into the VECTOR block, leaving that block with both core types.
// Every block_id below must keep its own core type after the pass.
//
// CHECK-LABEL: func.func @chunk_fwd_kernel_o_npu_kernel
// CUBE blocks must not gain a VECTOR op.
// CHECK-NOT: ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"
// CHECK-NOT: ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"
// CHECK-NOT: ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"
// VECTOR blocks must not gain a CUBE op.
// CHECK-NOT: ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"
// CHECK-NOT: ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"
// CHECK-NOT: ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"
// CHECK-NOT: ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"
// CHECK-NOT: ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"
// CHECK-NOT: ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"
// The checks above cover everything up to the next positive CHECK.
// CHECK: return
func.func @chunk_fwd_kernel_o_npu_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.tensor_kind = 1 : i32}, %arg8: memref<?xi64> {tt.tensor_kind = 0 : i32}, %arg9: memref<?xi64> {tt.tensor_kind = 0 : i32}, %arg10: f32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
  %c32768_i64 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 32768 : i64
  %c0_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0 : i32
  %c4096 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 4096 : index
  %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
  %c128 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 128 : index
  %c256 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 256 : index
  %c64 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 64 : index
  %c8192 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 8192 : index
  %c1 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 1 : index
  %c0 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0 : index
  %c32_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 32 : i32
  %c256_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 256 : i32
  %c128_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 128 : i32
  %c32_i64 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 32 : i64
  %c64_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 64 : i32
  %0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x128xf32>
  %1 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
  %3 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %4 = arith.divsi %arg16, %c32_i32 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
  %5 = arith.remsi %arg16, %c32_i32 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
  %6 = arith.index_cast %4 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%6], sizes: [1], strides: [1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %7 = memref.load %reinterpret_cast[%c0] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi64, strided<[1], offset: ?>>
  %8 = arith.trunci %7 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i64 to i32
  %9 = arith.addi %6, %c1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
  %reinterpret_cast_0 = memref.reinterpret_cast %arg8 to offset: [%9], sizes: [1], strides: [1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %10 = memref.load %reinterpret_cast_0[%c0] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi64, strided<[1], offset: ?>>
  %11 = arith.trunci %10 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i64 to i32
  %12 = arith.subi %11, %8 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
  %reinterpret_cast_1 = memref.reinterpret_cast %arg9 to offset: [%6], sizes: [1], strides: [1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %13 = memref.load %reinterpret_cast_1[%c0] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi64, strided<[1], offset: ?>>
  %14 = arith.muli %8, %c32_i32 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
  %15 = arith.extsi %5 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to i64
  %16 = arith.addi %14, %5 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
  %17 = arith.muli %16, %c256_i32 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
  %18 = arith.index_cast %17 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
  %19 = arith.muli %16, %c128_i32 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
  %20 = arith.index_cast %19 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
  %21 = arith.muli %arg15, %c128_i32 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
  %cst_2 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 0xFF800000 : f32
  %c2_i32 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 2 : i32
  %c63_i32 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 63 : i32
  %c0_i32_3 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
  %c4096_4 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 4096 : index
  %cst_5 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
  %cst_6 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : bf16
  %c128_7 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 128 : index
  %c64_8 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 64 : index
  %c1_9 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
  %c0_10 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
  %c32_i32_11 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 32 : i32
  %c1_i32 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
  %c128_i32_12 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 128 : i32
  %c64_i32_13 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 64 : i32
  %22 = tensor.empty() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
  %23 = tensor.empty() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
  %24 = linalg.fill {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_5 : f32) outs(%23 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %25 = linalg.fill {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_2 : f32) outs(%23 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %26 = arith.divsi %arg16, %c32_i32_11 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32
  %27 = arith.remsi %arg16, %c32_i32_11 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32
  %28 = arith.index_cast %26 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
  %reinterpret_cast_14 = memref.reinterpret_cast %arg8 to offset: [%28], sizes: [1], strides: [1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %29 = memref.load %reinterpret_cast_14[%c0_10] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi64, strided<[1], offset: ?>>
  %30 = arith.trunci %29 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i64 to i32
  %31 = arith.addi %28, %c1_9 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : index
  %reinterpret_cast_15 = memref.reinterpret_cast %arg8 to offset: [%31], sizes: [1], strides: [1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
  %32 = memref.load %reinterpret_cast_15[%c0_10] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi64, strided<[1], offset: ?>>
  %33 = arith.trunci %32 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i64 to i32
  %34 = arith.subi %33, %30 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32
  %35 = arith.addi %34, %c63_i32 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32
  %36 = arith.divsi %35, %c64_i32_13 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32
  %37 = arith.muli %30, %c32_i32_11 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32
  %38 = arith.addi %37, %27 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32
  %39 = arith.muli %38, %c128_i32_12 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32
  %40 = arith.index_cast %39 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
  %41 = arith.muli %arg15, %c128_i32_12 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32
  %42 = tensor.empty() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xi32>
  %43 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%42 : tensor<64xi32>) attrs =  {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR", tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 64 : index} {
  ^bb0(%out: i32):
    %52 = linalg.index 0 : index
    %53 = arith.index_cast %52 : index to i32
    linalg.yield %53 : i32
  } -> tensor<64xi32>
  %44 = arith.index_cast %30 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
  %45 = arith.muli %27, %arg11 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32
  %46 = arith.index_cast %45 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
  %47 = arith.addi %44, %46 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : index
  %48 = arith.sitofp %43 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xi32> to tensor<64xf32>
  %broadcasted = linalg.broadcast ins(%48 : tensor<64xf32>) outs(%23 : tensor<64x64xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}
  %broadcasted_16 = linalg.broadcast ins(%48 : tensor<64xf32>) outs(%23 : tensor<64x64xf32>) dimensions = [0]  {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"}
  %49 = arith.cmpf oge, %broadcasted, %broadcasted_16 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
  %50 = linalg.fill {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} ins(%arg10 : f32) outs(%22 : tensor<64x128xf32>) -> tensor<64x128xf32>
  scf.for %arg18 = %c0_i32 to %36 step %c1_i32  : i32 {
    %52 = arith.extsi %arg18 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32 to i64
    %53 = arith.addi %13, %52 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i64
    %54 = arith.muli %53, %c32_i64 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i64
    %55 = arith.addi %54, %15 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i64
    %56 = arith.muli %55, %c32768_i64 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i64
    %57 = arith.index_cast %56 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i64 to index
    %58 = arith.muli %arg18, %c64_i32 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32
    %59:2 = scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg20 = %1, %arg21 = %3) -> (tensor<64x128xf32>, tensor<64x64xf32>)  : i32 {
      %141 = arith.muli %arg19, %c128_i32 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32
      %142 = arith.maxsi %58, %c0_i32 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32
      %143 = arith.index_cast %142 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %144 = arith.maxsi %141, %c0_i32 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32
      %145 = arith.index_cast %144 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %146 = arith.muli %143, %c8192 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %147 = arith.addi %146, %18 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %148 = arith.index_cast %12 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %149 = arith.addi %147, %145 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %reinterpret_cast_28 = memref.reinterpret_cast %arg2 to offset: [%149], sizes: [64, 128], strides: [8192, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<64x128xbf16, strided<[8192, 1], offset: ?>>
      %alloc_29 = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16>
      %150 = arith.subi %149, %18 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %151 = arith.divsi %150, %c8192 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %152 = arith.subi %148, %151 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %153 = arith.maxsi %152, %c0 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %154 = arith.minsi %153, %c64 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %155 = arith.remsi %150, %c8192 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %156 = arith.subi %c256, %155 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %157 = arith.maxsi %156, %c0 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %158 = arith.minsi %157, %c128 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %159 = arith.subi %c0_i32, %58 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32
      %160 = arith.maxsi %159, %c0_i32 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32
      %161 = arith.index_cast %160 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %162 = arith.minsi %161, %154 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %163 = arith.subi %154, %162 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %164 = arith.subi %c0_i32, %141 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32
      %165 = arith.maxsi %164, %c0_i32 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32
      %166 = arith.index_cast %165 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %167 = arith.minsi %166, %158 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %168 = arith.subi %158, %167 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %169 = arith.cmpi slt, %163, %c64 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %170 = arith.cmpi slt, %168, %c128 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %171 = arith.ori %169, %170 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i1
      scf.if %171 {
        linalg.fill {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_6 : bf16) outs(%alloc_29 : memref<64x128xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
      %subview_30 = memref.subview %reinterpret_cast_28[0, 0] [%163, %168] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16, strided<[8192, 1], offset: ?>> to memref<?x?xbf16, strided<[8192, 1], offset: ?>>
      %subview_31 = memref.subview %alloc_29[%162, %167] [%163, %168] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      memref.copy %subview_30, %subview_31 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[8192, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      %172 = bufferization.to_tensor %alloc_29 restrict writable {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16> to tensor<64x128xbf16>
      %reinterpret_cast_32 = memref.reinterpret_cast %arg3 to offset: [%149], sizes: [64, 128], strides: [8192, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<64x128xbf16, strided<[8192, 1], offset: ?>>
      %alloc_33 = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16>
      scf.if %171 {
        linalg.fill {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_6 : bf16) outs(%alloc_33 : memref<64x128xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
      %subview_34 = memref.subview %reinterpret_cast_32[0, 0] [%163, %168] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16, strided<[8192, 1], offset: ?>> to memref<?x?xbf16, strided<[8192, 1], offset: ?>>
      %subview_35 = memref.subview %alloc_33[%162, %167] [%163, %168] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      memref.copy %subview_34, %subview_35 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[8192, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      %173 = bufferization.to_tensor %alloc_33 restrict writable {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16> to tensor<64x128xbf16>
      %174 = tensor.empty() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x64xbf16>
      %transposed = linalg.transpose ins(%173 : tensor<64x128xbf16>) outs(%174 : tensor<128x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"}
      %175 = arith.maxsi %21, %c0_i32 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32
      %176 = arith.index_cast %175 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %177 = arith.muli %145, %c128 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %178 = arith.addi %177, %57 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %179 = arith.addi %178, %176 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %reinterpret_cast_36 = memref.reinterpret_cast %arg5 to offset: [%179], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<128x128xbf16, strided<[128, 1], offset: ?>>
      %alloc_37 = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
      %180 = arith.subi %179, %57 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %181 = arith.divsi %180, %c128 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %182 = arith.subi %c256, %181 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %183 = arith.maxsi %182, %c0 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %184 = arith.minsi %183, %c128 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %185 = arith.remsi %180, %c128 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %186 = arith.subi %c128, %185 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %187 = arith.maxsi %186, %c0 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %188 = arith.minsi %187, %c128 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %189 = arith.minsi %166, %184 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %190 = arith.subi %184, %189 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %191 = arith.subi %c0_i32, %21 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32
      %192 = arith.maxsi %191, %c0_i32 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32
      %193 = arith.index_cast %192 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %194 = arith.minsi %193, %188 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %195 = arith.subi %188, %194 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %196 = arith.cmpi slt, %190, %c128 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %197 = arith.cmpi slt, %195, %c128 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
      %198 = arith.ori %196, %197 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i1
      scf.if %198 {
        linalg.fill {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_6 : bf16) outs(%alloc_37 : memref<128x128xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
      %subview_38 = memref.subview %reinterpret_cast_36[0, 0] [%190, %195] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16, strided<[128, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      %subview_39 = memref.subview %alloc_37[%189, %194] [%190, %195] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      memref.copy %subview_38, %subview_39 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[128, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
      %199 = bufferization.to_tensor %alloc_37 restrict writable {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
      %200 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%172, %199 : tensor<64x128xbf16>, tensor<128x128xbf16>) outs(%arg20 : tensor<64x128xf32>) -> tensor<64x128xf32>
      %201 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%172, %transposed : tensor<64x128xbf16>, tensor<128x64xbf16>) outs(%arg21 : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE, CUBE"} %200, %201 : tensor<64x128xf32>, tensor<64x64xf32>
    } {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE, CUBE"}
    %60 = arith.muli %arg18, %c64_i32_13 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %61 = arith.index_cast %60 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
    %62 = arith.addi %47, %61 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %reinterpret_cast_17 = memref.reinterpret_cast %arg6 to offset: [%62], sizes: [64], strides: [1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<64xf32>
    %63 = arith.addi %61, %c64_8 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %64 = arith.index_cast %34 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
    %65 = arith.maxsi %61, %64 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %66 = arith.minsi %63, %65 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %67 = arith.subi %66, %61 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %68 = arith.cmpi slt, %67, %c64_8 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    scf.if %68 {
      linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_5 : f32) outs(%alloc : memref<64xf32>)
    } {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32}
    %subview = memref.subview %reinterpret_cast_17[0] [%67] [1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<64xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_18 = memref.subview %alloc[0] [%67] [1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<64xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_18 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %69 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<64xf32> to tensor<64xf32>
    %70 = math.exp %69 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %broadcasted_19 = linalg.broadcast ins(%70 : tensor<64xf32>) outs(%22 : tensor<64x128xf32>) dimensions = [1]  {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"}
    %broadcasted_20 = linalg.broadcast ins(%69 : tensor<64xf32>) outs(%23 : tensor<64x64xf32>) dimensions = [1]  {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"}
    %broadcasted_21 = linalg.broadcast ins(%69 : tensor<64xf32>) outs(%23 : tensor<64x64xf32>) dimensions = [0]  {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"}
    %71 = arith.subf %broadcasted_20, %broadcasted_21 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    %72 = arith.cmpf ole, %71, %24 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    %73 = arith.select %72, %71, %25 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xi1>, tensor<64x64xf32>
    %74 = math.exp %73 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    %75 = arith.maxsi %60, %c0_i32_3 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %76 = arith.index_cast %75 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
    %77 = arith.maxsi %41, %c0_i32_3 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %78 = arith.index_cast %77 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
    %79 = arith.muli %76, %c4096_4 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %80 = arith.addi %79, %40 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %81 = arith.addi %80, %78 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %reinterpret_cast_22 = memref.reinterpret_cast %arg7 to offset: [%81], sizes: [64, 128], strides: [4096, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<64x128xbf16, strided<[4096, 1], offset: ?>>
    %82 = arith.subi %81, %40 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %83 = arith.divsi %82, %c4096_4 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %84 = arith.subi %64, %83 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %85 = arith.maxsi %84, %c0_10 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %86 = arith.minsi %85, %c64_8 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %87 = arith.remsi %82, %c4096_4 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %88 = arith.subi %c128_7, %87 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %89 = arith.maxsi %88, %c0_10 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %90 = arith.minsi %89, %c128_7 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %91 = arith.subi %c0_i32_3, %60 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %92 = arith.maxsi %91, %c0_i32_3 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %93 = arith.index_cast %92 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
    %94 = arith.minsi %93, %86 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %95 = arith.subi %86, %94 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %96 = arith.subi %c0_i32_3, %41 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %97 = arith.maxsi %96, %c0_i32_3 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %98 = arith.index_cast %97 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
    %99 = arith.minsi %98, %90 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %100 = arith.subi %90, %99 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
    %subview_23 = memref.subview %reinterpret_cast_22[0, 0] [%95, %100] [1, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<64x128xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
    %101 = arith.mulf %59#0, %broadcasted_19 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
    %102 = arith.mulf %101, %50 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
    %103 = arith.mulf %59#1, %74 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    %104 = arith.select %49, %103, %24 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xi1>, tensor<64x64xf32>
    %105 = arith.truncf %104 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xbf16>
    %106 = arith.index_cast %12 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %107 = arith.maxsi %58, %c0_i32 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32
    %108 = arith.index_cast %107 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %109 = arith.maxsi %21, %c0_i32 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32
    %110 = arith.index_cast %109 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %111 = arith.muli %108, %c4096 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %112 = arith.addi %111, %20 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %113 = arith.addi %112, %110 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %reinterpret_cast_24 = memref.reinterpret_cast %arg4 to offset: [%113], sizes: [64, 128], strides: [4096, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<64x128xbf16, strided<[4096, 1], offset: ?>>
    %alloc_25 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16>
    %114 = arith.subi %113, %20 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %115 = arith.divsi %114, %c4096 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %116 = arith.subi %106, %115 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %117 = arith.maxsi %116, %c0 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %118 = arith.minsi %117, %c64 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %119 = arith.remsi %114, %c4096 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %120 = arith.subi %c128, %119 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %121 = arith.maxsi %120, %c0 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %122 = arith.minsi %121, %c128 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %123 = arith.subi %c0_i32, %58 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32
    %124 = arith.maxsi %123, %c0_i32 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32
    %125 = arith.index_cast %124 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %126 = arith.minsi %125, %118 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %127 = arith.subi %118, %126 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %128 = arith.subi %c0_i32, %21 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32
    %129 = arith.maxsi %128, %c0_i32 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32
    %130 = arith.index_cast %129 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %131 = arith.minsi %130, %122 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %132 = arith.subi %122, %131 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %133 = arith.cmpi slt, %127, %c64 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %134 = arith.cmpi slt, %132, %c128 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
    %135 = arith.ori %133, %134 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i1
    scf.if %135 {
      linalg.fill {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_6 : bf16) outs(%alloc_25 : memref<64x128xbf16>)
    } {hivm.unlikely_condition, ssbuffer.block_id = 3 : i32}
    %subview_26 = memref.subview %reinterpret_cast_24[0, 0] [%127, %132] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
    %subview_27 = memref.subview %alloc_25[%126, %131] [%127, %132] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
    memref.copy %subview_26, %subview_27 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
    %136 = bufferization.to_tensor %alloc_25 restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xbf16> to tensor<64x128xbf16>
    %137 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%105, %136 : tensor<64x64xbf16>, tensor<64x128xbf16>) outs(%1 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %138 = arith.mulf %137, %50 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
    %139 = arith.addf %102, %138 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32>
    %140 = arith.truncf %139 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32> to tensor<64x128xbf16>
    %extracted_slice = tensor.extract_slice %140[%94, %99] [%95, %100] [1, 1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xbf16> to tensor<?x?xbf16>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_23 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[4096, 1], offset: ?>>) -> ()
  }
  %51 = linalg.fill {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_5 : f32) outs(%22 : tensor<64x128xf32>) -> tensor<64x128xf32>
  return
}
