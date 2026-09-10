// RUN: triton-opt %s --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=SAFE --implicit-check-not=scf.if
// RUN: sed 's/to %upper step/to %unrelated step/' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=UNKNOWN
// RUN: sed 's@// CLOBBER_METADATA@memref.store %%zero_i32, %%packed[%%zero] : memref<?xi32>@' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=CLOBBER
// RUN: sed 's/memref.reinterpret_cast %%packed to offset: \[%%next_idx\]/memref.reinterpret_cast %%different to offset: [%%next_idx]/' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=CLOBBER
// RUN: sed 's/%%within_count, %%within_topk/%%within_count, %%enabled/' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=UNKNOWN
// RUN: sed 's@// CLOBBER_METADATA@memref.dealloc %%packed : memref<?xi32>@' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=CLOBBER
// RUN: sed 's@// CLOBBER_COUNT@memref.store %%zero_i32, %%counts[%%zero] : memref<?xi32>@' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=UNKNOWN

// The loop can be empty. Inside an executing iteration its first metadata
// address is valid because iv < min(count, topk). A tail page must select that
// address before loading, then select -1 afterwards. Do not hoist the reads out
// of the loop or lose the row offset in the fallback address.
// SAFE-LABEL: func.func @page_metadata(
// SAFE: %[[OTHER:.*]] = arith.constant -1 : i32
// SAFE: scf.for
// SAFE: %[[FIRST_OFFSET:.*]] = arith.addi
// SAFE: %[[FIRST_VIEW:.*]] = memref.reinterpret_cast {{.*}}offset: [%[[FIRST_OFFSET]]]
// SAFE: memref.load %[[FIRST_VIEW]]
// SAFE: hivm.hir.nd2nz
// SAFE: %[[ADDRESS:.*]] = arith.select %[[MASK:.*]], %{{.*}}, %[[FIRST_OFFSET]]
// SAFE: %[[VIEW:.*]] = memref.reinterpret_cast {{.*}}offset: [%[[ADDRESS]]]
// SAFE: %[[RAW:.*]] = memref.load %[[VIEW]]
// SAFE: arith.select %[[MASK]], %[[RAW]], %[[OTHER]]
// SAFE: hivm.hir.nd2nz
// SAFE: scf.yield

// A loop with an unrelated upper bound provides no known valid address.
// UNKNOWN-LABEL: func.func @page_metadata(
// UNKNOWN: scf.for
// UNKNOWN: scf.if
// UNKNOWN: memref.load
// UNKNOWN: hivm.hir.nd2nz
// UNKNOWN: scf.if
// UNKNOWN: memref.load
// UNKNOWN: hivm.hir.nd2nz

// Do not reuse a proof across a possibly aliasing write, or use an address
// from a different allocation as the fallback.
// CLOBBER-LABEL: func.func @page_metadata(
// CLOBBER: scf.for
// CLOBBER-NOT: scf.if
// CLOBBER: memref.load
// CLOBBER: hivm.hir.nd2nz
// CLOBBER: scf.if
// CLOBBER: memref.load
// CLOBBER: hivm.hir.nd2nz
func.func @page_metadata(%packed: memref<?xi32>, %different: memref<?xi32>, %cache: memref<?xf16>, %row: index, %counts: memref<?xi32>, %topk: i32, %unrelated: i32, %enabled: i1, %q: tensor<16x16xf16>, %initial: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %zero = arith.constant 0 : index
  %zero_i32 = arith.constant 0 : i32
  %one_i32 = arith.constant 1 : i32
  %two_i32 = arith.constant 2 : i32
  %other = arith.constant -1 : i32
  // The planner may clone a read for CUBE while leaving the loop bound on VECTOR.
  %count_offset0 = arith.addi %row, %zero {ssbuffer.block_id = 6 : i32} : index
  %count_view0 = memref.reinterpret_cast %counts to offset: [%count_offset0], sizes: [1], strides: [1] {ssbuffer.block_id = 6 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
  %count_bound = memref.load %count_view0[%zero] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi32, strided<[1], offset: ?>>
  %tmp = tensor.empty() : tensor<16xi32>
  %filled = linalg.fill ins(%zero_i32 : i32) outs(%tmp : tensor<16xi32>) -> tensor<16xi32>
  // CLOBBER_COUNT
  %count_offset1 = arith.addi %row, %zero {ssbuffer.block_id = 1 : i32} : index
  %count_view1 = memref.reinterpret_cast %counts to offset: [%count_offset1], sizes: [1], strides: [1] {ssbuffer.block_id = 1 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
  %count = memref.load %count_view1[%zero] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi32, strided<[1], offset: ?>>
  %upper = arith.minsi %count_bound, %topk : i32
  %result = scf.for %iv = %zero_i32 to %upper step %two_i32 iter_args(%acc = %initial) -> tensor<16x16xf32> : i32 {
    %within_count = arith.cmpi slt, %iv, %count : i32
    %within_topk = arith.cmpi slt, %iv, %topk : i32
    %active = arith.andi %within_count, %within_topk : i1
    %meta0 = scf.if %active -> i32 {
      %iv_idx = arith.index_cast %iv : i32 to index
      %first_idx = arith.addi %row, %iv_idx : index
      %ptr = memref.reinterpret_cast %packed to offset: [%first_idx], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %loaded = memref.load %ptr[%zero] : memref<1xi32, strided<[1], offset: ?>>
      scf.yield %loaded : i32
    } else {
      scf.yield %other : i32
    } {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"}
    %offset0 = arith.index_cast %meta0 : i32 to index
    %src0 = memref.reinterpret_cast %cache to offset: [%offset0], sizes: [8, 16], strides: [16, 1] : memref<?xf16> to memref<8x16xf16, strided<[16, 1], offset: ?>>
    %page0 = memref.alloc() : memref<8x16xf16>
    memref.copy %src0, %page0 {ssbuffer.block_id = 3 : i32} : memref<8x16xf16, strided<[16, 1], offset: ?>> to memref<8x16xf16>
    %tensor0 = bufferization.to_tensor %page0 restrict writable : memref<8x16xf16> to tensor<8x16xf16>
    // CLOBBER_METADATA
    %next = arith.addi %iv, %one_i32 : i32
    %next_count = arith.cmpi slt, %next, %count : i32
    %next_topk = arith.cmpi slt, %next, %topk : i32
    %next_active = arith.andi %next_count, %next_topk : i1
    %meta1 = scf.if %next_active -> i32 {
      %iv_idx = arith.index_cast %next : i32 to index
      %next_idx = arith.addi %row, %iv_idx : index
      %ptr = memref.reinterpret_cast %packed to offset: [%next_idx], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %loaded = memref.load %ptr[%zero] : memref<1xi32, strided<[1], offset: ?>>
      scf.yield %loaded : i32
    } else {
      scf.yield %other : i32
    } {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"}
    %safe_meta1 = arith.select %next_active, %meta1, %zero_i32 : i32
    %offset1 = arith.index_cast %safe_meta1 : i32 to index
    %src1 = memref.reinterpret_cast %cache to offset: [%offset1], sizes: [8, 16], strides: [16, 1] : memref<?xf16> to memref<8x16xf16, strided<[16, 1], offset: ?>>
    %page1 = memref.alloc() : memref<8x16xf16>
    memref.copy %src1, %page1 {ssbuffer.block_id = 3 : i32} : memref<8x16xf16, strided<[16, 1], offset: ?>> to memref<8x16xf16>
    %tensor1 = bufferization.to_tensor %page1 restrict writable : memref<8x16xf16> to tensor<8x16xf16>
    %empty = tensor.empty() : tensor<16x16xf16>
    %insert0 = tensor.insert_slice %tensor0 into %empty[0, 0] [8, 16] [1, 1] {ssbuffer.block_id = 3 : i32} : tensor<8x16xf16> into tensor<16x16xf16>
    %insert1 = tensor.insert_slice %tensor1 into %insert0[8, 0] [8, 16] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<8x16xf16> into tensor<16x16xf16>
    %mm = linalg.matmul ins(%q, %insert1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mm : tensor<16x16xf32>
  }
  return %result : tensor<16x16xf32>
}
