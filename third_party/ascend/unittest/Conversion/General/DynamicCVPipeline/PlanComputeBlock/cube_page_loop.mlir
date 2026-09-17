// RUN: triton-opt --split-input-file --op-classifier --verify-each %s | FileCheck %s --check-prefixes=CLASSIFY,COMMON
// RUN: triton-opt --split-input-file --plan-compute-block --compute-block-opt --verify-each %s | FileCheck %s --check-prefixes=MATERIAL,COMMON

// Ordinary range and tl.range both lower to this loop. Four 8-row pages
// exercise both halves of an NZ tile and advance into the next tile.
// COMMON-LABEL: func.func @loop_pages_i32
// CLASSIFY: scf.for
// CLASSIFY: memref.load {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY: arith.andi {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY: arith.shrsi {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY: memref.copy {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY: tensor.insert_slice {{.*}}ssbuffer.core_type = "CUBE"
// MATERIAL: %[[BUFFER:.*]] = memref.alloc() {{.*}}memref<1x2x16x16xf16, #hivm.address_space<cbuf>>
// MATERIAL: linalg.fill {{.*}}outs(%[[BUFFER]]
// MATERIAL: scf.for
// MATERIAL-NOT: iter_args
// MATERIAL: memref.load {{.*}}ssbuffer.block_id = [[ID:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"
// MATERIAL: arith.constant {{.*}} 16 : index
// MATERIAL: %[[OFFSET:.*]] = arith.muli
// MATERIAL: %[[OUTER:.*]] = arith.divui %[[OFFSET]]
// MATERIAL: %[[INNER:.*]] = arith.remui %[[OFFSET]]
// MATERIAL: memref.subview %[[BUFFER]][0, %[[OUTER]], %[[INNER]], 0] [1, 1, 8, 16]
// MATERIAL: arith.cmpi sgt
// MATERIAL: scf.if
// MATERIAL: hivm.hir.nd2nz {{.*}}dst_continuous{{.*}}ssbuffer.block_id = [[ID]] : i32, ssbuffer.core_type = "CUBE"
// MATERIAL-NOT: tensor.insert_slice
// MATERIAL: hivm.hir.convert_layout
// MATERIAL: linalg.matmul {{.*}}ssbuffer.block_id = [[ID]] : i32, ssbuffer.core_type = "CUBE"
// COMMON: return
func.func @loop_pages_i32(%metadata: memref<?xi32>, %cache: memref<?xf16>, %active: i1, %limit: i32, %q: tensor<16x16xf16>) -> tensor<16x32xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %i0 = arith.constant 0 : index
  %i8 = arith.constant 8 : index
  %i128 = arith.constant 128 : index
  %mask = arith.constant 16777215 : i32
  %shift = arith.constant 24 : i32
  %padding = arith.constant -1 : i32
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<32x16xf16>
  %initial = linalg.fill ins(%zero : f16) outs(%empty : tensor<32x16xf16>) -> tensor<32x16xf16>
  %pages = scf.for %iv = %c0 to %c4 step %c1 iter_args(%agg = %initial) -> (tensor<32x16xf16>) : i32 {
    %slot = arith.index_cast %iv : i32 to index
    %meta = scf.if %active -> (i32) {
      %loaded = memref.load %metadata[%slot] : memref<?xi32>
      scf.yield %loaded : i32
    } else {
      scf.yield %padding : i32
    }
    %physical = arith.andi %meta, %mask : i32
    %valid = arith.shrsi %meta, %shift : i32
    %valid_idx = arith.index_cast %valid : i32 to index
    %nonnegative = arith.maxsi %valid_idx, %i0 : index
    %size = arith.minsi %nonnegative, %i8 : index
    %physical_idx = arith.index_cast %physical : i32 to index
    %address = arith.muli %physical_idx, %i128 : index
    %src = memref.reinterpret_cast %cache to offset: [%address], sizes: [8, 16], strides: [16, 1] : memref<?xf16> to memref<8x16xf16, strided<[16, 1], offset: ?>>
    %page = memref.alloc() : memref<8x16xf16>
    %partial = arith.cmpi slt, %size, %i8 : index
    scf.if %partial {
      linalg.fill ins(%zero : f16) outs(%page : memref<8x16xf16>)
    }
    %src_view = memref.subview %src[0, 0] [%size, 16] [1, 1] : memref<8x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1], offset: ?>>
    %dst_view = memref.subview %page[0, 0] [%size, 16] [1, 1] : memref<8x16xf16> to memref<?x16xf16, strided<[16, 1]>>
    memref.copy %src_view, %dst_view : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1]>>
    %tensor = bufferization.to_tensor %page restrict writable : memref<8x16xf16> to tensor<8x16xf16>
    %product = arith.muli %iv, %c8 : i32
    %row = arith.index_cast %product : i32 to index
    %insert = tensor.insert_slice %tensor into %agg[%row, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<32x16xf16>
    scf.yield %insert : tensor<32x16xf16>
  }
  %transpose_empty = tensor.empty() : tensor<16x32xf16>
  %transposed = linalg.transpose ins(%pages : tensor<32x16xf16>) outs(%transpose_empty : tensor<16x32xf16>) permutation = [1, 0]
  %out = tensor.empty() : tensor<16x32xf32>
  %mm = linalg.matmul ins(%q, %transposed : tensor<16x16xf16>, tensor<16x32xf16>) outs(%out : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %mm : tensor<16x32xf32>
}

// -----

// Also accept index induction variables and a direct matmul input (V).
// COMMON-LABEL: func.func @loop_pages_index
// CLASSIFY: scf.for
// CLASSIFY: memref.load {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY: arith.andi {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY: arith.shrsi {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY: memref.copy {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY: tensor.insert_slice {{.*}}ssbuffer.core_type = "CUBE"
// MATERIAL: %[[BUFFER:.*]] = memref.alloc() {{.*}}memref<1x2x16x16xf16, #hivm.address_space<cbuf>>
// MATERIAL: linalg.fill {{.*}}outs(%[[BUFFER]]
// MATERIAL: scf.for
// MATERIAL-NOT: iter_args
// MATERIAL: memref.load {{.*}}ssbuffer.block_id = [[ID:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"
// MATERIAL: arith.constant {{.*}} 16 : index
// MATERIAL: %[[OFFSET:.*]] = arith.muli
// MATERIAL: %[[OUTER:.*]] = arith.divui %[[OFFSET]]
// MATERIAL: %[[INNER:.*]] = arith.remui %[[OFFSET]]
// MATERIAL: memref.subview %[[BUFFER]][0, %[[OUTER]], %[[INNER]], 0] [1, 1, 8, 16]
// MATERIAL: arith.cmpi sgt
// MATERIAL: scf.if
// MATERIAL: hivm.hir.nd2nz {{.*}}dst_continuous{{.*}}ssbuffer.block_id = [[ID]] : i32, ssbuffer.core_type = "CUBE"
// MATERIAL-NOT: tensor.insert_slice
// MATERIAL: hivm.hir.convert_layout
// MATERIAL: linalg.matmul {{.*}}ssbuffer.block_id = [[ID]] : i32, ssbuffer.core_type = "CUBE"
// COMMON: return
func.func @loop_pages_index(%metadata: memref<?xi32>, %cache: memref<?xf16>, %active: i1, %limit: index, %q: tensor<16x32xf16>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %i0 = arith.constant 0 : index
  %i8 = arith.constant 8 : index
  %i128 = arith.constant 128 : index
  %mask = arith.constant 16777215 : i32
  %shift = arith.constant 24 : i32
  %padding = arith.constant -1 : i32
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<32x16xf16>
  %initial = linalg.fill ins(%zero : f16) outs(%empty : tensor<32x16xf16>) -> tensor<32x16xf16>
  %pages = scf.for %iv = %c0 to %c4 step %c1 iter_args(%agg = %initial) -> (tensor<32x16xf16>) {
    %slot = arith.addi %iv, %i0 : index
    %meta = scf.if %active -> (i32) {
      %loaded = memref.load %metadata[%slot] : memref<?xi32>
      scf.yield %loaded : i32
    } else {
      scf.yield %padding : i32
    }
    %physical = arith.andi %meta, %mask : i32
    %valid = arith.shrsi %meta, %shift : i32
    %valid_idx = arith.index_cast %valid : i32 to index
    %nonnegative = arith.maxsi %valid_idx, %i0 : index
    %size = arith.minsi %nonnegative, %i8 : index
    %physical_idx = arith.index_cast %physical : i32 to index
    %address = arith.muli %physical_idx, %i128 : index
    %src = memref.reinterpret_cast %cache to offset: [%address], sizes: [8, 16], strides: [16, 1] : memref<?xf16> to memref<8x16xf16, strided<[16, 1], offset: ?>>
    %page = memref.alloc() : memref<8x16xf16>
    %partial = arith.cmpi slt, %size, %i8 : index
    scf.if %partial {
      linalg.fill ins(%zero : f16) outs(%page : memref<8x16xf16>)
    }
    %src_view = memref.subview %src[0, 0] [%size, 16] [1, 1] : memref<8x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1], offset: ?>>
    %dst_view = memref.subview %page[0, 0] [%size, 16] [1, 1] : memref<8x16xf16> to memref<?x16xf16, strided<[16, 1]>>
    memref.copy %src_view, %dst_view : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1]>>
    %tensor = bufferization.to_tensor %page restrict writable : memref<8x16xf16> to tensor<8x16xf16>
    %product = arith.muli %iv, %c8 : index
    %row = arith.addi %product, %i0 : index
    %insert = tensor.insert_slice %tensor into %agg[%product, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<32x16xf16>
    scf.yield %insert : tensor<32x16xf16>
  }
  %out = tensor.empty() : tensor<16x16xf32>
  %mm = linalg.matmul ins(%q, %pages : tensor<16x32xf16>, tensor<32x16xf16>) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mm : tensor<16x16xf32>
}

// -----

// Runtime trip counts may leave some aggregate rows unwritten.
// COMMON-LABEL: func.func @dynamic_trip_count
// COMMON: memref.copy {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: tensor.insert_slice {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
// COMMON: return
func.func @dynamic_trip_count(%metadata: memref<?xi32>, %cache: memref<8x16xf16>, %limit: i32, %q: tensor<16x32xf16>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %i0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<32x16xf16>
  %initial = linalg.fill ins(%zero : f16) outs(%empty : tensor<32x16xf16>) -> tensor<32x16xf16>
  %pages = scf.for %iv = %c0 to %limit step %c1 iter_args(%agg = %initial) -> (tensor<32x16xf16>) : i32 {
    %page = memref.alloc() : memref<8x16xf16>
    memref.copy %cache, %page : memref<8x16xf16> to memref<8x16xf16>
    %tensor = bufferization.to_tensor %page restrict writable : memref<8x16xf16> to tensor<8x16xf16>
    %product = arith.muli %iv, %c8 : i32
    %row = arith.index_cast %product : i32 to index
    %insert = tensor.insert_slice %tensor into %agg[%row, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<32x16xf16>
    scf.yield %insert : tensor<32x16xf16>
  }
  %out = tensor.empty() : tensor<16x16xf32>
  %mm = linalg.matmul ins(%q, %pages : tensor<16x32xf16>, tensor<32x16xf16>) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mm : tensor<16x16xf32>
}

// -----

// Repeated writes to one page do not cover the aggregate.
// COMMON-LABEL: func.func @overlapping_pages
// COMMON: memref.copy {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: tensor.insert_slice {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
// COMMON: return
func.func @overlapping_pages(%metadata: memref<?xi32>, %cache: memref<8x16xf16>, %limit: i32, %q: tensor<16x32xf16>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %i0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<32x16xf16>
  %initial = linalg.fill ins(%zero : f16) outs(%empty : tensor<32x16xf16>) -> tensor<32x16xf16>
  %pages = scf.for %iv = %c0 to %c4 step %c1 iter_args(%agg = %initial) -> (tensor<32x16xf16>) : i32 {
    %page = memref.alloc() : memref<8x16xf16>
    memref.copy %cache, %page : memref<8x16xf16> to memref<8x16xf16>
    %tensor = bufferization.to_tensor %page restrict writable : memref<8x16xf16> to tensor<8x16xf16>
    %product = arith.muli %iv, %c8 : i32
    %row = arith.index_cast %product : i32 to index
    %insert = tensor.insert_slice %tensor into %agg[%i0, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<32x16xf16>
    scf.yield %insert : tensor<32x16xf16>
  }
  %out = tensor.empty() : tensor<16x16xf32>
  %mm = linalg.matmul ins(%q, %pages : tensor<16x32xf16>, tensor<32x16xf16>) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mm : tensor<16x16xf32>
}

// -----

// Do not recolor unrelated side effects along with the loader.
// COMMON-LABEL: func.func @extra_store
// COMMON: memref.copy {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: tensor.insert_slice {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
// COMMON: return
func.func @extra_store(%metadata: memref<?xi32>, %cache: memref<8x16xf16>, %limit: i32, %q: tensor<16x32xf16>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %i0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<32x16xf16>
  %initial = linalg.fill ins(%zero : f16) outs(%empty : tensor<32x16xf16>) -> tensor<32x16xf16>
  %pages = scf.for %iv = %c0 to %c4 step %c1 iter_args(%agg = %initial) -> (tensor<32x16xf16>) : i32 {
    %page = memref.alloc() : memref<8x16xf16>
    memref.copy %cache, %page : memref<8x16xf16> to memref<8x16xf16>
    %tensor = bufferization.to_tensor %page restrict writable : memref<8x16xf16> to tensor<8x16xf16>
    %product = arith.muli %iv, %c8 : i32
    %row = arith.index_cast %product : i32 to index
    memref.store %iv, %metadata[%i0] : memref<?xi32>
    %insert = tensor.insert_slice %tensor into %agg[%row, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<32x16xf16>
    scf.yield %insert : tensor<32x16xf16>
  }
  %out = tensor.empty() : tensor<16x16xf32>
  %mm = linalg.matmul ins(%q, %pages : tensor<16x32xf16>, tensor<32x16xf16>) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mm : tensor<16x16xf32>
}

// -----

// Observing the carried tensor would see the progressively assembled value.
// COMMON-LABEL: func.func @observed_aggregate
// COMMON: memref.copy {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: tensor.insert_slice {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
// COMMON: return
func.func @observed_aggregate(%metadata: memref<?xi32>, %cache: memref<8x16xf16>, %limit: i32, %q: tensor<16x32xf16>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %i0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<32x16xf16>
  %initial = linalg.fill ins(%zero : f16) outs(%empty : tensor<32x16xf16>) -> tensor<32x16xf16>
  %pages = scf.for %iv = %c0 to %c4 step %c1 iter_args(%agg = %initial) -> (tensor<32x16xf16>) : i32 {
    %page = memref.alloc() : memref<8x16xf16>
    memref.copy %cache, %page : memref<8x16xf16> to memref<8x16xf16>
    %tensor = bufferization.to_tensor %page restrict writable : memref<8x16xf16> to tensor<8x16xf16>
    %product = arith.muli %iv, %c8 : i32
    %row = arith.index_cast %product : i32 to index
    %observed = tensor.extract %agg[%i0, %i0] : tensor<32x16xf16>
    memref.store %observed, %cache[%i0, %i0] : memref<8x16xf16>
    %insert = tensor.insert_slice %tensor into %agg[%row, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<32x16xf16>
    scf.yield %insert : tensor<32x16xf16>
  }
  %out = tensor.empty() : tensor<16x16xf32>
  %mm = linalg.matmul ins(%q, %pages : tensor<16x32xf16>, tensor<32x16xf16>) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mm : tensor<16x16xf32>
}
