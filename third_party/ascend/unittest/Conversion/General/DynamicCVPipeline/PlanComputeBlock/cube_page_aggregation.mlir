// RUN: triton-opt --split-input-file --op-classifier --verify-each %s | FileCheck %s --check-prefixes=CLASSIFY,COMMON
// RUN: triton-opt --split-input-file --plan-compute-block --compute-block-opt --verify-each %s | FileCheck %s --check-prefixes=MATERIAL,COMMON

// static_range unrolls page loads into insert_slice chains. Both a direct
// matmul operand (V) and an operand behind transpose (K) must be recognized.
// The masked scalar metadata load must follow its address users onto CUBE.
// Classification and materialization both run without an opt-in flag.
// COMMON-LABEL: func.func @unrolled_pages
// CLASSIFY-DAG: memref.load {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY-DAG: arith.andi {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY-DAG: arith.shrsi {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY-DAG: memref.copy {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY-DAG: memref.copy {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY-DAG: tensor.insert_slice {{.*}}ssbuffer.core_type = "CUBE"
// CLASSIFY-DAG: tensor.insert_slice {{.*}}ssbuffer.core_type = "CUBE"
// MATERIAL-DAG: memref.load {{.*}}ssbuffer.core_type = "CUBE"
// MATERIAL: %[[BUFFER:.*]] = memref.alloc() {{.*}}memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
// MATERIAL: %[[ZERO:.*]] = arith.constant {{.*}} 0.000000e+00 : f16
// MATERIAL: linalg.fill {{.*}}ins(%[[ZERO]] : f16) outs(%[[BUFFER]]
// MATERIAL: memref.subview %[[BUFFER]][0, 0, 0, 0] [1, 1, 8, 16]
// MATERIAL: arith.cmpi sgt
// MATERIAL: scf.if
// MATERIAL: hivm.hir.nd2nz {{.*}}dst_continuous{{.*}}ssbuffer.core_type = "CUBE"{{.*}}memref<?x8xf16
// MATERIAL: memref.subview %[[BUFFER]][0, 0, 8, 0] [1, 1, 8, 16]
// MATERIAL: arith.cmpi sgt
// MATERIAL: scf.if
// MATERIAL: hivm.hir.nd2nz {{.*}}dst_continuous{{.*}}ssbuffer.core_type = "CUBE"
// MATERIAL: hivm.hir.convert_layout
// MATERIAL-NOT: tensor.insert_slice
// MATERIAL: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
// COMMON: return
func.func @unrolled_pages(%metadata: memref<?xi32>, %cache: memref<?xf16>, %active: i1, %q: tensor<16x16xf16>) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %mask = arith.constant 16777215 : i32
  %shift = arith.constant 24 : i32
  %padding = arith.constant -1 : i32
  %zero = arith.constant 0.0 : f16
  %zero_f32 = arith.constant 0.0 : f32
  %meta = scf.if %active -> (i32) {
    %loaded = memref.load %metadata[%c0] : memref<?xi32>
    scf.yield %loaded : i32
  } else {
    scf.yield %padding : i32
  }
  %physical = arith.andi %meta, %mask : i32
  %valid = arith.shrsi %meta, %shift : i32
  %valid_idx = arith.index_cast %valid : i32 to index
  %nonnegative = arith.maxsi %valid_idx, %c0 : index
  %size = arith.minsi %nonnegative, %c8 : index
  %physical_idx = arith.index_cast %physical : i32 to index
  %offset = arith.muli %physical_idx, %c128 : index
  %src0 = memref.reinterpret_cast %cache to offset: [%offset], sizes: [8, 16], strides: [16, 1] : memref<?xf16> to memref<8x16xf16, strided<[16, 1], offset: ?>>
  %page0 = memref.alloc() : memref<8x16xf16>
  linalg.fill ins(%zero : f16) outs(%page0 : memref<8x16xf16>)
  %src_view0 = memref.subview %src0[0, 0] [%size, 8] [1, 1] : memref<8x16xf16, strided<[16, 1], offset: ?>> to memref<?x8xf16, strided<[16, 1], offset: ?>>
  %dst_view0 = memref.subview %page0[0, 0] [%size, 8] [1, 1] : memref<8x16xf16> to memref<?x8xf16, strided<[16, 1]>>
  memref.copy %src_view0, %dst_view0 : memref<?x8xf16, strided<[16, 1], offset: ?>> to memref<?x8xf16, strided<[16, 1]>>
  %tensor0 = bufferization.to_tensor %page0 restrict writable : memref<8x16xf16> to tensor<8x16xf16>
  %next = arith.addi %offset, %c128 : index
  %src1 = memref.reinterpret_cast %cache to offset: [%next], sizes: [8, 16], strides: [16, 1] : memref<?xf16> to memref<8x16xf16, strided<[16, 1], offset: ?>>
  %page1 = memref.alloc() : memref<8x16xf16>
  memref.copy %src1, %page1 : memref<8x16xf16, strided<[16, 1], offset: ?>> to memref<8x16xf16>
  %tensor1 = bufferization.to_tensor %page1 restrict writable : memref<8x16xf16> to tensor<8x16xf16>
  %empty = tensor.empty() : tensor<16x16xf16>
  %initial = linalg.fill ins(%zero : f16) outs(%empty : tensor<16x16xf16>) -> tensor<16x16xf16>
  %insert0 = tensor.insert_slice %tensor0 into %initial[0, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<16x16xf16>
  %insert1 = tensor.insert_slice %tensor1 into %insert0[8, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<16x16xf16>
  %transposed = linalg.transpose ins(%insert1 : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>) permutation = [1, 0]
  %out = tensor.empty() : tensor<16x16xf32>
  %init = linalg.fill ins(%zero_f32 : f32) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  %qk = linalg.matmul ins(%q, %transposed : tensor<16x16xf16>, tensor<16x16xf16>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
  %pv = linalg.matmul ins(%q, %insert1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %qk, %pv : tensor<16x16xf32>, tensor<16x16xf32>
}

// -----

// A computed page is not a pure loader, even if its aggregation feeds matmul.
// COMMON-LABEL: func.func @computed_page
// COMMON: arith.mulf {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: tensor.insert_slice {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
func.func @computed_page(%page: tensor<8x16xf16>, %q: tensor<16x16xf16>) -> tensor<16x16xf32> {
  %empty = tensor.empty() : tensor<16x16xf16>
  %zero = arith.constant 0.0 : f16
  %initial = linalg.fill ins(%zero : f16) outs(%empty : tensor<16x16xf16>) -> tensor<16x16xf16>
  %computed = arith.mulf %page, %page : tensor<8x16xf16>
  %insert = tensor.insert_slice %computed into %initial[0, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<16x16xf16>
  %out = tensor.empty() : tensor<16x16xf32>
  %mm = linalg.matmul ins(%q, %insert : tensor<16x16xf16>, tensor<16x16xf16>) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mm : tensor<16x16xf32>
}

// -----

// The initial aggregate must also have a known loader provenance.
// COMMON-LABEL: func.func @unknown_aggregate
// COMMON: bufferization.to_tensor {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: tensor.insert_slice {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
func.func @unknown_aggregate(%page: memref<8x16xf16>, %initial: tensor<16x16xf16>, %q: tensor<16x16xf16>) -> tensor<16x16xf32> {
  %tensor = bufferization.to_tensor %page restrict writable : memref<8x16xf16> to tensor<8x16xf16>
  %insert = tensor.insert_slice %tensor into %initial[0, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<16x16xf16>
  %out = tensor.empty() : tensor<16x16xf32>
  %mm = linalg.matmul ins(%q, %insert : tensor<16x16xf16>, tensor<16x16xf16>) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mm : tensor<16x16xf32>
}

// -----

// Nonzero masked padding cannot be replaced by L1 zero initialization.
// COMMON-LABEL: func.func @nonzero_padding
// COMMON: tensor.insert_slice {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
func.func @nonzero_padding(%cache: memref<8x16xf16>, %q: tensor<16x16xf16>) -> tensor<16x16xf32> {
  %zero = arith.constant 1.0 : f16
  %empty = tensor.empty() : tensor<16x16xf16>
  %page0 = memref.alloc() : memref<8x16xf16>
  linalg.fill ins(%zero : f16) outs(%page0 : memref<8x16xf16>)
  %src0 = memref.subview %cache[0, 0] [4, 16] [1, 1] : memref<8x16xf16> to memref<4x16xf16, strided<[16, 1]>>
  %dst0 = memref.subview %page0[0, 0] [4, 16] [1, 1] : memref<8x16xf16> to memref<4x16xf16, strided<[16, 1]>>
  memref.copy %src0, %dst0 : memref<4x16xf16, strided<[16, 1]>> to memref<4x16xf16, strided<[16, 1]>>
  %tensor0 = bufferization.to_tensor %page0 restrict writable : memref<8x16xf16> to tensor<8x16xf16>
  %page1 = memref.alloc() : memref<8x16xf16>
  linalg.fill ins(%zero : f16) outs(%page1 : memref<8x16xf16>)
  %src1 = memref.subview %cache[0, 0] [4, 16] [1, 1] : memref<8x16xf16> to memref<4x16xf16, strided<[16, 1]>>
  %dst1 = memref.subview %page1[0, 0] [4, 16] [1, 1] : memref<8x16xf16> to memref<4x16xf16, strided<[16, 1]>>
  memref.copy %src1, %dst1 : memref<4x16xf16, strided<[16, 1]>> to memref<4x16xf16, strided<[16, 1]>>
  %tensor1 = bufferization.to_tensor %page1 restrict writable : memref<8x16xf16> to tensor<8x16xf16>
  %insert0 = tensor.insert_slice %tensor0 into %empty[0, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<16x16xf16>
  %insert1 = tensor.insert_slice %tensor1 into %insert0[8, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<16x16xf16>
  %out = tensor.empty() : tensor<16x16xf32>
  %mm = linalg.matmul ins(%q, %insert1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mm : tensor<16x16xf32>
}

// -----

// A page buffer escaping the loader cannot be removed.
// COMMON-LABEL: func.func @shared_page_buffer
// COMMON: tensor.insert_slice {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
func.func @shared_page_buffer(%cache: memref<8x16xf16>, %q: tensor<16x16xf16>) -> (tensor<16x16xf32>, memref<8x16xf16>) {
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<16x16xf16>
  %page0 = memref.alloc() : memref<8x16xf16>
  linalg.fill ins(%zero : f16) outs(%page0 : memref<8x16xf16>)
  memref.copy %cache, %page0 : memref<8x16xf16> to memref<8x16xf16>
  %tensor0 = bufferization.to_tensor %page0 restrict writable : memref<8x16xf16> to tensor<8x16xf16>
  %page1 = memref.alloc() : memref<8x16xf16>
  linalg.fill ins(%zero : f16) outs(%page1 : memref<8x16xf16>)
  memref.copy %cache, %page1 : memref<8x16xf16> to memref<8x16xf16>
  %tensor1 = bufferization.to_tensor %page1 restrict writable : memref<8x16xf16> to tensor<8x16xf16>
  %insert0 = tensor.insert_slice %tensor0 into %empty[0, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<16x16xf16>
  %insert1 = tensor.insert_slice %tensor1 into %insert0[8, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<16x16xf16>
  %out = tensor.empty() : tensor<16x16xf32>
  %mm = linalg.matmul ins(%q, %insert1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mm, %page0 : tensor<16x16xf32>, memref<8x16xf16>
}

// -----

// A later fill overwrites loaded values and must not be moved before the copy.
// COMMON-LABEL: func.func @fill_after_copy
// COMMON: tensor.insert_slice {{.*}}ssbuffer.core_type = "VECTOR"
// COMMON: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
func.func @fill_after_copy(%cache: memref<8x16xf16>, %q: tensor<16x16xf16>) -> tensor<16x16xf32> {
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<16x16xf16>
  %page0 = memref.alloc() : memref<8x16xf16>
  memref.copy %cache, %page0 : memref<8x16xf16> to memref<8x16xf16>
  linalg.fill ins(%zero : f16) outs(%page0 : memref<8x16xf16>)
  %tensor0 = bufferization.to_tensor %page0 restrict writable : memref<8x16xf16> to tensor<8x16xf16>
  %page1 = memref.alloc() : memref<8x16xf16>
  memref.copy %cache, %page1 : memref<8x16xf16> to memref<8x16xf16>
  linalg.fill ins(%zero : f16) outs(%page1 : memref<8x16xf16>)
  %tensor1 = bufferization.to_tensor %page1 restrict writable : memref<8x16xf16> to tensor<8x16xf16>
  %insert0 = tensor.insert_slice %tensor0 into %empty[0, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<16x16xf16>
  %insert1 = tensor.insert_slice %tensor1 into %insert0[8, 0] [8, 16] [1, 1] : tensor<8x16xf16> into tensor<16x16xf16>
  %out = tensor.empty() : tensor<16x16xf32>
  %mm = linalg.matmul ins(%q, %insert1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mm : tensor<16x16xf32>
}
