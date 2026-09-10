// RUN: triton-opt --split-input-file --materialize-cube-page-loaders --reorder-ops-by-block-id --verify-each %s | FileCheck %s --check-prefix=MATERIAL
// RUN: triton-opt --split-input-file --materialize-cube-page-loaders --reorder-ops-by-block-id --clone-ops --verify-each %s | FileCheck %s --check-prefix=CLONE --implicit-check-not=triton_ascend.dynamic_cv_pipeline.rc
// RUN: triton-opt --materialize-cube-page-loaders --reorder-ops-by-block-id --add-control-flow-condition --verify-each %s | FileCheck %s --check-prefix=CONDITION --implicit-check-not=triton_ascend.dynamic_cv_pipeline.rc

// RUN: sed 's@// CLOBBER_V@memref.store %%zero, %%v[%%c0] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16>@' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=UNSHARED

// RUN: sed '/%%meta0 = memref.load/s/{ssbuffer.block_id/{volatile, ssbuffer.block_id/' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=UNSHARED

// A nested read cannot be stored outside its branch.
// RUN: sed '/%%meta0 = memref.load/s/\(%%meta0 = \)\(memref.load.*\)/\1scf.if %%enabled -> (i32) { %%nested = \2 scf.yield %%nested : i32 } else { scf.yield %%shift : i32 } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"}/' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=UNSHARED

// RUN: sed '1i module attributes {ssbuffer.reserved_bytes = 1024 : i64} {' %s | sed '$a }' | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=UNSHARED

// UNSHARED-LABEL: func.func @shared_pages
// UNSHARED-NOT: ssbuffer.shared_page_write
// UNSHARED: return

// RUN: sed 's/%%c0 to %%c3 step %%c1/%%c1 to %%c8 step %%c2/' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=RING

// Nonzero lower bounds and non-unit steps select slots by trip number.
// RING-LABEL: func.func @shared_pages
// RING: %[[ONE:.*]] = arith.constant 1 : index
// RING: %[[TWO:.*]] = arith.constant 2 : index
// RING: scf.for %[[IV:.*]] = %[[ONE]] to %{{.*}} step %[[TWO]]
// RING: %[[REL:.*]] = arith.subi %[[IV]], %[[ONE]]
// RING: %[[ITER:.*]] = arith.divui %[[REL]], %[[TWO]]
// RING: arith.remui %[[ITER]]
// RING: linalg.matmul
// RING: %[[CREL:.*]] = arith.subi %[[IV]], %[[ONE]]
// RING: %[[CITER:.*]] = arith.divui %[[CREL]], %[[TWO]]
// RING: arith.remui %[[CITER]]
// RING: linalg.matmul

// Cache the raw i32 page metadata in two slots. QK can prepare the next K
// while PV independently loads V from the metadata for its own iteration.
// MATERIAL-LABEL: func.func @shared_pages
// MATERIAL: %[[CACHE:.*]] = hivm.hir.pointer_cast(%{{.*}}) {{.*}}memref<2x2xi32, #hivm.address_space<ssbuf>>
// MATERIAL: scf.for
// MATERIAL: arith.subi
// MATERIAL: arith.divui
// MATERIAL: arith.remui
// MATERIAL: scf.if
// MATERIAL: hivm.hir.nd2nz {{.*}}ssbuffer.block_id = 1 : i32
// MATERIAL: memref.store %{{.*}}, %[[CACHE]]{{.*}}ssbuffer.intraDeps = [0 : i32, 1 : i32]{{.*}}ssbuffer.shared_page_slots = 2 : i32{{.*}}ssbuffer.shared_page_write
// MATERIAL: scf.if
// MATERIAL: hivm.hir.nd2nz {{.*}}ssbuffer.block_id = 1 : i32
// MATERIAL: memref.store %{{.*}}, %[[CACHE]]
// MATERIAL: linalg.matmul {{.*}}ssbuffer.block_id = 1 : i32
// MATERIAL: arith.remui
// MATERIAL: memref.load %[[CACHE]]{{.*}}ssbuffer.intraDeps = [0 : i32, 0 : i32]
// MATERIAL: hivm.hir.nd2nz {{.*}}ssbuffer.block_id = 3 : i32
// MATERIAL: memref.load %[[CACHE]]
// MATERIAL: hivm.hir.nd2nz {{.*}}ssbuffer.block_id = 3 : i32
// MATERIAL: linalg.matmul {{.*}}ssbuffer.block_id = 3 : i32
// CLONE-LABEL: func.func @shared_pages(
// CLONE-SAME: %[[META:[^:]+]]:
// CLONE: %[[CACHE:.*]] = hivm.hir.pointer_cast
// CLONE: scf.for
// CLONE: memref.load %[[META]]
// CLONE-NOT: memref.load
// CLONE: hivm.hir.nd2nz
// CLONE: memref.load %[[META]]
// CLONE-NOT: memref.load
// CLONE: hivm.hir.nd2nz
// CLONE: linalg.matmul {{.*}}ssbuffer.block_id = 1 : i32
// CLONE-NOT: memref.load %[[META]]
// CLONE: memref.load %[[CACHE]]
// CLONE: hivm.hir.nd2nz
// CLONE-NOT: memref.load %[[META]]
// CLONE: memref.load %[[CACHE]]
// CLONE: hivm.hir.nd2nz
// CLONE-NOT: memref.load %[[META]]
// CLONE: linalg.matmul {{.*}}ssbuffer.block_id = 3 : i32

// CONDITION-LABEL: func.func @shared_pages
// CONDITION: scf.for {{.*}}iter_args(%[[PITER:[^ ]+]] = %{{.*}}, %[[CITER:[^ ]+]] = %{{.*}}, %[[COUNT:[^ ]+]] = %{{.*}}) -> (index, index, i32)
// CONDITION: %[[CAPACITY:.*]] = arith.constant 2 : i32
// CONDITION: arith.cmpi ne, %[[COUNT]], %[[CAPACITY]] : i32
// CONDITION: %[[PRODUCED:.*]]:2 = scf.if
// CONDITION: arith.subi %[[PITER]],
// CONDITION: arith.remui
// CONDITION: linalg.matmul {{.*}}ssbuffer.block_id = 1 : i32
// CONDITION: arith.addi %[[COUNT]], %{{.*}} : i32
// CONDITION: ssbuffer.if = 1 : i32
// CONDITION: arith.cmpi sgt, %[[PRODUCED]]#{{[01]}}, %{{.*}} : i32
// CONDITION: arith.subi %[[CITER]],
// CONDITION: arith.remui
// CONDITION: linalg.matmul {{.*}}ssbuffer.block_id = 3 : i32
// CONDITION: arith.subi %[[PRODUCED]]#{{[01]}}, %{{.*}} : i32
// CONDITION: ssbuffer.if = 3 : i32

func.func @shared_pages(%metadata: memref<?xi32>, %k: memref<?xf16>, %v: memref<?xf16>, %q: tensor<16x16xf16>, %p: tensor<16x16xf16>, %enabled: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %mask = arith.constant 16777215 : i32
  %shift = arith.constant 24 : i32
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<16x16xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<16x16xf16>) -> tensor<16x16xf16>
  %acc = arith.constant dense<0.0> : tensor<16x16xf32>
  scope.scope : () -> () {
    scf.for %iv = %c0 to %c3 step %c1 {
      %slot0 = arith.muli %iv, %c2 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
      %meta0 = memref.load %metadata[%slot0] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32>
      %physical0 = arith.andi %meta0, %mask {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
      %valid0 = arith.shrsi %meta0, %shift {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
      %rows0 = arith.index_cast %valid0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %nonnegative0 = arith.maxsi %rows0, %c0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
      %size0 = arith.minsi %nonnegative0, %c8 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
      %physical_idx0 = arith.index_cast %physical0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %offset0 = arith.muli %physical_idx0, %c128 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
      %srck0 = memref.reinterpret_cast %k to offset: [%offset0], sizes: [8, 16], strides: [16, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<8x16xf16, strided<[16, 1], offset: ?>>
      %pagek0 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16>
      linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%zero : f16) outs(%pagek0 : memref<8x16xf16>)
      %src_viewk0 = memref.subview %srck0[0, 0] [%size0, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %dst_viewk0 = memref.subview %pagek0[0, 0] [%size0, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16> to memref<?x16xf16, strided<[16, 1]>>
      memref.copy %src_viewk0, %dst_viewk0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1]>>
      %tensork0 = bufferization.to_tensor %pagek0 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16> to tensor<8x16xf16>
      %insertk0 = tensor.insert_slice %tensork0 into %init[0, 0] [8, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<8x16xf16> into tensor<16x16xf16>
      %slot1 = arith.addi %slot0, %c1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
      %meta1 = memref.load %metadata[%slot1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32>
      %physical1 = arith.andi %meta1, %mask {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
      %valid1 = arith.shrsi %meta1, %shift {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
      %rows1 = arith.index_cast %valid1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %nonnegative1 = arith.maxsi %rows1, %c0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
      %size1 = arith.minsi %nonnegative1, %c8 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
      %physical_idx1 = arith.index_cast %physical1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %offset1 = arith.muli %physical_idx1, %c128 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
      %srck1 = memref.reinterpret_cast %k to offset: [%offset1], sizes: [8, 16], strides: [16, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<8x16xf16, strided<[16, 1], offset: ?>>
      %pagek1 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16>
      linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%zero : f16) outs(%pagek1 : memref<8x16xf16>)
      %src_viewk1 = memref.subview %srck1[0, 0] [%size1, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %dst_viewk1 = memref.subview %pagek1[0, 0] [%size1, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16> to memref<?x16xf16, strided<[16, 1]>>
      memref.copy %src_viewk1, %dst_viewk1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1]>>
      %tensork1 = bufferization.to_tensor %pagek1 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16> to tensor<8x16xf16>
      %insertk1 = tensor.insert_slice %tensork1 into %insertk0[8, 0] [8, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<8x16xf16> into tensor<16x16xf16>
      %mmk = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %insertk1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      // CLOBBER_V
      %srcv0 = memref.reinterpret_cast %v to offset: [%offset0], sizes: [8, 16], strides: [16, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<8x16xf16, strided<[16, 1], offset: ?>>
      %pagev0 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16>
      linalg.fill {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%zero : f16) outs(%pagev0 : memref<8x16xf16>)
      %src_viewv0 = memref.subview %srcv0[0, 0] [%size0, 16] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %dst_viewv0 = memref.subview %pagev0[0, 0] [%size0, 16] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16> to memref<?x16xf16, strided<[16, 1]>>
      memref.copy %src_viewv0, %dst_viewv0 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1]>>
      %tensorv0 = bufferization.to_tensor %pagev0 restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16> to tensor<8x16xf16>
      %insertv0 = tensor.insert_slice %tensorv0 into %init[0, 0] [8, 16] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<8x16xf16> into tensor<16x16xf16>
      %srcv1 = memref.reinterpret_cast %v to offset: [%offset1], sizes: [8, 16], strides: [16, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<8x16xf16, strided<[16, 1], offset: ?>>
      %pagev1 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16>
      linalg.fill {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%zero : f16) outs(%pagev1 : memref<8x16xf16>)
      %src_viewv1 = memref.subview %srcv1[0, 0] [%size1, 16] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %dst_viewv1 = memref.subview %pagev1[0, 0] [%size1, 16] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16> to memref<?x16xf16, strided<[16, 1]>>
      memref.copy %src_viewv1, %dst_viewv1 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16, strided<[16, 1]>>
      %tensorv1 = bufferization.to_tensor %pagev1 restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<8x16xf16> to tensor<8x16xf16>
      %insertv1 = tensor.insert_slice %tensorv1 into %insertv0[8, 0] [8, 16] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<8x16xf16> into tensor<16x16xf16>
      %mmv = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%p, %insertv1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    } {ssbuffer.block_id = 0 : i32, ssbuffer.main_loop = 0 : i32}
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  return
}
