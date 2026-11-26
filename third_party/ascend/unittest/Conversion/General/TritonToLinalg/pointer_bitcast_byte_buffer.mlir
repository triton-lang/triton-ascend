// RUN: triton-opt --triton-to-unstructure --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-unstructure --split-input-file %s | FileCheck %s --check-prefix=UNSTRUCTURE
// RUN: triton-opt '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' --split-input-file %s | FileCheck %s --check-prefix=UNSTRUCTURE
// RUN: triton-opt '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-linalg=compile-on-910-95=True compile-mode=simd_simt_template' --split-input-file %s | FileCheck %s --check-prefix=SIMT-LINALG

// The scalar byte offset 1 and lane offsets 63 + 4*i address i32 elements 16+i.
// CHECK-LABEL: func.func @widen_tensor_offset
// CHECK: %[[BASE:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<?xi8> -> index
// CHECK: %[[ADDRESS:.*]] = arith.index_cast %[[BASE]] : index to i64
// CHECK: %[[CAPACITY:.*]] = arith.constant 24 : index
// CHECK: %[[PTR:.*]] = hivm.hir.pointer_cast(%[[ADDRESS]]) [%[[CAPACITY]]] : memref<?xi32>
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[PTR]] to offset: [16], sizes: [8], strides: [1]
// CHECK: memref.copy %[[VIEW]], {{.*}} : memref<8xi32, strided<[1], offset: 16>> to memref<8xi32>
// CHECK: return
// UNSTRUCTURE-LABEL: tt.func public @widen_tensor_offset
// UNSTRUCTURE: %[[SHIFTED:.*]] = tt.addptr {{.*}} : !tt.ptr<i8>, i32
// UNSTRUCTURE: %[[BASES:.*]] = tt.splat %[[SHIFTED]]
// UNSTRUCTURE: %[[BYTES:.*]] = tt.addptr %[[BASES]], {{.*}} : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
// UNSTRUCTURE: %[[WORDS:.*]] = tt.bitcast %[[BYTES]] : tensor<8x!tt.ptr<i8>> -> tensor<8x!tt.ptr<i32>>
// UNSTRUCTURE: tt.load %[[WORDS]]
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @widen_tensor_offset(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %four = arith.constant dense<4> : tensor<8xi32>
    %scaled = arith.muli %range, %four : tensor<8xi32>
    %bias = arith.constant dense<63> : tensor<8xi32>
    %offsets = arith.addi %scaled, %bias : tensor<8xi32>
    %one = arith.constant 1 : i32
    %shifted = tt.addptr %src, %one : !tt.ptr<i8>, i32
    %srcs = tt.splat %shifted : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %byte_ptrs = tt.addptr %srcs, %offsets : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %i32_ptrs = tt.bitcast %byte_ptrs : tensor<8x!tt.ptr<i8>> -> tensor<8x!tt.ptr<i32>>
    %values = tt.load %i32_ptrs : tensor<8x!tt.ptr<i32>>
    %dsts = tt.splat %dst : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %dst_ptrs = tt.addptr %dsts, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %dst_ptrs, %values : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @widen_scalar_multi_addptr
// CHECK: arith.addi
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xbf16>
// CHECK: return
// UNSTRUCTURE-LABEL: tt.func public @widen_scalar_multi_addptr
// UNSTRUCTURE: %[[BLOCK:.*]] = tt.addptr {{.*}} : !tt.ptr<i8>, i64
// UNSTRUCTURE: %[[TOKEN:.*]] = tt.addptr %[[BLOCK]],
// UNSTRUCTURE: %[[FIELD:.*]] = tt.addptr %[[TOKEN]],
// UNSTRUCTURE: %[[BF16:.*]] = tt.bitcast %[[FIELD]] : !tt.ptr<i8> -> !tt.ptr<bf16>
// UNSTRUCTURE: %[[DST_BASES:.*]] = tt.splat %[[BF16]]
// UNSTRUCTURE: %[[DST:.*]] = tt.addptr %[[DST_BASES]],
// UNSTRUCTURE: tt.store %[[DST]],
// SIMT-LINALG-LABEL: func.func @widen_scalar_multi_addptr
// SIMT-LINALG-NOT: tt.int_to_ptr
// SIMT-LINALG: memref.extract_aligned_pointer_as_index
// SIMT-LINALG: memref.reinterpret_cast
// SIMT-LINALG: bufferization.materialize_in_destination
// SIMT-LINALG-NOT: tt.int_to_ptr
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @widen_scalar_multi_addptr(
      %values: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %cache: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %block: i64,
      %position: i64,
      %block_stride: i64 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %c576_i64 = arith.constant 576 : i64
    %c448_i64 = arith.constant 448 : i64
    %block_offset = arith.muli %block, %block_stride : i64
    %block_base = tt.addptr %cache, %block_offset : !tt.ptr<i8>, i64
    %token_offset = arith.muli %position, %c576_i64 : i64
    %token_base = tt.addptr %block_base, %token_offset : !tt.ptr<i8>, i64
    %rope_bytes = tt.addptr %token_base, %c448_i64 : !tt.ptr<i8>, i64
    %rope_bf16 = tt.bitcast %rope_bytes : !tt.ptr<i8> -> !tt.ptr<bf16>
    %rope_bases = tt.splat %rope_bf16 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
    %rope_ptrs = tt.addptr %rope_bases, %range : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    %value_bases = tt.splat %values : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
    %value_ptrs = tt.addptr %value_bases, %range : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    %loaded = tt.load %value_ptrs : tensor<8x!tt.ptr<bf16>>
    tt.store %rope_ptrs, %loaded : tensor<8x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @widen_pointer_inside_loop
// CHECK: scf.for
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xbf16>
// CHECK: return
// UNSTRUCTURE-LABEL: tt.func public @widen_pointer_inside_loop
// UNSTRUCTURE: scf.for
// UNSTRUCTURE: %[[LOOP_BYTES:.*]] = tt.addptr {{.*}} : !tt.ptr<i8>, i64
// UNSTRUCTURE: %[[LOOP_FIELD:.*]] = tt.addptr %[[LOOP_BYTES]],
// UNSTRUCTURE: %[[LOOP_BF16:.*]] = tt.bitcast %[[LOOP_FIELD]] : !tt.ptr<i8> -> !tt.ptr<bf16>
// UNSTRUCTURE: %[[LOOP_BASES:.*]] = tt.splat %[[LOOP_BF16]]
// UNSTRUCTURE: %[[LOOP_PTRS:.*]] = tt.addptr %[[LOOP_BASES]],
// UNSTRUCTURE: tt.load %[[LOOP_PTRS]]
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @widen_pointer_inside_loop(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %count: i64) attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c576_i64 = arith.constant 576 : i64
    %c448_i64 = arith.constant 448 : i64
    scf.for %position = %c0_i64 to %count step %c1_i64 : i64 {
      %token_offset = arith.muli %position, %c576_i64 : i64
      %token_base = tt.addptr %src, %token_offset : !tt.ptr<i8>, i64
      %rope_bytes = tt.addptr %token_base, %c448_i64 : !tt.ptr<i8>, i64
      %rope_bf16 = tt.bitcast %rope_bytes : !tt.ptr<i8> -> !tt.ptr<bf16>
      %rope_bases = tt.splat %rope_bf16 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
      %rope_ptrs = tt.addptr %rope_bases, %range : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
      %values = tt.load %rope_ptrs : tensor<8x!tt.ptr<bf16>>
      %dsts = tt.splat %dst : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
      %dst_ptrs = tt.addptr %dsts, %range : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
      tt.store %dst_ptrs, %values : tensor<8x!tt.ptr<bf16>>
    }
    tt.return
  }
}

// -----

// A caller-supplied even origin remains usable as a dynamic memref offset.
// CHECK-LABEL: func.func @widen_dynamic_origin
// CHECK: memref.reinterpret_cast {{.*}} to offset: [%{{.*}}], sizes: [8], strides: [1]
// CHECK: memref.copy
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @widen_dynamic_origin(%src: !tt.ptr<i16>, %dst: !tt.ptr<i32>,
                                       %start: i32) attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %two = arith.constant dense<2> : tensor<8xi32>
    %scaled = arith.muli %range, %two : tensor<8xi32>
    %shifted = tt.addptr %src, %start : !tt.ptr<i16>, i32
    %srcs = tt.splat %shifted : !tt.ptr<i16> -> tensor<8x!tt.ptr<i16>>
    %ptrs = tt.addptr %srcs, %scaled : tensor<8x!tt.ptr<i16>>, tensor<8xi32>
    %wide = tt.bitcast %ptrs : tensor<8x!tt.ptr<i16>> -> tensor<8x!tt.ptr<i32>>
    %values = tt.load %wide : tensor<8x!tt.ptr<i32>>
    %dsts = tt.splat %dst : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %out = tt.addptr %dsts, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %out, %values : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
