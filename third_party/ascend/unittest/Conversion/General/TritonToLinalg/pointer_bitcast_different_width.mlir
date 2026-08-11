// RUN: triton-opt --triton-to-unstructure --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s --implicit-check-not="arith.divsi" --implicit-check-not="arith.remsi" --implicit-check-not="tt.assert" --implicit-check-not="triton_assert" --implicit-check-not="builtin.unrealized_conversion_cast" --implicit-check-not="tt.pointer_bitcast_rescaled_offset" --implicit-check-not="tt.pointer_bitcast_offset_divisor"

// CHECK-LABEL: func.func @widen_scalar_multi_addptr
// CHECK: arith.addi
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xbf16>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @widen_scalar_multi_addptr(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %block: i64,
      %position: i64,
      %block_stride: i64 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c576_i64 = arith.constant 576 : i64
    %c448_i64 = arith.constant 448 : i64
    %block_offset = arith.muli %block, %block_stride : i64
    %block_base = tt.addptr %src, %block_offset : !tt.ptr<i8>, i64
    %token_offset = arith.muli %position, %c576_i64 : i64
    %token_base = tt.addptr %block_base, %token_offset : !tt.ptr<i8>, i64
    %rope_bytes = tt.addptr %token_base, %c448_i64 : !tt.ptr<i8>, i64
    %rope_bf16 = tt.bitcast %rope_bytes : !tt.ptr<i8> -> !tt.ptr<bf16>
    %value = tt.load %rope_bf16 : !tt.ptr<bf16>
    tt.store %dst, %value : !tt.ptr<bf16>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @widen_tensor_offset
// CHECK: scf.for
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xi32>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @widen_tensor_offset(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %four = arith.constant dense<4> : tensor<8xi32>
    %offsets = arith.muli %range, %four : tensor<8xi32>
    %srcs = tt.splat %src : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
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

// CHECK-LABEL: func.func @narrow_scalar_pointer
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xi8>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @narrow_scalar_pointer(
      %src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %offset: i32) attributes {noinline = false} {
    %source_element = tt.addptr %src, %offset : !tt.ptr<i32>, i32
    %byte_ptr = tt.bitcast %source_element : !tt.ptr<i32> -> !tt.ptr<i8>
    %value = tt.load %byte_ptr : !tt.ptr<i8>
    tt.store %dst, %value : !tt.ptr<i8>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @direct_base_pointer
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xbf16>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @direct_base_pointer(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %bf16_ptr = tt.bitcast %src : !tt.ptr<i8> -> !tt.ptr<bf16>
    %value = tt.load %bf16_ptr : !tt.ptr<bf16>
    tt.store %dst, %value : !tt.ptr<bf16>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @widen_pointer_inside_loop
// CHECK: scf.for
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xbf16>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @widen_pointer_inside_loop(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %count: i64) attributes {noinline = false} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c576_i64 = arith.constant 576 : i64
    %c448_i64 = arith.constant 448 : i64
    scf.for %position = %c0_i64 to %count step %c1_i64 : i64 {
      %block_offset = arith.muli %position, %c1024_i64 : i64
      %block_base = tt.addptr %src, %block_offset : !tt.ptr<i8>, i64
      %token_offset = arith.muli %position, %c576_i64 : i64
      %token_base = tt.addptr %block_base, %token_offset : !tt.ptr<i8>, i64
      %rope_bytes = tt.addptr %token_base, %c448_i64 : !tt.ptr<i8>, i64
      %rope_bf16 = tt.bitcast %rope_bytes : !tt.ptr<i8> -> !tt.ptr<bf16>
      %value = tt.load %rope_bf16 : !tt.ptr<bf16>
      %dst_ptr = tt.addptr %dst, %position : !tt.ptr<bf16>, i64
      tt.store %dst_ptr, %value : !tt.ptr<bf16>
    }
    tt.return
  }
}
