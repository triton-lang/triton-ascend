// RUN: triton-opt --triton-to-unstructure --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s --implicit-check-not="arith.divsi" --implicit-check-not="arith.remsi" --implicit-check-not="tt.assert" --implicit-check-not="triton_assert" --implicit-check-not="builtin.unrealized_conversion_cast" --implicit-check-not="tt.pointer_bitcast_rescaled_offset" --implicit-check-not="tt.pointer_bitcast_offset_divisor"

// CHECK-LABEL: func.func @dynamic_scalar_address
// CHECK-NOT: arith.remsi
// CHECK-NOT: arith.divsi
// CHECK-NOT: triton_assert
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xi32>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @dynamic_scalar_address(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %pre_bytes: i32,
      %post_elements: i32) attributes {noinline = false} {
    %byte_ptr = tt.addptr %src, %pre_bytes : !tt.ptr<i8>, i32
    %wide_ptr = tt.bitcast %byte_ptr : !tt.ptr<i8> -> !tt.ptr<i32>
    %final_ptr = tt.addptr %wide_ptr, %post_elements : !tt.ptr<i32>, i32
    %value = tt.load %final_ptr : !tt.ptr<i32>
    tt.store %dst, %value : !tt.ptr<i32>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @tensor_offsets_before_and_after_bitcast
// CHECK-NOT: arith.remsi
// CHECK-NOT: arith.divsi
// CHECK-NOT: triton_assert
// CHECK: scf.for
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xi32>
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @tensor_offsets_before_and_after_bitcast(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %offset_src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
      attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 4 : i32}
        : tensor<4xi32>
    %four = arith.constant dense<4> : tensor<4xi32>
    %pre_bytes = arith.muli %range, %four : tensor<4xi32>
    %srcs = tt.splat %src : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %byte_ptrs = tt.addptr %srcs, %pre_bytes
        : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %wide_ptrs = tt.bitcast %byte_ptrs
        : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
    %offset_bases = tt.splat %offset_src
        : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %offset_ptrs = tt.addptr %offset_bases, %range
        : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %post_elements = tt.load %offset_ptrs : tensor<4x!tt.ptr<i32>>
    %final_ptrs = tt.addptr %wide_ptrs, %post_elements
        : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %values = tt.load %final_ptrs : tensor<4x!tt.ptr<i32>>
    %dsts = tt.splat %dst : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %dst_ptrs = tt.addptr %dsts, %range
        : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @multiple_address_boundaries
// CHECK-NOT: arith.remsi
// CHECK-NOT: arith.divsi
// CHECK-NOT: triton_assert
// CHECK-COUNT-2: arith.muli
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xi32>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @multiple_address_boundaries(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %pre_bytes: i32,
      %half_elements: i32,
      %word_elements: i32) attributes {noinline = false} {
    %byte_ptr = tt.addptr %src, %pre_bytes : !tt.ptr<i8>, i32
    %half_ptr = tt.bitcast %byte_ptr : !tt.ptr<i8> -> !tt.ptr<i16>
    %half_offset_ptr = tt.addptr %half_ptr, %half_elements
        : !tt.ptr<i16>, i32
    %wide_ptr = tt.bitcast %half_offset_ptr
        : !tt.ptr<i16> -> !tt.ptr<i32>
    %final_ptr = tt.addptr %wide_ptr, %word_elements
        : !tt.ptr<i32>, i32
    %value = tt.load %final_ptr : !tt.ptr<i32>
    tt.store %dst, %value : !tt.ptr<i32>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @static_pre_and_post_offsets_preserve_address
// CHECK-NOT: arith.remsi
// CHECK-NOT: arith.divsi
// CHECK-NOT: triton_assert
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xbf16>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @static_pre_and_post_offsets_preserve_address(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32})
      attributes {noinline = false} {
    %four_bytes = arith.constant 4 : i32
    %one_element = arith.constant 1 : i32
    %byte_ptr = tt.addptr %src, %four_bytes : !tt.ptr<i8>, i32
    %bf16_ptr = tt.bitcast %byte_ptr : !tt.ptr<i8> -> !tt.ptr<bf16>
    %final_ptr = tt.addptr %bf16_ptr, %one_element
        : !tt.ptr<bf16>, i32
    %value = tt.load %final_ptr : !tt.ptr<bf16>
    tt.store %dst, %value : !tt.ptr<bf16>
    tt.return
  }
}

// -----

// A non-divisible byte address is intentionally not diagnosed. This is a
// compile-only contract: executing an unaligned device access is the caller's
// responsibility.
// CHECK-LABEL: func.func @static_nondivisible_offset_is_unchecked
// CHECK: arith.addi
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?xi32>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @static_nondivisible_offset_is_unchecked(
      %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
      attributes {noinline = false} {
    %one_byte = arith.constant 1 : i32
    %byte_ptr = tt.addptr %src, %one_byte : !tt.ptr<i8>, i32
    %wide_ptr = tt.bitcast %byte_ptr : !tt.ptr<i8> -> !tt.ptr<i32>
    %value = tt.load %wide_ptr : !tt.ptr<i32>
    tt.store %dst, %value : !tt.ptr<i32>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @value_bitcast_is_unchanged
// CHECK: arith.bitcast {{.*}} : tensor<4xi32> to tensor<4xf32>
// CHECK-NOT: triton_assert
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @value_bitcast_is_unchanged(
      %src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32})
      attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 4 : i32}
        : tensor<4xi32>
    %src_bases = tt.splat %src
        : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %src_ptrs = tt.addptr %src_bases, %range
        : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %values = tt.load %src_ptrs : tensor<4x!tt.ptr<i32>>
    %bits = tt.bitcast %values : tensor<4xi32> -> tensor<4xf32>
    %dst_bases = tt.splat %dst
        : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_bases, %range
        : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %dst_ptrs, %bits : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
