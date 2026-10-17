// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=true compile-mode=simd_simt_template" --split-input-file %s | FileCheck %s --check-prefix=A5
// RUN: triton-opt --triton-to-linalg="named-ops=True compile-on-910-95=true compile-mode=simd_simt" --split-input-file %s | FileCheck %s --check-prefix=A5

// TRITON_DEBUG=1 makes the frontend insert an integer-overflow-check
// prologue (widen to i64, compare against i32 bounds, tt.assert) ahead of
// the real arith.addi inside a tt.scan combine body. ScanConverter must
// still recognize arith.addi as the sole real reduction op and lower to the
// triton_cumsum library call, instead of mis-selecting one of the prologue
// ops (e.g. arith.extsi, which is the first op in program order) and
// falling back to the slow generic associative-scan expansion.
// A5 additionally needs MIX launcher metadata for the SIMT cumsum template.
// A5-LABEL: func.func @cumsum_with_debug_assert
// A5-SAME: parallel_mode = "mix_simd_simt"
// A5: call @triton_cumsum
// CHECK-LABEL: func.func @cumsum_with_debug_assert
// CHECK: call @triton_cumsum
// CHECK-NOT: memref.alloc
// CHECK-NOT: bufferization.to_buffer
tt.func public @cumsum_with_debug_assert(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %off = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %in_splat = tt.splat %in : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %in_ptrs = tt.addptr %in_splat, %off : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %x = tt.load %in_ptrs : tensor<128x!tt.ptr<i32>>
  %0 = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg0: i32, %arg1: i32):
    %lhs64 = arith.extsi %arg0 : i32 to i64
    %rhs64 = arith.extsi %arg1 : i32 to i64
    %sum64 = arith.addi %lhs64, %rhs64 : i64
    %max = arith.constant 2147483647 : i64
    %min = arith.constant -2147483648 : i64
    %le = arith.cmpi sle, %sum64, %max : i64
    %ge = arith.cmpi sge, %sum64, %min : i64
    %cond = arith.andi %le, %ge : i1
    tt.assert %cond, "int32 overflow detected for operation add" : i1
    %res = arith.addi %arg0, %arg1 : i32
    tt.scan.return %res : i32
  }) : (tensor<128xi32>) -> tensor<128xi32>
  %out_splat = tt.splat %out : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %out_ptrs = tt.addptr %out_splat, %off : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  tt.store %out_ptrs, %0 : tensor<128x!tt.ptr<i32>>
  tt.return
}

// -----

// Control: the same cumsum WITHOUT the debug-assert prologue must still
// take the triton_cumsum fast path. Guards against a fix that only works
// when the assert prologue is present.
// A5-LABEL: func.func @cumsum_without_debug_assert
// A5-SAME: parallel_mode = "mix_simd_simt"
// A5: call @triton_cumsum
// CHECK-LABEL: func.func @cumsum_without_debug_assert
// CHECK: call @triton_cumsum
// CHECK-NOT: memref.alloc
// CHECK-NOT: bufferization.to_buffer
tt.func public @cumsum_without_debug_assert(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %off = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %in_splat = tt.splat %in : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %in_ptrs = tt.addptr %in_splat, %off : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %x = tt.load %in_ptrs : tensor<128x!tt.ptr<i32>>
  %0 = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg0: i32, %arg1: i32):
    %res = arith.addi %arg0, %arg1 : i32
    tt.scan.return %res : i32
  }) : (tensor<128xi32>) -> tensor<128xi32>
  %out_splat = tt.splat %out : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %out_ptrs = tt.addptr %out_splat, %off : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  tt.store %out_ptrs, %0 : tensor<128x!tt.ptr<i32>>
  tt.return
}

// -----

// A product scan stays SIMD even with a dead debug-check prologue.
// CHECK-LABEL: func.func @cumprod_with_debug_assert
// CHECK: call @triton_cumprod
// A5-LABEL: func.func @cumprod_with_debug_assert
// A5-SAME: parallel_mode = "simd"
// A5: call @triton_cumprod
tt.func public @cumprod_with_debug_assert(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %off = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %in_splat = tt.splat %in : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %in_ptrs = tt.addptr %in_splat, %off : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %x = tt.load %in_ptrs : tensor<128x!tt.ptr<i32>>
  %0 = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg0: i32, %arg1: i32):
    %lhs64 = arith.extsi %arg0 : i32 to i64
    %rhs64 = arith.extsi %arg1 : i32 to i64
    %product64 = arith.muli %lhs64, %rhs64 : i64
    %max = arith.constant 2147483647 : i64
    %le = arith.cmpi sle, %product64, %max : i64
    tt.assert %le, "int32 overflow detected for operation mul" : i1
    %res = arith.muli %arg0, %arg1 : i32
    tt.scan.return %res : i32
  }) : (tensor<128xi32>) -> tensor<128xi32>
  %out_splat = tt.splat %out : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %out_ptrs = tt.addptr %out_splat, %off : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  tt.store %out_ptrs, %0 : tensor<128x!tt.ptr<i32>>
  tt.return
}

// -----

// Unit-sized non-scan axes collapse to the same 1-D SIMT template.
// CHECK-LABEL: func.func @cumsum_unit_axis
// CHECK: call @triton_cumsum
// A5-LABEL: func.func @cumsum_unit_axis
// A5-SAME: parallel_mode = "mix_simd_simt"
// A5: call @triton_cumsum
tt.func public @cumsum_unit_axis(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %off = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %in_splat = tt.splat %in : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %in_ptrs = tt.addptr %in_splat, %off : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %x = tt.load %in_ptrs : tensor<128x!tt.ptr<i32>>
  %expanded = tt.expand_dims %x {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %scan = "tt.scan"(%expanded) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %res = arith.addi %lhs, %rhs : i32
    tt.scan.return %res : i32
  }) : (tensor<1x128xi32>) -> tensor<1x128xi32>
  %result = tt.reshape %scan : tensor<1x128xi32> -> tensor<128xi32>
  %out_splat = tt.splat %out : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %out_ptrs = tt.addptr %out_splat, %off : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  tt.store %out_ptrs, %result : tensor<128x!tt.ptr<i32>>
  tt.return
}

// -----

// A non-unit non-scan axis retains the existing SIMD classification.
// CHECK-LABEL: func.func @cumsum_nonunit_axis
// CHECK: call @triton_cumsum
// A5-LABEL: func.func @cumsum_nonunit_axis
// A5-SAME: parallel_mode = "simd"
// A5: call @triton_cumsum
tt.func public @cumsum_nonunit_axis(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
  %off = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  %in_splat = tt.splat %in : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
  %in_ptrs = tt.addptr %in_splat, %off : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
  %x = tt.load %in_ptrs : tensor<256x!tt.ptr<i32>>
  %matrix = tt.reshape %x : tensor<256xi32> -> tensor<2x128xi32>
  %scan = "tt.scan"(%matrix) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %res = arith.addi %lhs, %rhs : i32
    tt.scan.return %res : i32
  }) : (tensor<2x128xi32>) -> tensor<2x128xi32>
  %result = tt.reshape %scan : tensor<2x128xi32> -> tensor<256xi32>
  %out_splat = tt.splat %out : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
  %out_ptrs = tt.addptr %out_splat, %off : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
  tt.store %out_ptrs, %result : tensor<256x!tt.ptr<i32>>
  tt.return
}
