// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// TRITON_DEBUG=1 makes the frontend insert an integer-overflow-check
// prologue (widen to i64, compare against i32 bounds, tt.assert) ahead of
// the real arith.addi inside a tt.scan combine body. ScanConverter must
// still recognize arith.addi as the sole real reduction op and lower to the
// triton_cumsum library call, instead of mis-selecting one of the prologue
// ops (e.g. arith.extsi, which is the first op in program order) and
// falling back to the slow generic associative-scan expansion.
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
