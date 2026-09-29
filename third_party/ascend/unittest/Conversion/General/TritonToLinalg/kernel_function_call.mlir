// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s
// RUN: triton-opt --triton-to-linalg --split-input-file %s 2>&1 | FileCheck %s --check-prefix=WARN

// Unit tests for kernel-function-call lowering in TritonToLinalg:
// tt.call with multiple results / tensor results / noinline callees, and the
// post-conversion inlining that keeps a single kernel function for the
// downstream ttadapter.

// -----
// Case 1: a noinline callee returning (tensor, scalar) is inlined into the
// caller. The final module contains a single kernel function: no func.call,
// no private function, and the user is warned that noinline is not honored.

// WARN: noinline attribute is not honored
// CHECK-LABEL: func.func @kernel_block_sum
// CHECK-NOT: func.call
// CHECK-NOT: func.func private
// CHECK: linalg.reduce
// CHECK: bufferization.materialize_in_destination

tt.func public @kernel_block_sum(%x_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>, %sum_ptr: !tt.ptr<i32>) attributes {noinline = false} {
  %c0 = arith.constant 0 : i32
  %idx = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %blk:2 = tt.call @helper_block_sum(%x_ptr, %c0) : (!tt.ptr<i32>, i32) -> (tensor<8xi32>, i32)
  %splat = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %offs = tt.addptr %splat, %idx : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  tt.store %offs, %blk#0 : tensor<8x!tt.ptr<i32>>
  tt.store %sum_ptr, %blk#1 : !tt.ptr<i32>
  tt.return
}
tt.func private @helper_block_sum(%p: !tt.ptr<i32>, %offset: i32) -> (tensor<8xi32>, i32) attributes {noinline = true} {
  %c0 = arith.constant 0 : i32
  %idx = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %splat = tt.splat %p : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %offs = tt.addptr %splat, %idx : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  %v = tt.load %offs : tensor<8x!tt.ptr<i32>>
  %red = "tt.reduce"(%v) <{axis = 0 : i32}> ({
  ^bb0(%arg0: i32, %arg1: i32):
    %s = arith.addi %arg0, %arg1 : i32
    tt.reduce.return %s : i32
  }) : (tensor<8xi32>) -> i32
  tt.return %v, %red : tensor<8xi32>, i32
}

// -----
// Case 2: a noinline callee returning a scalar pair is inlined into the
// caller; both results feed scalar stores inside the single kernel function.

// CHECK-LABEL: func.func @kernel_pair
// CHECK-NOT: func.call
// CHECK-NOT: func.func private
// CHECK: arith.addi
// CHECK: arith.subi

tt.func public @kernel_pair(%x_ptr: !tt.ptr<i32>, %a_ptr: !tt.ptr<i32>, %b_ptr: !tt.ptr<i32>) attributes {noinline = false} {
  %c0 = arith.constant 0 : i32
  %ab:2 = tt.call @helper_pair(%x_ptr, %c0) : (!tt.ptr<i32>, i32) -> (i32, i32)
  tt.store %a_ptr, %ab#0 : !tt.ptr<i32>
  tt.store %b_ptr, %ab#1 : !tt.ptr<i32>
  tt.return
}
tt.func private @helper_pair(%p: !tt.ptr<i32>, %idx: i32) -> (i32, i32) attributes {noinline = true} {
  %v = tt.load %p : !tt.ptr<i32>
  %c1 = arith.constant 1 : i32
  %a = arith.addi %v, %c1 : i32
  %b = arith.subi %v, %c1 : i32
  tt.return %a, %b : i32, i32
}

// -----
// Case 3: a multi-block noinline callee is also folded into the kernel
// (its blocks are merged into arith.select); the final module remains a
// single kernel function with the tensor result materialized in place.

// CHECK-LABEL: func.func @kernel_mb
// CHECK-NOT: func.call
// CHECK-NOT: func.func private
// CHECK: arith.select

tt.func public @kernel_mb(%x_ptr: !tt.ptr<i32>, %cond: i32, %out_ptr: !tt.ptr<i32>) attributes {noinline = false} {
  %idx = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %blk = tt.call @helper_mb(%x_ptr, %cond) : (!tt.ptr<i32>, i32) -> tensor<8xi32>
  %splat = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %offs = tt.addptr %splat, %idx : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  tt.store %offs, %blk : tensor<8x!tt.ptr<i32>>
  tt.return
}
tt.func private @helper_mb(%p: !tt.ptr<i32>, %cond: i32) -> tensor<8xi32> attributes {noinline = true} {
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi ne, %cond, %c0 : i32
  cf.cond_br %cmp, ^bb1, ^bb2
^bb1:
  %c5 = arith.constant 5 : i32
  %t1 = tt.splat %c5 : i32 -> tensor<8xi32>
  tt.return %t1 : tensor<8xi32>
^bb2:
  %c6 = arith.constant 6 : i32
  %t2 = tt.splat %c6 : i32 -> tensor<8xi32>
  tt.return %t2 : tensor<8xi32>
}

// -----
// Case 4: an early-return callee (multiple return points flattened into
// scf.if by the frontend) without the noinline attribute: the call also
// survives the frontend inliner in a bare --triton-to-linalg run and is
// inlined here, with the tensor result flowing through the scf.if into the
// single kernel function.

// CHECK-LABEL: func.func @kernel_early_ret
// CHECK-NOT: func.call
// CHECK-NOT: func.func private
// CHECK: scf.if
// CHECK: bufferization.materialize_in_destination

tt.func public @kernel_early_ret(%x_ptr: !tt.ptr<i32>, %cond: i32, %out_ptr: !tt.ptr<i32>) attributes {noinline = false} {
  %idx = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %blk = tt.call @helper_early_ret(%x_ptr, %cond) : (!tt.ptr<i32>, i32) -> tensor<8xi32>
  %splat = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %offs = tt.addptr %splat, %idx : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  tt.store %offs, %blk : tensor<8x!tt.ptr<i32>>
  tt.return
}
tt.func private @helper_early_ret(%p: !tt.ptr<i32>, %cond: i32) -> tensor<8xi32> attributes {noinline = false} {
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi ne, %cond, %c0 : i32
  %idx = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %splat = tt.splat %p : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %offs = tt.addptr %splat, %idx : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  %v = tt.load %offs : tensor<8x!tt.ptr<i32>>
  %sel = scf.if %cmp -> (tensor<8xi32>) {
    %c1 = arith.constant 1 : i32
    %one = tt.splat %c1 : i32 -> tensor<8xi32>
    %r1 = arith.addi %v, %one : tensor<8xi32>
    scf.yield %r1 : tensor<8xi32>
  } else {
    %cz = arith.constant 0 : i32
    %z = tt.splat %cz : i32 -> tensor<8xi32>
    scf.yield %z : tensor<8xi32>
  }
  tt.return %sel : tensor<8xi32>
}
