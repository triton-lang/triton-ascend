// RUN: triton-opt --triton-to-linalg="named-ops=true" --split-input-file %s | FileCheck %s

// Address-only tensor computations lose their last uses when memory
// operations are lowered. The intermediate cleanup must remove them before
// the numerical TTIR conversion can turn them into linalg operations.
// CHECK-LABEL: func.func @address_only_chain_is_removed
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.generic
// CHECK: memref.copy
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.generic
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.generic
// CHECK-NOT: tt.make_range
// CHECK-NOT: tt.addptr

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @address_only_chain_is_removed(
      %src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
      attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %two = arith.constant dense<2> : tensor<8xi32>
    %offsets = arith.muli %range, %two : tensor<8xi32>
    %srcs = tt.splat %src : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %src_ptrs = tt.addptr %srcs, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %value = tt.load %src_ptrs : tensor<8x!tt.ptr<i32>>
    %dsts = tt.splat %dst : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %dst_ptrs = tt.addptr %dsts, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %dst_ptrs, %value : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// A producer shared by the address and data paths must remain live after the
// address use disappears. In named-ops mode the range becomes a generic while
// tensor additions stay as arith.addi. Check that the shared offsets still
// contribute to the value written to the destination.
// CHECK-LABEL: func.func @shared_address_and_data
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[ONES:.*]] = linalg.fill ins(%[[ONE]] : i32)
// CHECK: %[[RANGE:.*]] = linalg.generic
// CHECK-SAME: tt.from_make_range
// CHECK: %[[OFFSETS:.*]] = arith.addi %[[RANGE]], %[[ONES]] : tensor<8xi32>
// CHECK: memref.copy %{{.*}}, %[[BUFFER:.*]] :
// CHECK: %[[LOADED:.*]] = bufferization.to_tensor %[[BUFFER]]
// CHECK: %[[VALUE:.*]] = arith.addi %[[LOADED]], %[[OFFSETS]] : tensor<8xi32>
// CHECK: bufferization.materialize_in_destination %[[VALUE]] in writable
// CHECK-NOT: tt.make_range
// CHECK-NOT: tt.addptr

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @shared_address_and_data(
      %src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
      attributes {noinline = false} {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %one = arith.constant dense<1> : tensor<8xi32>
    %offsets = arith.addi %range, %one : tensor<8xi32>
    %srcs = tt.splat %src : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %src_ptrs = tt.addptr %srcs, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %loaded = tt.load %src_ptrs : tensor<8x!tt.ptr<i32>>
    %value = arith.addi %loaded, %offsets : tensor<8xi32>
    %dsts = tt.splat %dst : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %dst_ptrs = tt.addptr %dsts, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %dst_ptrs, %value : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
