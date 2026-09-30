// RUN: triton-opt %s --triton-to-linalg --split-input-file | FileCheck %s

// The stored value is computed after the indirect address, as in the community
// test_conditional_store_pipeline test.
// CHECK-LABEL: func.func @conditional_store_value_after_pointer
// CHECK: scf.for
// CHECK: scf.if
// CHECK: %[[VALUE:.*]] = arith.addi
// CHECK: %[[VALUES:.*]] = tensor.insert %[[VALUE]]
// CHECK: bufferization.materialize_in_destination %[[VALUES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @conditional_store_value_after_pointer(%indices: !tt.ptr<i32>, %output: !tt.ptr<i32>, %condition: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c17 = arith.constant 17 : i32
    scf.for %i = %c0 to %c17 step %c1 : i32 {
      %index_ptr = tt.addptr %indices, %i : !tt.ptr<i32>, i32
      %index_ptrs = tt.splat %index_ptr : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
      %index = tt.load %index_ptrs : tensor<1x!tt.ptr<i32>>
      scf.if %condition {
        %base = tt.splat %output : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
        %ptr = tt.addptr %base, %index : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
        %value = arith.addi %i, %c1 : i32
        %values = tt.splat %value : i32 -> tensor<1xi32>
        tt.store %ptr, %values : tensor<1x!tt.ptr<i32>>
      }
    }
    tt.return
  }
}

// -----

// Both the value and mask are defined after the address. The store must remain
// inside the conditional region where these operands are available.
// CHECK-LABEL: func.func @masked_store_value_after_pointer
// CHECK: scf.if
// CHECK: %[[VALUE:.*]] = arith.addi
// CHECK: %[[VALUES:.*]] = tensor.insert %[[VALUE]]
// CHECK: %[[MASK:.*]] = arith.cmpi
// CHECK: %[[SIZE:.*]] = arith.index_castui %[[MASK]]
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[VALUES]][0] [%[[SIZE]]]
// CHECK: bufferization.materialize_in_destination %[[SLICE]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @masked_store_value_after_pointer(%indices: !tt.ptr<i32>, %output: !tt.ptr<i32>, %value: i32, %limit: i32, %condition: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %index_ptrs = tt.splat %indices : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %index = tt.load %index_ptrs : tensor<1x!tt.ptr<i32>>
    %base = tt.splat %output : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %index : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    scf.if %condition {
      %next = arith.addi %value, %c1 : i32
      %values = tt.splat %next : i32 -> tensor<1xi32>
      %enabled = arith.cmpi slt, %c0, %limit : i32
      %mask = tt.splat %enabled : i1 -> tensor<1xi1>
      tt.store %ptr, %values, %mask : tensor<1x!tt.ptr<i32>>
    }
    tt.return
  }
}

// -----

// An indirect load must observe the store between the address calculation and
// the load. Moving the load next to its address would reverse these accesses.
// CHECK-LABEL: func.func @indirect_load_after_store
// CHECK: bufferization.materialize_in_destination
// CHECK: %[[LOADED:.*]] = memref.load
// CHECK: %[[VALUES:.*]] = tensor.insert %[[LOADED]]
// CHECK: bufferization.materialize_in_destination %[[VALUES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @indirect_load_after_store(%indices: !tt.ptr<i32>, %data: !tt.ptr<i32>, %output: !tt.ptr<i32>, %value: i32) {
    %index_ptrs = tt.splat %indices : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %index = tt.load %index_ptrs : tensor<1x!tt.ptr<i32>>
    %base = tt.splat %data : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %index : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    tt.store %data, %value : !tt.ptr<i32>
    %loaded = tt.load %ptr : tensor<1x!tt.ptr<i32>>
    %dest = tt.splat %output : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    tt.store %dest, %loaded : tensor<1x!tt.ptr<i32>>
    tt.return
  }
}
