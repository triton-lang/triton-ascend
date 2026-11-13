// RUN: triton-opt %s --triton-to-linalg='named-ops=true' -verify-each | FileCheck %s
// RUN: triton-opt %s --triton-to-linalg -verify-each | FileCheck %s
// RUN: sed 's/Ascend910B2/Ascend950PR_9579/' %s | triton-opt --triton-to-linalg='named-ops=true compile-on-910-95=true' -verify-each | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  // Inductor embedding pattern: a boolean is numerical address data, not a
  // memory mask. Preserve the zero extension and allow multiplication by a
  // second loaded singleton without treating both operands as gather sources.
  // CHECK-LABEL: func.func @cmp_extui_row_offset(
  // CHECK: %[[WIDE:.*]] = arith.extui %{{.*}} : tensor<1xi1> to tensor<1xi32>
  // CHECK: %[[VALUE:.*]] = tensor.extract %[[WIDE]][%{{.*}}] : tensor<1xi32>
  // CHECK: %[[INDEX:.*]] = arith.index_cast %[[VALUE]] : i32 to index
  // CHECK: arith.muli %{{.*}}, %[[INDEX]] : index
  // CHECK: memref.reinterpret_cast
  // CHECK: return
  tt.func public @cmp_extui_row_offset(%tokens: !tt.ptr<i64>, %counts: !tt.ptr<i64>, %data: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %pad = arith.constant dense<2047> : tensor<1xi64>
    %width = arith.constant dense<8> : tensor<1xi64>
    %token_ptr = tt.splat %tokens : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>>
    %count_ptr = tt.splat %counts : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>>
    %token = tt.load %token_ptr : tensor<1x!tt.ptr<i64>>
    %count64 = tt.load %count_ptr : tensor<1x!tt.ptr<i64>>
    %count = arith.trunci %count64 : tensor<1xi64> to tensor<1xi32>
    %not_pad = arith.cmpi ne, %token, %pad : tensor<1xi64>
    %flag = arith.extui %not_pad : tensor<1xi1> to tensor<1xi32>
    %selected_count = arith.muli %count, %flag : tensor<1xi32>
    %selected_count64 = arith.extsi %selected_count : tensor<1xi32> to tensor<1xi64>
    %row = arith.addi %selected_count64, %pad : tensor<1xi64>
    %row_offset = arith.muli %row, %width : tensor<1xi64>
    %columns = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %columns64 = arith.extsi %columns : tensor<8xi32> to tensor<8xi64>
    %rows = tt.broadcast %row_offset : tensor<1xi64> -> tensor<8xi64>
    %offsets = arith.addi %rows, %columns64 : tensor<8xi64>
    %base = tt.splat %data : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
    %value = tt.load %ptr : tensor<8x!tt.ptr<f32>>
    %out_base = tt.splat %output : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %out = tt.addptr %out_base, %columns : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %out, %value : tensor<8x!tt.ptr<f32>>
    tt.return
  }

  // Values with the high bit set must stay unsigned: i8 255 indexes row 255,
  // rather than row -1. Extract the cast result, not its narrow input.
  // CHECK-LABEL: func.func @unsigned_singleton_offset(
  // CHECK: %[[WIDE:.*]] = arith.extui %{{.*}} : tensor<1xi8> to tensor<1xi32>
  // CHECK: %[[VALUE:.*]] = tensor.extract %[[WIDE]][%{{.*}}] : tensor<1xi32>
  // CHECK: %[[INDEX:.*]] = arith.index_cast %[[VALUE]] : i32 to index
  // CHECK: memref.reinterpret_cast %{{.*}} to offset: [%[[INDEX]]]
  // CHECK: return
  tt.func public @unsigned_singleton_offset(%indices: !tt.ptr<i8>, %data: !tt.ptr<i32>, %output: !tt.ptr<i32>) {
    %index_ptr = tt.splat %indices : !tt.ptr<i8> -> tensor<1x!tt.ptr<i8>>
    %index8 = tt.load %index_ptr : tensor<1x!tt.ptr<i8>>
    %index = arith.extui %index8 : tensor<1xi8> to tensor<1xi32>
    %base = tt.splat %data : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %index : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %value = tt.load %ptr : tensor<1x!tt.ptr<i32>>
    %out = tt.splat %output : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    tt.store %out, %value : tensor<1x!tt.ptr<i32>>
    tt.return
  }

  // Multiple boolean results must remain per-element offsets, not a shared
  // scalar or a mask controlling the gather.
  // CHECK-LABEL: func.func @cmp_extui_vector_offset(
  // CHECK: %[[WIDE:.*]] = arith.extui %{{.*}} : tensor<8xi1> to tensor<8xi32>
  // CHECK: scf.for %[[IV:[^ ]+]] =
  // CHECK: %[[VALUE:.*]] = tensor.extract %[[WIDE]][%[[IV]]] : tensor<8xi32>
  // CHECK: %[[INDEX:.*]] = arith.index_cast %[[VALUE]] : i32 to index
  // CHECK: memref.reinterpret_cast %{{.*}} to offset: [%[[INDEX]]]
  // CHECK: memref.load
  // CHECK: return
  tt.func public @cmp_extui_vector_offset(%indices: !tt.ptr<i32>, %data: !tt.ptr<i32>, %output: !tt.ptr<i32>) {
    %zero = arith.constant dense<0> : tensor<8xi32>
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %index_base = tt.splat %indices : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %index_ptrs = tt.addptr %index_base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %indices32 = tt.load %index_ptrs : tensor<8x!tt.ptr<i32>>
    %positive = arith.cmpi sgt, %indices32, %zero : tensor<8xi32>
    %offsets = arith.extui %positive : tensor<8xi1> to tensor<8xi32>
    %base = tt.splat %data : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %value = tt.load %ptr : tensor<8x!tt.ptr<i32>>
    %out_base = tt.splat %output : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %out = tt.addptr %out_base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %out, %value : tensor<8x!tt.ptr<i32>>
    tt.return
  }

  // CHECK-LABEL: func.func @unsigned_vector_offset(
  // CHECK: %[[WIDE:.*]] = arith.extui %{{.*}} : tensor<8xi8> to tensor<8xi32>
  // CHECK: scf.for %[[IV:[^ ]+]] =
  // CHECK: %[[VALUE:.*]] = tensor.extract %[[WIDE]][%[[IV]]] : tensor<8xi32>
  // CHECK: %[[INDEX:.*]] = arith.index_cast %[[VALUE]] : i32 to index
  // CHECK: memref.reinterpret_cast %{{.*}} to offset: [%[[INDEX]]]
  // CHECK: memref.load
  // CHECK: return
  tt.func public @unsigned_vector_offset(%indices: !tt.ptr<i8>, %data: !tt.ptr<i32>, %output: !tt.ptr<i32>) {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %index_base = tt.splat %indices : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %index_ptrs = tt.addptr %index_base, %range : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %indices8 = tt.load %index_ptrs : tensor<8x!tt.ptr<i8>>
    %offsets = arith.extui %indices8 : tensor<8xi8> to tensor<8xi32>
    %base = tt.splat %data : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %value = tt.load %ptr : tensor<8x!tt.ptr<i32>>
    %out_base = tt.splat %output : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %out = tt.addptr %out_base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %out, %value : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
