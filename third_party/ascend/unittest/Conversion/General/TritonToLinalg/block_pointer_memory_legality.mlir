// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s \
// RUN:   --implicit-check-not=tt.advance --implicit-check-not=tt.make_tensor_ptr \
// RUN:   --implicit-check-not='!tt.ptr'

// Memory-stage legality must select block pointers for rewriting without
// attempting to convert their tensor pointee into a memref element type.
// The advance chain must preserve the accumulated offset of the loaded tile.
// CHECK-LABEL: func.func @block_pointer_advance_chain
// CHECK: %[[OFFSET:.*]] = arith.constant 128 : index
// CHECK: %[[TILE:.*]] = memref.reinterpret_cast {{.*}} to offset: [%[[OFFSET]]], sizes: [64], strides: [1]
// CHECK: memref.copy %[[TILE]],
// CHECK: linalg.reduce
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @block_pointer_advance_chain(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %base = tt.make_tensor_ptr %src, [%c256_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<64xf32>>
    %first = tt.advance %base, [%c64_i32] : <tensor<64xf32>>
    %second = tt.advance %first, [%c64_i32] : <tensor<64xf32>>
    %data = tt.load %second {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xf32>>
    %sum = "tt.reduce"(%data) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) : (tensor<64xf32>) -> f32
    tt.store %dst, %sum : !tt.ptr<f32>
    tt.return
  }
}

// -----

// An advance inside a nested loop keeps the block pointer live across the
// loop boundary. Its dynamic offset and the loaded data must survive memory
// conversion and the subsequent cleanup.
// CHECK-LABEL: func.func @nested_dynamic_block_pointer
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.reinterpret_cast
// CHECK: memref.copy
// CHECK: linalg.reduce
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @nested_dynamic_block_pointer(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %stride: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    scf.for %i = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
      %outer_offset = arith.muli %i, %stride : i32
      %outer_base = tt.addptr %src, %outer_offset : !tt.ptr<f32>, i32
      %block = tt.make_tensor_ptr %outer_base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<8xf32>>
      %out_offset = arith.muli %i, %c2_i32 : i32
      %out_base = tt.addptr %dst, %out_offset : !tt.ptr<f32>, i32
      scf.for %j = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
        %inner_offset = arith.muli %j, %c8_i32 : i32
        %advanced = tt.advance %block, [%inner_offset] : <tensor<8xf32>>
        %data = tt.load %advanced {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
        %sum = "tt.reduce"(%data) <{axis = 0 : i32}> ({
        ^bb0(%lhs: f32, %rhs: f32):
          %add = arith.addf %lhs, %rhs : f32
          tt.reduce.return %add : f32
        }) : (tensor<8xf32>) -> f32
        %out = tt.addptr %out_base, %j : !tt.ptr<f32>, i32
        tt.store %out, %sum : !tt.ptr<f32>
      }
    }
    tt.return
  }
}
