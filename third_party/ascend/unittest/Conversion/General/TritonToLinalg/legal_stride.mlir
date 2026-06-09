// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s
// CHECK-LABEL: func.func @triton_fn_broadcast_nested
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[CAST1:.*]] = memref.reinterpret_cast %[[ARG2:.*]] to offset: [%[[ARG13:.*]]], sizes: [4, 1], strides: [%c4, %[[C1]]] : memref<?xf32> to memref<4x1xf32, strided<[?, ?], offset: ?>>
// CHECK:       %[[CAST2:.*]] = memref.reinterpret_cast %[[ARG3:.*]] to offset: [%[[ARG13]]], sizes: [4, 1], strides: [%c4, %[[C1]]] : memref<?xf32> to memref<4x1xf32, strided<[?, ?], offset: ?>>

module {
  tt.func @triton_fn_broadcast_nested(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32){
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %c0) -> (index)  : i32 {
      %1 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %arg11) -> (index)  : i32 {
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%arg13], sizes: [4, 1], strides: [%c4, %c0] : memref<?xf32> to memref<4x1xf32, strided<[?, ?], offset: ?>>
        %alloc = memref.alloc() : memref<4x1xf32>
        memref.copy %reinterpret_cast, %alloc : memref<4x1xf32, strided<[?, ?], offset: ?>> to memref<4x1xf32>
        %2 = bufferization.to_tensor %alloc restrict writable : memref<4x1xf32>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%arg13], sizes: [4, 1], strides: [%c4, %c0] : memref<?xf32> to memref<4x1xf32, strided<[?, ?], offset: ?>>
        bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<4x1xf32>, memref<4x1xf32, strided<[?, ?], offset: ?>>) -> ()
        %3 = arith.addi %arg13, %c1 : index
        scf.yield %3 : index
      }
      scf.yield %1 : index
    }
    tt.return
  }
}
