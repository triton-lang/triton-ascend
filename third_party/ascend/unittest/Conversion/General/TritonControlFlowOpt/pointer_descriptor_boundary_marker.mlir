// RUN: triton-opt --triton-control-flow-opt --split-input-file %s | FileCheck %s

module {
  tt.func public @block_ptr_loop_marker(%base: !tt.ptr<f32>, %upper: index) -> !tt.ptr<tensor<16xf32>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %initial = tt.make_tensor_ptr %base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    %final = scf.for %iv = %c0 to %upper step %c1 iter_args(%ptr = %initial) -> (!tt.ptr<tensor<16xf32>>) {
      %next = tt.advance %ptr, [%c1_i32] : !tt.ptr<tensor<16xf32>>
      scf.yield %next : !tt.ptr<tensor<16xf32>>
    }
    tt.return %final : !tt.ptr<tensor<16xf32>>
  }
}

// CHECK-LABEL: tt.func public @block_ptr_loop_marker
// CHECK:       scf.for
// CHECK:       PointerDescriptorBoundary = array<i32: 0, 1, 2, 3>

// -----

module {
  tt.func public @tensor_ptr_loop_marker(%base: !tt.ptr<f32>, %upper: index) -> tensor<4x!tt.ptr<f32>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = tt.splat %c0_i32 : i32 -> tensor<4xi32>
    %delta = tt.splat %c1_i32 : i32 -> tensor<4xi32>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %initial = tt.addptr %base_tensor, %zero : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %final = scf.for %iv = %c0 to %upper step %c1 iter_args(%ptr = %initial) -> (tensor<4x!tt.ptr<f32>>) {
      %next = tt.addptr %ptr, %delta : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %next : tensor<4x!tt.ptr<f32>>
    }
    tt.return %final : tensor<4x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: tt.func public @tensor_ptr_loop_marker
// CHECK:       scf.for
// CHECK:       PointerDescriptorBoundary = array<i32: 0>

// -----

module {
  tt.func public @mixed_pointer_loop_marker(%block_base: !tt.ptr<f16>, %tensor_base: !tt.ptr<f32>, %upper: index) -> tensor<4x!tt.ptr<f32>> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = tt.splat %c0_i32 : i32 -> tensor<4xi32>
    %delta = tt.splat %c1_i32 : i32 -> tensor<4xi32>
    %block = tt.make_tensor_ptr %block_base, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf16>>
    %tensor_base_splat = tt.splat %tensor_base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %tensor = tt.addptr %tensor_base_splat, %zero : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %results:4 = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc = %c0_i32, %block_arg = %block, %ordinary = %c1_i32, %tensor_arg = %tensor) -> (i32, !tt.ptr<tensor<16xf16>>, i32, tensor<4x!tt.ptr<f32>>) {
      %next_block = tt.advance %block_arg, [%c1_i32] : !tt.ptr<tensor<16xf16>>
      %next_tensor = tt.addptr %tensor_arg, %delta : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %acc, %next_block, %ordinary, %next_tensor : i32, !tt.ptr<tensor<16xf16>>, i32, tensor<4x!tt.ptr<f32>>
    }
    tt.return %results#3 : tensor<4x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: tt.func public @mixed_pointer_loop_marker
// CHECK:       scf.for
// CHECK:       PointerDescriptorBoundary = array<i32: 1, 2, 3, 4, 6>
