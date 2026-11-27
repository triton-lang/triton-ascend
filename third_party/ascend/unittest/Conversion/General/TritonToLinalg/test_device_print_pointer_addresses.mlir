// RUN: triton-opt --triton-to-linalg --split-input-file %s -verify-each | FileCheck %s --implicit-check-not='!tt.ptr'

// A runtime element offset must contribute to the printed byte address.
// CHECK-LABEL: func.func @pointer_dynamic_offset(
// CHECK-SAME: %[[BASE:[^ ,]+]]: i64, %[[OFFSET:[^ ,)]+]]: i32
// CHECK-DAG: %[[WIDTH:.*]] = arith.constant 4 : i64
// CHECK: %[[WIDE:.*]] = arith.extsi %[[OFFSET]] : i32 to i64
// CHECK: %[[BYTES:.*]] = arith.muli %[[WIDE]], %[[WIDTH]] : i64
// CHECK: %[[ADDRESS:.*]] = arith.addi %[[BASE]], %[[BYTES]] : i64
// CHECK: call @triton_print{{[_0-9]*}}(%[[ADDRESS]]) : (i64) -> ()
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @pointer_dynamic_offset(%base: i64, %offset: i32) {
    %pointer = tt.int_to_ptr %base : i64 -> !tt.ptr<i32>
    %advanced = tt.addptr %pointer, %offset : !tt.ptr<i32>, i32
    tt.print "dynamic" {hex = false, isSigned = array<i32: 0>} : %advanced : !tt.ptr<i32>
    tt.return
  }
}

// -----

// The ordinary kernel-argument path extracts the allocation address and adds
// each runtime tensor offset in bytes before extracting the printed lane.
// CHECK-LABEL: func.func @pointer_argument_tensor_offset(
// CHECK-SAME: %[[BASE:[^ ,]+]]: memref<?xi32>, %[[OFFSET:[^ ,)]+]]: i32
// CHECK-DAG: %[[LANES:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK-DAG: %[[WIDTH:.*]] = arith.constant 4 : i64
// CHECK: %[[SCALE:.*]] = linalg.fill ins(%[[WIDTH]] : i64)
// CHECK: %[[RUNTIME:.*]] = linalg.fill ins(%[[OFFSET]] : i32)
// CHECK: %[[OFFSETS:.*]] = arith.addi %[[RUNTIME]], %[[LANES]] : tensor<4xi32>
// CHECK: %[[BASE_INDEX:.*]] = memref.extract_aligned_pointer_as_index %[[BASE]]
// CHECK: %[[BASE_ADDRESS:.*]] = arith.index_cast %[[BASE_INDEX]] : index to i64
// CHECK: %[[BASES:.*]] = linalg.fill ins(%[[BASE_ADDRESS]] : i64)
// CHECK: %[[WIDE:.*]] = arith.extsi %[[OFFSETS]] : tensor<4xi32> to tensor<4xi64>
// CHECK: %[[BYTES:.*]] = arith.muli %[[WIDE]], %[[SCALE]] : tensor<4xi64>
// CHECK: %[[ADDRESSES:.*]] = arith.addi %[[BASES]], %[[BYTES]] : tensor<4xi64>
// CHECK: scf.for %[[LANE:.*]] =
// CHECK:   %[[ADDRESS:.*]] = tensor.extract %[[ADDRESSES]][%[[LANE]]] : tensor<4xi64>
// CHECK:   call @triton_print{{[_0-9]*}}(%[[ADDRESS]]) : (i64) -> ()
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @pointer_argument_tensor_offset(%base: !tt.ptr<i32>, %offset: i32) {
    %lanes = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %runtime_offsets = tt.splat %offset : i32 -> tensor<4xi32>
    %offsets = arith.addi %runtime_offsets, %lanes : tensor<4xi32>
    %bases = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %pointers = tt.addptr %bases, %offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    tt.print "argument offsets" {hex = false, isSigned = array<i32: 0>} : %pointers : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// i8 increments are already byte offsets. Preserve all 64 bits of the offset.
// CHECK-LABEL: func.func @pointer_byte_offset(
// CHECK-SAME: %[[BASE:[^ ,]+]]: i64, %[[OFFSET:[^ ,)]+]]: i64
// CHECK-NOT: arith.trunci
// CHECK-NOT: arith.muli
// CHECK: %[[ADDRESS:.*]] = arith.addi %[[BASE]], %[[OFFSET]] : i64
// CHECK: call @triton_print{{[_0-9]*}}(%[[ADDRESS]]) : (i64) -> ()
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @pointer_byte_offset(%base: i64, %offset: i64) {
    %pointer = tt.int_to_ptr %base : i64 -> !tt.ptr<i8>
    %advanced = tt.addptr %pointer, %offset : !tt.ptr<i8>, i64
    tt.print "byte offset" {hex = false, isSigned = array<i32: 0>} : %advanced : !tt.ptr<i8>
    tt.return
  }
}

// -----

// Each addptr uses its own pointee width across a pointer bitcast.
// CHECK-LABEL: func.func @pointer_bitcast_offset(
// CHECK-SAME: %[[BASE:[^ ,]+]]: i64, %[[HALF_OFFSET:[^ ,]+]]: i32, %[[DOUBLE_OFFSET:[^ ,)]+]]: i64
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[EIGHT:.*]] = arith.constant 8 : i64
// CHECK: %[[WIDE:.*]] = arith.extsi %[[HALF_OFFSET]] : i32 to i64
// CHECK: %[[HALF_BYTES:.*]] = arith.muli %[[WIDE]], %[[TWO]] : i64
// CHECK: %[[INTERMEDIATE:.*]] = arith.addi %[[BASE]], %[[HALF_BYTES]] : i64
// CHECK: %[[DOUBLE_BYTES:.*]] = arith.muli %[[DOUBLE_OFFSET]], %[[EIGHT]] : i64
// CHECK: %[[ADDRESS:.*]] = arith.addi %[[INTERMEDIATE]], %[[DOUBLE_BYTES]] : i64
// CHECK: call @triton_print{{[_0-9]*}}(%[[ADDRESS]]) : (i64) -> ()
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @pointer_bitcast_offset(%base: i64, %half_offset: i32, %double_offset: i64) {
    %pointer = tt.int_to_ptr %base : i64 -> !tt.ptr<i16>
    %advanced = tt.addptr %pointer, %half_offset : !tt.ptr<i16>, i32
    %cast = tt.bitcast %advanced : !tt.ptr<i16> -> !tt.ptr<f64>
    %advanced_again = tt.addptr %cast, %double_offset : !tt.ptr<f64>, i64
    tt.print "bitcast offset" {hex = false, isSigned = array<i32: 0>} : %advanced_again : !tt.ptr<f64>
    tt.return
  }
}

// -----

// Visit the two tensor dimensions in logical row-major order, retaining i64
// values above UINT32_MAX. Integer-to-pointer input has a distinct base per lane.
// CHECK-LABEL: func.func @pointer_tensor_row_major(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[FOUR:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[ADDRESSES:.*]] = arith.constant dense<{{.*}}4294967296{{.*}}4294967300{{.*}}4294967304{{.*}}4294967308{{.*}}8589934592{{.*}}8589934596{{.*}}8589934600{{.*}}8589934604{{.*}}> : tensor<2x4xi64>
// CHECK: scf.for %[[ROW:.*]] = %[[ZERO]] to %[[TWO]] step %[[ONE]] {
// CHECK:   scf.for %[[COL:.*]] = %[[ZERO]] to %[[FOUR]] step %[[ONE]] {
// CHECK:     %[[ADDRESS:.*]] = tensor.extract %[[ADDRESSES]][%[[ROW]], %[[COL]]] : tensor<2x4xi64>
// CHECK:     call @triton_print{{[_0-9]*}}(%[[ADDRESS]]) : (i64) -> ()
// CHECK:   }
// CHECK: }
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @pointer_tensor_row_major() {
    %addresses = arith.constant dense<[[4294967296, 4294967300, 4294967304, 4294967308], [8589934592, 8589934596, 8589934600, 8589934604]]> : tensor<2x4xi64>
    %pointers = tt.int_to_ptr %addresses : tensor<2x4xi64> -> tensor<2x4x!tt.ptr<i32>>
    tt.print "row major" {hex = false, isSigned = array<i32: 0>} : %pointers : tensor<2x4x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// A tensor select may choose different allocations for different lanes. Retain
// the selected allocation and its lane's offset instead of printing one base.
// CHECK-LABEL: func.func @pointer_tensor_select_different_bases(
// CHECK: %[[ADDRESSES:.*]] = arith.constant dense<[4100, 8200, 4108, 8208]> : tensor<4xi64>
// CHECK: scf.for %[[LANE:.*]] =
// CHECK:   %[[ADDRESS:.*]] = tensor.extract %[[ADDRESSES]][%[[LANE]]] : tensor<4xi64>
// CHECK:   call @triton_print{{[_0-9]*}}(%[[ADDRESS]]) : (i64) -> ()
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @pointer_tensor_select_different_bases() {
    %lhs_address = arith.constant 4096 : i64
    %rhs_address = arith.constant 8192 : i64
    %offsets = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    %condition = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %lhs = tt.int_to_ptr %lhs_address : i64 -> !tt.ptr<f32>
    %rhs = tt.int_to_ptr %rhs_address : i64 -> !tt.ptr<f32>
    %lhs_tensor = tt.splat %lhs : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %rhs_tensor = tt.splat %rhs : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %lhs_ptrs = tt.addptr %lhs_tensor, %offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %rhs_ptrs = tt.addptr %rhs_tensor, %offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %selected = arith.select %condition, %lhs_ptrs, %rhs_ptrs : tensor<4xi1>, tensor<4x!tt.ptr<f32>>
    tt.print "different bases" {hex = false, isSigned = array<i32: 0>} : %selected : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Keep each consecutive group of numeric operands together, with signedness
// matching the original operand positions. Pointer output stays between groups.
// CHECK-DAG: func.func private @triton_print{{[_0-9]*}}(tensor<2xi32>, tensor<2xi8>, i64) attributes {hex = true, isSigned = array<i32: 1, 0, 0>, prefix = "mixed"}
// CHECK-DAG: func.func private @[[$PTR_LANE:triton_print[_0-9]*]](i64) attributes {hex = true, isSigned = array<i32: 0>, prefix = "mixed"}
// CHECK-DAG: func.func private @triton_print{{[_0-9]*}}(i32, tensor<2xi32>) attributes {hex = true, isSigned = array<i32: 0, 1>, prefix = "mixed"}
// CHECK-DAG: func.func private @[[$PTR_SCALAR:triton_print[_0-9]*]](i64) attributes {hex = true, isSigned = array<i32: 0>, prefix = "mixed"}
// CHECK-DAG: func.func private @triton_print{{[_0-9]*}}(i32, i32) attributes {hex = false, isSigned = array<i32: 1, 0>, prefix = "scalar pair"}
// CHECK-LABEL: func.func @mixed_pointer_and_numeric_operands(
// CHECK: call @triton_print{{[_0-9]*}}({{.*}}) : (tensor<2xi32>, tensor<2xi8>, i64) -> ()
// CHECK: scf.for %{{.*}} =
// CHECK:   call @[[$PTR_LANE]]({{.*}}) : (i64) -> ()
// CHECK: }
// CHECK: call @triton_print{{[_0-9]*}}({{.*}}) : (i32, tensor<2xi32>) -> ()
// CHECK: call @[[$PTR_SCALAR]]({{.*}}) : (i64) -> ()
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @mixed_pointer_and_numeric_operands(%address: i64, %unsigned64: i64, %unsigned32: i32) {
    %signed_values = arith.constant dense<[-1, 7]> : tensor<2xi32>
    %unsigned_values = arith.constant dense<[128, 255]> : tensor<2xi8>
    %addresses = arith.constant dense<[4096, 8192]> : tensor<2xi64>
    %pointers = tt.int_to_ptr %addresses : tensor<2xi64> -> tensor<2x!tt.ptr<i32>>
    %scalar_pointer = tt.int_to_ptr %address : i64 -> !tt.ptr<i32>
    tt.print "mixed" {hex = true, isSigned = array<i32: 1, 0, 0, 0, 0, 1, 0>} : %signed_values, %unsigned_values, %unsigned64, %pointers, %unsigned32, %signed_values, %scalar_pointer : tensor<2xi32>, tensor<2xi8>, i64, tensor<2x!tt.ptr<i32>>, i32, tensor<2xi32>, !tt.ptr<i32>
    tt.print "scalar pair" {hex = false, isSigned = array<i32: 1, 0>} : %unsigned32, %unsigned32 : i32, i32
    tt.return
  }
}
