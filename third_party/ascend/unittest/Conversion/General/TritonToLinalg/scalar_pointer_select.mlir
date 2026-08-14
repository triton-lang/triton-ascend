// RUN: triton-opt --triton-to-linalg %s -verify-each | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  func.func private @consume_offset_view(memref<1xf32, strided<[1], offset: 1>>)
  func.func private @next_offset_view(memref<1xf32, strided<[1], offset: 1>>) -> memref<1xf32, strided<[1], offset: 1>>

  tt.func public @scalar_pointer_select(%lhs: !tt.ptr<f32>, %rhs: !tt.ptr<f32>, %condition: i1) -> i64 {
    %selected = arith.select %condition, %lhs, %rhs : !tt.ptr<f32>
    %address = tt.ptr_to_int %selected : !tt.ptr<f32> -> i64
    tt.return %address : i64
  }

  tt.func public @scalar_pointer_roundtrip(%address: i64) -> i64 {
    %pointer = tt.int_to_ptr %address : i64 -> !tt.ptr<f32>
    %roundtrip = tt.ptr_to_int %pointer : !tt.ptr<f32> -> i64
    tt.return %roundtrip : i64
  }

  func.func @pointer_cast_static_offset(%address: i64) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %pointer = hivm.hir.pointer_cast(%address) [%c1] : memref<?xf32>
    %view = memref.reinterpret_cast %pointer to offset: [1], sizes: [1], strides: [1]
        : memref<?xf32> to memref<1xf32, strided<[1], offset: 1>>
    %value = memref.load %view[%c0] : memref<1xf32, strided<[1], offset: 1>>
    return %value : f32
  }

  func.func @pointer_cast_dynamic_offset_subviews(%address: i64, %offset: index, %size: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %pointer = hivm.hir.pointer_cast(%address) [%c8] : memref<?xf32>
    %view = memref.reinterpret_cast %pointer to offset: [%offset], sizes: [1, 8], strides: [8, 1]
        : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
    %rank_reduced = memref.subview %view[0, 0] [1, %size] [1, 1]
        : memref<1x8xf32, strided<[8, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview = memref.subview %rank_reduced[0] [%size] [1]
        : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %value = memref.load %subview[%c0] : memref<?xf32, strided<[1], offset: ?>>
    return %value : f32
  }

  func.func @pointer_cast_offset_return(%address: i64) -> memref<1xf32, strided<[1], offset: 1>> {
    %c1 = arith.constant 1 : index
    %pointer = hivm.hir.pointer_cast(%address) [%c1] : memref<?xf32>
    %view = memref.reinterpret_cast %pointer to offset: [1], sizes: [1], strides: [1]
        : memref<?xf32> to memref<1xf32, strided<[1], offset: 1>>
    return %view : memref<1xf32, strided<[1], offset: 1>>
  }

  func.func @pointer_cast_dynamic_offset_return(%address: i64, %offset: index) -> memref<1xf32, strided<[1], offset: ?>> {
    %c1 = arith.constant 1 : index
    %pointer = hivm.hir.pointer_cast(%address) [%c1] : memref<?xf32>
    %view = memref.reinterpret_cast %pointer to offset: [%offset], sizes: [1], strides: [1]
        : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
    return %view : memref<1xf32, strided<[1], offset: ?>>
  }

  func.func @pointer_cast_offset_call(%address: i64) {
    %c1 = arith.constant 1 : index
    %pointer = hivm.hir.pointer_cast(%address) [%c1] : memref<?xf32>
    %view = memref.reinterpret_cast %pointer to offset: [1], sizes: [1], strides: [1]
        : memref<?xf32> to memref<1xf32, strided<[1], offset: 1>>
    call @consume_offset_view(%view) : (memref<1xf32, strided<[1], offset: 1>>) -> ()
    return
  }

  func.func @pointer_cast_offset_if(%address: i64, %condition: i1, %alternate: memref<1xf32, strided<[1], offset: 1>>) -> memref<1xf32, strided<[1], offset: 1>> {
    %c1 = arith.constant 1 : index
    %pointer = hivm.hir.pointer_cast(%address) [%c1] : memref<?xf32>
    %view = memref.reinterpret_cast %pointer to offset: [1], sizes: [1], strides: [1]
        : memref<?xf32> to memref<1xf32, strided<[1], offset: 1>>
    %result = scf.if %condition -> (memref<1xf32, strided<[1], offset: 1>>) {
      %then = call @next_offset_view(%view) : (memref<1xf32, strided<[1], offset: 1>>) -> memref<1xf32, strided<[1], offset: 1>>
      scf.yield %then : memref<1xf32, strided<[1], offset: 1>>
    } else {
      %else = call @next_offset_view(%alternate) : (memref<1xf32, strided<[1], offset: 1>>) -> memref<1xf32, strided<[1], offset: 1>>
      scf.yield %else : memref<1xf32, strided<[1], offset: 1>>
    }
    return %result : memref<1xf32, strided<[1], offset: 1>>
  }

  func.func @pointer_cast_offset_loop(%address: i64, %upper: index) -> memref<1xf32, strided<[1], offset: 1>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %pointer = hivm.hir.pointer_cast(%address) [%c1] : memref<?xf32>
    %view = memref.reinterpret_cast %pointer to offset: [1], sizes: [1], strides: [1]
        : memref<?xf32> to memref<1xf32, strided<[1], offset: 1>>
    %result = scf.for %iv = %c0 to %upper step %c1 iter_args(%current = %view) -> (memref<1xf32, strided<[1], offset: 1>>) {
      %next = call @next_offset_view(%current) : (memref<1xf32, strided<[1], offset: 1>>) -> memref<1xf32, strided<[1], offset: 1>>
      scf.yield %next : memref<1xf32, strided<[1], offset: 1>>
    }
    return %result : memref<1xf32, strided<[1], offset: 1>>
  }

  func.func @pointer_cast_offset_while(%address: i64, %condition: i1) -> memref<1xf32, strided<[1], offset: 1>> {
    %c1 = arith.constant 1 : index
    %pointer = hivm.hir.pointer_cast(%address) [%c1] : memref<?xf32>
    %view = memref.reinterpret_cast %pointer to offset: [1], sizes: [1], strides: [1]
        : memref<?xf32> to memref<1xf32, strided<[1], offset: 1>>
    %result = scf.while (%current = %view) : (memref<1xf32, strided<[1], offset: 1>>) -> (memref<1xf32, strided<[1], offset: 1>>) {
      scf.condition(%condition) %current : memref<1xf32, strided<[1], offset: 1>>
    } do {
    ^bb0(%current: memref<1xf32, strided<[1], offset: 1>>):
      %next = call @next_offset_view(%current) : (memref<1xf32, strided<[1], offset: 1>>) -> memref<1xf32, strided<[1], offset: 1>>
      scf.yield %next : memref<1xf32, strided<[1], offset: 1>>
    }
    return %result : memref<1xf32, strided<[1], offset: 1>>
  }
}

// CHECK-LABEL: func.func @scalar_pointer_select(
// CHECK-SAME:  %[[LHS:[^ ,]+]]: memref<?xf32>, %[[RHS:[^ ,]+]]: memref<?xf32>
// CHECK:       %[[LHS_INDEX:.*]] = memref.extract_aligned_pointer_as_index %[[LHS]]
// CHECK:       %[[LHS_ADDRESS:.*]] = arith.index_cast %[[LHS_INDEX]] : index to i64
// CHECK:       %[[RHS_INDEX:.*]] = memref.extract_aligned_pointer_as_index %[[RHS]]
// CHECK:       %[[RHS_ADDRESS:.*]] = arith.index_cast %[[RHS_INDEX]] : index to i64
// CHECK:       %[[SELECTED:.*]] = arith.select %{{.*}}, %[[LHS_ADDRESS]], %[[RHS_ADDRESS]] : i64
// CHECK-NOT:   arith.select {{.*}} : memref
// CHECK-NOT:   hivm.hir.pointer_cast
// CHECK-NOT:   memref.extract_aligned_pointer_as_index
// CHECK:       return %[[SELECTED]] : i64

// CHECK-LABEL: func.func @scalar_pointer_roundtrip(
// CHECK-SAME:  %[[ADDRESS:[^ ,]+]]: i64
// CHECK-NOT:   hivm.hir.pointer_cast
// CHECK-NOT:   memref.extract_aligned_pointer_as_index
// CHECK:       return %[[ADDRESS]] : i64

// CHECK-LABEL: func.func @pointer_cast_static_offset(
// CHECK-SAME:  %[[ADDRESS:[^ ,]+]]: i64
// CHECK:       %[[OFFSET_I64:.*]] = arith.index_cast %[[OFFSET:.*]] : index to i64
// CHECK:       %[[BYTE_WIDTH:.*]] = arith.constant 4 : i64
// CHECK:       %[[BYTE_OFFSET:.*]] = arith.muli %[[OFFSET_I64]], %[[BYTE_WIDTH]] : i64
// CHECK:       %[[REAL_ADDRESS:.*]] = arith.addi %[[ADDRESS]], %[[BYTE_OFFSET]] : i64
// CHECK:       %[[REBASED_POINTER:.*]] = hivm.hir.pointer_cast(%[[REAL_ADDRESS]]) [%[[SIZE:.*]]] : memref<?xf32>
// CHECK:       %[[VIEW:.*]] = memref.reinterpret_cast %[[REBASED_POINTER]] to offset: [0], sizes: [1], strides: [1]
// CHECK-SAME:  to memref<1xf32, strided<[1]>>
// CHECK:       memref.load %[[VIEW]][%{{.*}}] : memref<1xf32, strided<[1]>>

// CHECK-LABEL: func.func @pointer_cast_dynamic_offset_subviews(
// CHECK-SAME:  %[[ADDRESS:[^ ,]+]]: i64, %[[OFFSET:[^ ,]+]]: index, %[[DYNAMIC_SIZE:[^ ,]+]]: index
// CHECK:       %[[OFFSET_I64:.*]] = arith.index_cast %[[OFFSET]] : index to i64
// CHECK:       %[[BYTE_WIDTH:.*]] = arith.constant 4 : i64
// CHECK:       %[[BYTE_OFFSET:.*]] = arith.muli %[[OFFSET_I64]], %[[BYTE_WIDTH]] : i64
// CHECK:       %[[REAL_ADDRESS:.*]] = arith.addi %[[ADDRESS]], %[[BYTE_OFFSET]] : i64
// CHECK:       %[[REBASED_POINTER:.*]] = hivm.hir.pointer_cast(%[[REAL_ADDRESS]]) [%[[SIZE:.*]]] : memref<?xf32>
// CHECK:       %[[VIEW:.*]] = memref.reinterpret_cast %[[REBASED_POINTER]] to offset: [0], sizes: [1, 8], strides: [8, 1]
// CHECK-SAME:  to memref<1x8xf32, strided<[8, 1]>>
// CHECK:       %[[RANK_REDUCED:.*]] = memref.subview %[[VIEW]][0, 0] [1, %[[DYNAMIC_SIZE]]] [1, 1]
// CHECK-SAME:  to memref<?xf32, strided<[1]>>
// CHECK:       %[[SUBVIEW:.*]] = memref.subview %[[RANK_REDUCED]][0] [%[[DYNAMIC_SIZE]]] [1]
// CHECK-SAME:  to memref<?xf32, strided<[1]>>
// CHECK:       memref.load %[[SUBVIEW]][%{{.*}}] : memref<?xf32, strided<[1]>>

// CHECK-LABEL: func.func @pointer_cast_offset_return(
// CHECK-SAME:  %[[ADDRESS:[^ ,]+]]: i64
// CHECK:       %[[CAPACITY:.*]] = arith.addi %[[VIEW_SIZE:.*]], %[[VIEW_OFFSET:.*]] : index
// CHECK:       %[[POINTER:.*]] = hivm.hir.pointer_cast(%[[ADDRESS]]) [%[[CAPACITY]]] : memref<?xf32>
// CHECK:       %[[VIEW:.*]] = memref.reinterpret_cast %[[POINTER]] to offset: [1], sizes: [1], strides: [1]
// CHECK-SAME:  to memref<1xf32, strided<[1], offset: 1>>
// CHECK:       return %[[VIEW]] : memref<1xf32, strided<[1], offset: 1>>

// CHECK-LABEL: func.func @pointer_cast_dynamic_offset_return(
// CHECK-SAME:  %[[ADDRESS:[^ ,]+]]: i64, %[[OFFSET:[^ ,]+]]: index
// CHECK:       %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:       %[[LEADING_EXTENT:.*]] = arith.maxsi %[[OFFSET]], %[[ZERO]] : index
// CHECK:       %[[CAPACITY:.*]] = arith.addi %[[VIEW_SIZE:.*]], %[[LEADING_EXTENT]] : index
// CHECK:       %[[POINTER:.*]] = hivm.hir.pointer_cast(%[[ADDRESS]]) [%[[CAPACITY]]] : memref<?xf32>
// CHECK:       %[[VIEW:.*]] = memref.reinterpret_cast %[[POINTER]] to offset: [%[[OFFSET]]], sizes: [1], strides: [1]
// CHECK-SAME:  to memref<1xf32, strided<[1], offset: ?>>
// CHECK:       return %[[VIEW]] : memref<1xf32, strided<[1], offset: ?>>

// CHECK-LABEL: func.func @pointer_cast_offset_call(
// CHECK-SAME:  %[[ADDRESS:[^ ,]+]]: i64
// CHECK:       %[[POINTER:.*]] = hivm.hir.pointer_cast(%[[ADDRESS]])
// CHECK:       %[[VIEW:.*]] = memref.reinterpret_cast %[[POINTER]] to offset: [1], sizes: [1], strides: [1]
// CHECK-SAME:  to memref<1xf32, strided<[1], offset: 1>>
// CHECK:       call @consume_offset_view(%[[VIEW]])

// CHECK-LABEL: func.func @pointer_cast_offset_if(
// CHECK-SAME:  %[[ADDRESS:[^ ,]+]]: i64
// CHECK:       %[[POINTER:.*]] = hivm.hir.pointer_cast(%[[ADDRESS]])
// CHECK:       %[[VIEW:.*]] = memref.reinterpret_cast %[[POINTER]] to offset: [1], sizes: [1], strides: [1]
// CHECK-SAME:  to memref<1xf32, strided<[1], offset: 1>>
// CHECK:       scf.if
// CHECK-SAME:  -> (memref<1xf32, strided<[1], offset: 1>>)
// CHECK:       call @next_offset_view(%[[VIEW]])
// CHECK:       scf.yield {{.*}} : memref<1xf32, strided<[1], offset: 1>>

// CHECK-LABEL: func.func @pointer_cast_offset_loop(
// CHECK-SAME:  %[[ADDRESS:[^ ,]+]]: i64
// CHECK:       %[[POINTER:.*]] = hivm.hir.pointer_cast(%[[ADDRESS]])
// CHECK:       %[[VIEW:.*]] = memref.reinterpret_cast %[[POINTER]] to offset: [1], sizes: [1], strides: [1]
// CHECK-SAME:  to memref<1xf32, strided<[1], offset: 1>>
// CHECK:       scf.for
// CHECK-SAME:  memref<1xf32, strided<[1], offset: 1>>

// CHECK-LABEL: func.func @pointer_cast_offset_while(
// CHECK-SAME:  %[[ADDRESS:[^ ,]+]]: i64
// CHECK:       %[[POINTER:.*]] = hivm.hir.pointer_cast(%[[ADDRESS]])
// CHECK:       %[[VIEW:.*]] = memref.reinterpret_cast %[[POINTER]] to offset: [1], sizes: [1], strides: [1]
// CHECK-SAME:  to memref<1xf32, strided<[1], offset: 1>>
// CHECK:       scf.while
// CHECK-SAME:  memref<1xf32, strided<[1], offset: 1>>
// CHECK:       scf.condition
// CHECK-SAME:  memref<1xf32, strided<[1], offset: 1>>
