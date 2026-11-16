// RUN: triton-opt %s --triton-to-unstructure --triton-to-linalg --canonicalize --cse --verify-each | FileCheck %s --implicit-check-not='!tt.ptr' --implicit-check-not=unrealized_conversion_cast

// Boolean elements occupy one byte. Casting the tensor of pointers after
// addptr must retain its nonzero origin and its stride in bytes.
// CHECK-LABEL: func.func @bool_pointer_bitcast
// CHECK: memref.reinterpret_cast {{.*}} to offset: [3], sizes: [8], strides: [1] : memref<?xi8> to memref<8xi8, strided<[1], offset: 3>>
// CHECK: memref.reinterpret_cast {{.*}} to offset: [5], sizes: [8], strides: [1] : memref<?xi8> to memref<8xi8, strided<[1], offset: 5>>
// CHECK: memref.copy
// CHECK: bufferization.materialize_in_destination
// CHECK: return
module {
  tt.func public @bool_pointer_bitcast(%src: !tt.ptr<i1>, %dst: !tt.ptr<i1>) {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %three = arith.constant dense<3> : tensor<8xi32>
    %five = arith.constant dense<5> : tensor<8xi32>
    %zero = arith.constant dense<0> : tensor<8xi8>
    %src_offsets = arith.addi %r, %three : tensor<8xi32>
    %dst_offsets = arith.addi %r, %five : tensor<8xi32>
    %s = tt.splat %src : !tt.ptr<i1> -> tensor<8x!tt.ptr<i1>>
    %d = tt.splat %dst : !tt.ptr<i1> -> tensor<8x!tt.ptr<i1>>
    %sp = tt.addptr %s, %src_offsets : tensor<8x!tt.ptr<i1>>, tensor<8xi32>
    %dp = tt.addptr %d, %dst_offsets : tensor<8x!tt.ptr<i1>>, tensor<8xi32>
    %sb = tt.bitcast %sp : tensor<8x!tt.ptr<i1>> -> tensor<8x!tt.ptr<i8>>
    %db = tt.bitcast %dp : tensor<8x!tt.ptr<i1>> -> tensor<8x!tt.ptr<i8>>
    %v = tt.load %sb {was_bool_to_int8 = true} : tensor<8x!tt.ptr<i8>>
    %bool = arith.cmpi ne, %v, %zero : tensor<8xi8>
    %bytes = arith.extui %bool : tensor<8xi1> to tensor<8xi8>
    tt.store %db, %bytes : tensor<8x!tt.ptr<i8>>
    tt.return
  }

  // index_select retains a dynamic source dimension even with a constant
  // size operand. Its scalar bool cast must describe all 64 * 128 bytes.
  // CHECK-LABEL: func.func @bool_index_select
  // CHECK: %[[CAPACITY:.*]] = arith.constant 8192 : index
  // CHECK: %[[BASE:.*]] = hivm.hir.pointer_cast({{.*}}) [%[[CAPACITY]]] : memref<?xi8>
  // CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [0], sizes: [%{{.*}}, 128], strides: [128, 1]
  // CHECK: memref.subview %[[VIEW]]
  // CHECK: memref.copy {{.*}} {was_bool_to_int8 = true}
  // CHECK: bufferization.materialize_in_destination
  // CHECK: return
  tt.func public @bool_index_select(%src: !tt.ptr<i1>, %indices: !tt.ptr<i32>, %dst: !tt.ptr<i1>) {
    %zero = arith.constant 0 : index
    %minus_one = arith.constant -1 : index
    %rows = arith.constant 64 : index
    %cols = arith.constant 128 : index
    %ir = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %ib = tt.splat %indices : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ip = tt.addptr %ib, %ir : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %index = tt.load %ip : tensor<4x!tt.ptr<i32>>
    %bytes = tt.bitcast %src : !tt.ptr<i1> -> !tt.ptr<i8>
    %value = ascend.index_select_simd %bytes, %index, 0, [%rows, %cols], [%minus_one, %zero], [-1, 128] {was_bool_to_int8 = true} : !tt.ptr<i8>, tensor<4xi32> -> tensor<4x128xi8>
    %range = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32>
    %offset = tt.reshape %range : tensor<512xi32> -> tensor<4x128xi32>
    %out = tt.splat %dst : !tt.ptr<i1> -> tensor<4x128x!tt.ptr<i1>>
    %ptr = tt.addptr %out, %offset : tensor<4x128x!tt.ptr<i1>>, tensor<4x128xi32>
    %byte_ptr = tt.bitcast %ptr : tensor<4x128x!tt.ptr<i1>> -> tensor<4x128x!tt.ptr<i8>>
    tt.store %byte_ptr, %value : tensor<4x128x!tt.ptr<i8>>
    tt.return
  }
}
