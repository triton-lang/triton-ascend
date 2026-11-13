// RUN: triton-opt --split-input-file --merge-outer-vector-block %s | FileCheck %s

// Retain the real Q-load shape, optional tail zero-fill, and partial copy.
// The load remains under first-event control even if the KV work guard is false.
// Task-tensor slicing can also leave a dead GM metadata tail after to_tensor.
// CHECK-LABEL: func.func @padded_q_load
// CHECK: iter_args(%{{.*}} = %{{.*}}, %[[OLD:.*]] = %{{.*}})
// CHECK: %[[FIRST:.*]] = arith.cmpi eq
// CHECK: %[[Q:.*]] = scf.if %[[FIRST]]
// CHECK: %[[BUF:.*]] = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"}
// CHECK: scf.if
// CHECK: linalg.fill {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"}
// CHECK-SAME: outs(%[[BUF]] : memref<128x128xbf16>)
// CHECK: memref.subview
// CHECK: memref.subview %[[BUF]]
// CHECK: memref.copy {{.*}}ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"
// CHECK: bufferization.to_tensor %[[BUF]] restrict writable
// CHECK: memref.load {{.*}}ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"
// CHECK: arith.minsi
// CHECK: } else {
// CHECK: scf.yield {{.*}}%[[OLD]]
// CHECK: } {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"}
// CHECK: scf.if %arg7
// CHECK: linalg.matmul {{.*}}ins(%[[Q]], %arg1
// CHECK: scf.yield {{.*}}%[[Q]]
// CHECK: ssbuffer.core_type = "UNDETERMINED, CUBE"
func.func @padded_q_load(%q: memref<?xbf16>, %k: tensor<128x128xbf16>, %acc: tensor<128x128xf32>, %n: index, %valid: index, %offset: index, %stride: index, %work: i1, %metadata: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %zero = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<128x128xbf16>
  %result:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%phase = %c0, %old = %empty) -> (index, tensor<128x128xbf16>) {
    %first = arith.cmpi eq, %phase, %c0 {ssbuffer.block_id = 5 : i32, ssbuffer.outer_event_control} : index
    %loaded = scf.if %first -> (tensor<128x128xbf16>) {
      %view = memref.reinterpret_cast %q to offset: [%offset], sizes: [128, 128], strides: [%stride, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
      %buffer = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
      %pad = arith.cmpi slt, %valid, %c128 : index
      scf.if %pad {
        linalg.fill {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} ins(%zero : bf16) outs(%buffer : memref<128x128xbf16>)
      } {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"}
      %src = memref.subview %view[0, 0] [%valid, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
      %dst = memref.subview %buffer[0, 0] [%valid, 128] [1, 1] : memref<128x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
      memref.copy %src, %dst {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
      %tensor = bufferization.to_tensor %buffer restrict writable {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>
      %meta_view = memref.reinterpret_cast %metadata to offset: [%offset], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %count = memref.load %meta_view[%c0] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : memref<1xi32, strided<[1], offset: ?>>
      %limit = arith.index_cast %valid : index to i32
      %clamped = arith.minsi %count, %limit : i32
      scf.yield %tensor : tensor<128x128xbf16>
    } else {
      scf.yield %old : tensor<128x128xbf16>
    } {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"}
    scf.if %work {
      %product = linalg.matmul {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} ins(%loaded, %k : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    } {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"}
    scf.yield %c1, %loaded : index, tensor<128x128xbf16>
  } {ssbuffer.core_type = "UNDETERMINED, VECTOR", ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}

// -----

// Private writes alone do not prove Cube legality: the loaded buffer is
// transformed by a Vector computation before becoming the carried tensor.
// CHECK-LABEL: func.func @private_vector_computation_rejected
// CHECK: %[[BUF:.*]] = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: memref.copy
// CHECK: linalg.generic
// CHECK: arith.negf
// CHECK: bufferization.to_tensor %[[BUF]] restrict writable
// CHECK: } {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"}
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"
// CHECK: ssbuffer.core_type = "UNDETERMINED, VECTOR"
func.func @private_vector_computation_rejected(%q: memref<128x128xbf16>, %k: tensor<128x128xbf16>, %acc: tensor<128x128xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<128x128xbf16>
  %result:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%phase = %c0, %old = %empty) -> (index, tensor<128x128xbf16>) {
    %first = arith.cmpi eq, %phase, %c0 {ssbuffer.block_id = 5 : i32, ssbuffer.outer_event_control} : index
    %loaded = scf.if %first -> (tensor<128x128xbf16>) {
      %buffer = memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
      memref.copy %q, %buffer {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to memref<128x128xbf16>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%buffer : memref<128x128xbf16>) attrs = {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} {
      ^bb0(%value: bf16):
        %negated = arith.negf %value : bf16
        linalg.yield %negated : bf16
      }
      %tensor = bufferization.to_tensor %buffer restrict writable {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>
      scf.yield %tensor : tensor<128x128xbf16>
    } else {
      scf.yield %old : tensor<128x128xbf16>
    } {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"}
    %product = linalg.matmul {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "CUBE"} ins(%loaded, %k : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %c1, %loaded : index, tensor<128x128xbf16>
  } {ssbuffer.core_type = "UNDETERMINED, VECTOR", ssbuffer.explicit_main_loop, ssbuffer.outer_event_control_count = 1 : i32}
  return
}
