// RUN: triton-opt %s --bubble-up-operation --canonicalize --cse | FileCheck %s

// The original start is part of every lane, including singleton slices.
// CHECK-LABEL: tt.func @singleton_nonzero_start
// CHECK: %[[START:.*]] = arith.constant 16 : i32
// CHECK: %[[OFFSET:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK: %[[BASE:.*]] = arith.addi %[[OFFSET]], %[[START]] : i32
// CHECK: tt.splat %[[BASE]] : i32 -> tensor<1xi32>
tt.func @singleton_nonzero_start(%offset: index) -> tensor<1xi32> {
  %range = tt.make_range {start = 16 : i32, end = 80 : i32} : tensor<64xi32>
  %slice = tensor.extract_slice %range[%offset] [1] [1] : tensor<64xi32> to tensor<1xi32>
  tt.return %slice : tensor<1xi32>
}

// A multi-lane slice is offset + start + arange(size) * stride, not a splat.
// CHECK-LABEL: tt.func @multiple_lanes_static_stride
// CHECK-DAG: %[[START:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[STRIDE:.*]] = arith.constant dense<3> : tensor<8xi32>
// CHECK: %[[OFFSET:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK: %[[BASE:.*]] = arith.addi %[[OFFSET]], %[[START]] : i32
// CHECK: %[[BASES:.*]] = tt.splat %[[BASE]] : i32 -> tensor<8xi32>
// CHECK: %[[RANGE:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK: %[[SCALED:.*]] = arith.muli %[[RANGE]], %[[STRIDE]] : tensor<8xi32>
// CHECK: %[[RESULT:.*]] = arith.addi %[[BASES]], %[[SCALED]] : tensor<8xi32>
// CHECK: tt.return %[[RESULT]] : tensor<8xi32>
tt.func @multiple_lanes_static_stride(%offset: index) -> tensor<8xi32> {
  %range = tt.make_range {start = 16 : i32, end = 80 : i32} : tensor<64xi32>
  %slice = tensor.extract_slice %range[%offset] [8] [3] : tensor<64xi32> to tensor<8xi32>
  tt.return %slice : tensor<8xi32>
}

// CHECK-LABEL: tt.func @multiple_lanes_dynamic_stride
// CHECK: arith.index_cast %{{.*}} : index to i32
// CHECK: tt.splat %{{.*}} : i32 -> tensor<8xi32>
// CHECK: %[[RANGE:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK: %[[STRIDE:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK: %[[STRIDES:.*]] = tt.splat %[[STRIDE]] : i32 -> tensor<8xi32>
// CHECK: %[[SCALED:.*]] = arith.muli %[[RANGE]], %[[STRIDES]] : tensor<8xi32>
// CHECK: arith.addi %{{.*}}, %[[SCALED]] : tensor<8xi32>
tt.func @multiple_lanes_dynamic_stride(%offset: index, %stride: index) -> tensor<8xi32> {
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %slice = tensor.extract_slice %range[%offset] [8] [%stride] : tensor<64xi32> to tensor<8xi32>
  tt.return %slice : tensor<8xi32>
}

// MLIR slices can have non-power-of-two extents even though the original
// frontend arange has a power-of-two extent. Preserve those lanes as well.
// CHECK-LABEL: tt.func @non_power_of_two_slice
// CHECK: %[[BASE:.*]] = tt.splat %{{.*}} : i32 -> tensor<3xi32>
// CHECK: %[[RANGE:.*]] = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
// CHECK: %[[RESULT:.*]] = arith.addi %[[BASE]], %[[RANGE]] : tensor<3xi32>
// CHECK: tt.return %[[RESULT]] : tensor<3xi32>
tt.func @non_power_of_two_slice(%offset: index) -> tensor<3xi32> {
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %slice = tensor.extract_slice %range[%offset] [3] [1] : tensor<64xi32> to tensor<3xi32>
  tt.return %slice : tensor<3xi32>
}

// A dropped singleton dimension still contains the original range element.
// CHECK-LABEL: tt.func @rank_reduced_singleton
// CHECK: arith.constant 16 : i32
// CHECK: arith.index_cast %{{.*}} : index to i32
// CHECK: %[[BASE:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK: tt.splat %[[BASE]] : i32 -> tensor<i32>
tt.func @rank_reduced_singleton(%offset: index) -> tensor<i32> {
  %range = tt.make_range {start = 16 : i32, end = 80 : i32} : tensor<64xi32>
  %slice = tensor.extract_slice %range[%offset] [1] [1] : tensor<64xi32> to tensor<i32>
  tt.return %slice : tensor<i32>
}

// Dynamic extents cannot be expressed by a new tt.make_range. Keep the slice.
// CHECK-LABEL: tt.func @dynamic_extent_preserved
// CHECK: %[[RANGE:.*]] = tt.make_range
// CHECK: tensor.extract_slice %[[RANGE]][%{{.*}}] [%{{.*}}] [1]
tt.func @dynamic_extent_preserved(%offset: index, %size: index) -> tensor<?xi32> {
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %slice = tensor.extract_slice %range[%offset] [%size] [1] : tensor<64xi32> to tensor<?xi32>
  tt.return %slice : tensor<?xi32>
}
