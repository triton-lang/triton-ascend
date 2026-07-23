// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=1' -o - | FileCheck %s --check-prefix=GRAPH
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=1' -triton-to-linalg -o /dev/null

// GRAPH-LABEL: tt.func @lower_math_tanh(
// GRAPH: %{{.*}} = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// GRAPH: %{{.*}} = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// GRAPH: %{{.*}} = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// GRAPH: %[[TANH_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// GRAPH: %[[TANH_LOADED:.*]] = tt.load %[[TANH_SOURCE]] : tensor<2x2x!tt.ptr<f32>>
// GRAPH-NEXT: %[[TANH_RESULT:.*]] = math.tanh %[[TANH_LOADED]] : tensor<2x2xf32>
// GRAPH: %{{.*}} = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// GRAPH: %[[TANH_DESTINATION:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// GRAPH: tt.store %[[TANH_DESTINATION]], %[[TANH_RESULT]] : tensor<2x2x!tt.ptr<f32>>

tt.func @lower_math_tanh(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>) {
  %source_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_row_expanded = tt.expand_dims %source_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %source_column_expanded = tt.expand_dims %source_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %two = arith.constant 2 : i32
  %one = arith.constant 1 : i32
  %source_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %source_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %source_row_offsets = arith.muli %source_row_expanded, %source_row_stride : tensor<2x1xi32>
  %source_column_offsets = arith.muli %source_column_expanded, %source_column_stride : tensor<1x2xi32>
  %source_row_broadcast = tt.broadcast %source_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %source_column_broadcast = tt.broadcast %source_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %source_offsets = arith.addi %source_row_broadcast, %source_column_broadcast : tensor<2x2xi32>
  %source_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %source_addresses = tt.addptr %source_base_splat, %source_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %destination_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_row_expanded = tt.expand_dims %destination_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %destination_column_expanded = tt.expand_dims %destination_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %destination_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %destination_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %destination_row_offsets = arith.muli %destination_row_expanded, %destination_row_stride : tensor<2x1xi32>
  %destination_column_offsets = arith.muli %destination_column_expanded, %destination_column_stride : tensor<1x2xi32>
  %destination_row_broadcast = tt.broadcast %destination_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %destination_column_broadcast = tt.broadcast %destination_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %destination_offsets = arith.addi %destination_row_broadcast, %destination_column_broadcast : tensor<2x2xi32>
  %destination_base_splat = tt.splat %destination : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %destination_addresses = tt.addptr %destination_base_splat, %destination_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %loaded = tt.load %source_addresses : tensor<2x2x!tt.ptr<f32>>
  %result = math.tanh %loaded : tensor<2x2xf32>
  tt.store %destination_addresses, %result : tensor<2x2x!tt.ptr<f32>>
  tt.return
}

// GRAPH-LABEL: tt.func @lower_precise_sqrt(
// GRAPH: %{{.*}} = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// GRAPH: %{{.*}} = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// GRAPH: %{{.*}} = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// GRAPH: %[[SQRT_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// GRAPH: %[[SQRT_LOADED:.*]] = tt.load %[[SQRT_SOURCE]] : tensor<2x2x!tt.ptr<f32>>
// GRAPH-NEXT: %[[SQRT_RESULT:.*]] = tt.precise_sqrt %[[SQRT_LOADED]] : tensor<2x2xf32>
// GRAPH: %{{.*}} = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// GRAPH: %[[SQRT_DESTINATION:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
// GRAPH: tt.store %[[SQRT_DESTINATION]], %[[SQRT_RESULT]] : tensor<2x2x!tt.ptr<f32>>

tt.func @lower_precise_sqrt(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>) {
  %source_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_row_expanded = tt.expand_dims %source_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %source_column_expanded = tt.expand_dims %source_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %two = arith.constant 2 : i32
  %one = arith.constant 1 : i32
  %source_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %source_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %source_row_offsets = arith.muli %source_row_expanded, %source_row_stride : tensor<2x1xi32>
  %source_column_offsets = arith.muli %source_column_expanded, %source_column_stride : tensor<1x2xi32>
  %source_row_broadcast = tt.broadcast %source_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %source_column_broadcast = tt.broadcast %source_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %source_offsets = arith.addi %source_row_broadcast, %source_column_broadcast : tensor<2x2xi32>
  %source_base_splat = tt.splat %source : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %source_addresses = tt.addptr %source_base_splat, %source_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %destination_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_row_expanded = tt.expand_dims %destination_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %destination_column_expanded = tt.expand_dims %destination_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %destination_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %destination_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %destination_row_offsets = arith.muli %destination_row_expanded, %destination_row_stride : tensor<2x1xi32>
  %destination_column_offsets = arith.muli %destination_column_expanded, %destination_column_stride : tensor<1x2xi32>
  %destination_row_broadcast = tt.broadcast %destination_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %destination_column_broadcast = tt.broadcast %destination_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %destination_offsets = arith.addi %destination_row_broadcast, %destination_column_broadcast : tensor<2x2xi32>
  %destination_base_splat = tt.splat %destination : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %destination_addresses = tt.addptr %destination_base_splat, %destination_offsets : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>

  %loaded = tt.load %source_addresses : tensor<2x2x!tt.ptr<f32>>
  %result = tt.precise_sqrt %loaded : tensor<2x2xf32>
  tt.store %destination_addresses, %result : tensor<2x2x!tt.ptr<f32>>
  tt.return
}

// GRAPH-LABEL: tt.func @lower_math_absi(
// GRAPH: %{{.*}} = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<i32>>, tensor<2x2xi32>
// GRAPH: %{{.*}} = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<i32>>, tensor<2x2xi32>
// GRAPH: %{{.*}} = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// GRAPH: %[[ABSI_SOURCE:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<i32>>, tensor<2x2xi32>
// GRAPH: %[[ABSI_LOADED:.*]] = tt.load %[[ABSI_SOURCE]] : tensor<2x2x!tt.ptr<i32>>
// GRAPH-NEXT: %[[ABSI_RESULT:.*]] = math.absi %[[ABSI_LOADED]] : tensor<2x2xi32>
// GRAPH: %{{.*}} = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// GRAPH: %[[ABSI_DESTINATION:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<2x2x!tt.ptr<i32>>, tensor<2x2xi32>
// GRAPH: tt.store %[[ABSI_DESTINATION]], %[[ABSI_RESULT]] : tensor<2x2x!tt.ptr<i32>>

tt.func @lower_math_absi(%source: !tt.ptr<i32>, %destination: !tt.ptr<i32>) {
  %source_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %source_row_expanded = tt.expand_dims %source_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %source_column_expanded = tt.expand_dims %source_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %two = arith.constant 2 : i32
  %one = arith.constant 1 : i32
  %source_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %source_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %source_row_offsets = arith.muli %source_row_expanded, %source_row_stride : tensor<2x1xi32>
  %source_column_offsets = arith.muli %source_column_expanded, %source_column_stride : tensor<1x2xi32>
  %source_row_broadcast = tt.broadcast %source_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %source_column_broadcast = tt.broadcast %source_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %source_offsets = arith.addi %source_row_broadcast, %source_column_broadcast : tensor<2x2xi32>
  %source_base_splat = tt.splat %source : !tt.ptr<i32> -> tensor<2x2x!tt.ptr<i32>>
  %source_addresses = tt.addptr %source_base_splat, %source_offsets : tensor<2x2x!tt.ptr<i32>>, tensor<2x2xi32>

  %destination_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_column = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %destination_row_expanded = tt.expand_dims %destination_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %destination_column_expanded = tt.expand_dims %destination_column {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %destination_row_stride = tt.splat %two : i32 -> tensor<2x1xi32>
  %destination_column_stride = tt.splat %one : i32 -> tensor<1x2xi32>
  %destination_row_offsets = arith.muli %destination_row_expanded, %destination_row_stride : tensor<2x1xi32>
  %destination_column_offsets = arith.muli %destination_column_expanded, %destination_column_stride : tensor<1x2xi32>
  %destination_row_broadcast = tt.broadcast %destination_row_offsets : tensor<2x1xi32> -> tensor<2x2xi32>
  %destination_column_broadcast = tt.broadcast %destination_column_offsets : tensor<1x2xi32> -> tensor<2x2xi32>
  %destination_offsets = arith.addi %destination_row_broadcast, %destination_column_broadcast : tensor<2x2xi32>
  %destination_base_splat = tt.splat %destination : !tt.ptr<i32> -> tensor<2x2x!tt.ptr<i32>>
  %destination_addresses = tt.addptr %destination_base_splat, %destination_offsets : tensor<2x2x!tt.ptr<i32>>, tensor<2x2xi32>

  %loaded = tt.load %source_addresses : tensor<2x2x!tt.ptr<i32>>
  %result = math.absi %loaded : tensor<2x2xi32>
  tt.store %destination_addresses, %result : tensor<2x2x!tt.ptr<i32>>
  tt.return
}
