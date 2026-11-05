// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=false' --split-input-file --verify-each %s | FileCheck %s
// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true' --split-input-file --verify-each %s | FileCheck %s

// bool 指针进入循环前被广播，随后 bitcast 为 i8 并执行 load。load 必须保持在
// 原始的小形状上，避免零 stride 进入 linalg；同时 bool 转换属性需要在广播
// 重写和指针分析后保留。
module {
  tt.func public @loop_broadcast_bool_bitcast_load(%input: !tt.ptr<i1>, %output: !tt.ptr<i1>, %trip: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %trip_index = arith.index_cast %trip : i32 to index
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_offsets = tt.expand_dims %rows {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %input_splat = tt.splat %input : !tt.ptr<i1> -> tensor<1x16x!tt.ptr<i1>>
    %input_ptrs = tt.addptr %input_splat, %row_offsets : tensor<1x16x!tt.ptr<i1>>, tensor<1x16xi32>
    %input_broadcast = tt.broadcast %input_ptrs : tensor<1x16x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i1>>
    %trip_splat = tt.splat %trip : i32 -> tensor<16xi32>
    %mask_rows = arith.cmpi slt, %rows, %trip_splat : tensor<16xi32>
    %mask_rows_2d = tt.expand_dims %mask_rows {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %mask = tt.broadcast %mask_rows_2d : tensor<1x16xi1> -> tensor<16x16xi1>
    %other = arith.constant dense<0> : tensor<16x16xi8>
    %output_splat = tt.splat %output : !tt.ptr<i1> -> tensor<16x16x!tt.ptr<i1>>
    %output_offsets = tt.broadcast %row_offsets : tensor<1x16xi32> -> tensor<16x16xi32>
    %output_ptrs = tt.addptr %output_splat, %output_offsets : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
    %step = tt.splat %c1_i32 : i32 -> tensor<16x16xi32>
    %result = scf.for %iv = %c0_index to %trip_index step %c1_index
        iter_args(%ptr = %input_broadcast) -> tensor<16x16x!tt.ptr<i1>> {
      %as_i8 = tt.bitcast %ptr : tensor<16x16x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i8>>
      %loaded = tt.load %as_i8, %mask, %other {was_bool_to_int8 = true} : tensor<16x16x!tt.ptr<i8>>
      %as_bool = arith.trunci %loaded : tensor<16x16xi8> to tensor<16x16xi1>
      tt.store %output_ptrs, %as_bool : tensor<16x16x!tt.ptr<i1>>
      %next = tt.addptr %ptr, %step : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
      scf.yield %next : tensor<16x16x!tt.ptr<i1>>
    }
    tt.return
  }
}

// CHECK-LABEL: tt.func public @loop_broadcast_bool_bitcast_load
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[ITER:.*]] = %{{.*}}) -> (i32) {
// CHECK: %[[OFFSET:.*]] = tt.splat %[[ITER]] : i32 -> tensor<1x16xi32>
// CHECK: %[[PTR:.*]] = tt.addptr %{{.*}}, %[[OFFSET]] : tensor<1x16x!tt.ptr<i1>>, tensor<1x16xi32>
// CHECK: %[[BITCAST:.*]] = tt.bitcast %[[PTR]] : tensor<1x16x!tt.ptr<i1>> -> tensor<1x16x!tt.ptr<i8>>
// CHECK: %[[LOAD:.*]] = tt.load %[[BITCAST]], %{{.*}}, %{{.*}} {was_bool_to_int8 = true} : tensor<1x16x!tt.ptr<i8>>
// CHECK: tt.broadcast %[[LOAD]] : tensor<1x16xi8> -> tensor<16x16xi8>
// CHECK: arith.addi %[[ITER]], %{{.*}} : i32
// CHECK-NOT: tt.load {{.*}} : tensor<16x16x!tt.ptr<i8>>

// -----

// CFO 会把广播指针重建为完整 [16x16] addptr，偏移形如
// broadcast([1x16]) + splat(uniform)。转换后应重新在 [1x16] 上更新指针并
// 加载，避免把广播轴变成零 stride。
module {
  tt.func public @loop_addptr_then_broadcast_bool_load(%input: !tt.ptr<i1>, %output: !tt.ptr<i8>, %trip: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %trip_index = arith.index_cast %trip : i32 to index
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_offsets = tt.expand_dims %rows {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %full_row_offsets = tt.broadcast %row_offsets : tensor<1x16xi32> -> tensor<16x16xi32>
    %input_splat_full = tt.splat %input : !tt.ptr<i1> -> tensor<16x16x!tt.ptr<i1>>
    %ptrs = tt.addptr %input_splat_full, %full_row_offsets : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
    %uniform = tt.splat %trip : i32 -> tensor<16x16xi32>
    %offset = arith.addi %full_row_offsets, %uniform : tensor<16x16xi32>
    %offset_ptrs = tt.addptr %input_splat_full, %offset : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
    %trip_splat = tt.splat %trip : i32 -> tensor<16xi32>
    %mask_rows = arith.cmpi slt, %rows, %trip_splat : tensor<16xi32>
    %mask_rows_2d = tt.expand_dims %mask_rows {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %mask = tt.broadcast %mask_rows_2d : tensor<1x16xi1> -> tensor<16x16xi1>
    %other = arith.constant dense<0> : tensor<16x16xi8>
    %output_splat = tt.splat %output : !tt.ptr<i8> -> tensor<16x16x!tt.ptr<i8>>
    %output_ptrs = tt.addptr %output_splat, %full_row_offsets : tensor<16x16x!tt.ptr<i8>>, tensor<16x16xi32>
    %result = scf.for %iv = %c0_index to %trip_index step %c1_index
        iter_args(%ptr = %offset_ptrs) -> tensor<16x16x!tt.ptr<i1>> {
      %as_i8 = tt.bitcast %ptr : tensor<16x16x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i8>>
      %loaded = tt.load %as_i8, %mask, %other {was_bool_to_int8 = true} : tensor<16x16x!tt.ptr<i8>>
      tt.store %output_ptrs, %loaded, %mask : tensor<16x16x!tt.ptr<i8>>
      %next = tt.addptr %ptr, %uniform : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
      scf.yield %next : tensor<16x16x!tt.ptr<i1>>
    }
    tt.return
  }
}

// CHECK-LABEL: tt.func public @loop_addptr_then_broadcast_bool_load
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[ITER:.*]] = %{{.*}}) -> (i32) {
// CHECK: %[[OFFSET:.*]] = tt.splat %[[ITER]] : i32 -> tensor<1x16xi32>
// CHECK: %[[PTR:.*]] = tt.addptr %{{.*}}, %[[OFFSET]] : tensor<1x16x!tt.ptr<i1>>, tensor<1x16xi32>
// CHECK: %[[BITCAST:.*]] = tt.bitcast %[[PTR]] : tensor<1x16x!tt.ptr<i1>> -> tensor<1x16x!tt.ptr<i8>>
// CHECK: %[[LOAD:.*]] = tt.load %[[BITCAST]], %{{.*}}, %{{.*}} {was_bool_to_int8 = true} : tensor<1x16x!tt.ptr<i8>>
// CHECK: arith.addi %[[ITER]], %{{.*}} : i32
// CHECK-NOT: tt.load {{.*}} : tensor<16x16x!tt.ptr<i8>>

// -----

// 两个正交广播相加时，指针广播轴会形成零 stride。转换后应以标量指针广播
// 加完整偏移，避免广播指针参与 addptr。
module {
  tt.func public @orthogonal_broadcast_addptr(%base: !tt.ptr<i1>) {
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %row_offsets = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %col_offsets = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %base_rows = tt.splat %base : !tt.ptr<i1> -> tensor<16x1x!tt.ptr<i1>>
    %row_ptrs = tt.addptr %base_rows, %row_offsets : tensor<16x1x!tt.ptr<i1>>, tensor<16x1xi32>
    %broadcast_ptrs = tt.broadcast %row_ptrs : tensor<16x1x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i1>>
    %broadcast_offsets = tt.broadcast %col_offsets : tensor<1x16xi32> -> tensor<16x16xi32>
    %ptrs = tt.addptr %broadcast_ptrs, %broadcast_offsets : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
    %as_i8 = tt.bitcast %ptrs : tensor<16x16x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i8>>
    %loaded = tt.load %as_i8 : tensor<16x16x!tt.ptr<i8>>
    %as_bool = arith.trunci %loaded : tensor<16x16xi8> to tensor<16x16xi1>
    tt.store %ptrs, %as_bool : tensor<16x16x!tt.ptr<i1>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @orthogonal_broadcast_addptr
// CHECK: %[[BASE:.*]] = tt.splat %{{.*}} : !tt.ptr<i1> -> tensor<16x16x!tt.ptr<i1>>
// CHECK: %[[ROW:.*]] = tt.broadcast %{{.*}} : tensor<16x1xi32> -> tensor<16x16xi32>
// CHECK: %[[COL:.*]] = tt.broadcast %{{.*}} : tensor<1x16xi32> -> tensor<16x16xi32>
// CHECK: %[[FULL_OFFSET:.*]] = arith.addi %[[ROW]], %[[COL]] : tensor<16x16xi32>
// CHECK: tt.addptr %[[BASE]], %[[FULL_OFFSET]] : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
// CHECK-NOT: tt.broadcast %{{.*}} : tensor<16x1x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i1>>
