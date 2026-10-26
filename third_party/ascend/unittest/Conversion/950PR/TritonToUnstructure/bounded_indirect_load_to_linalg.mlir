// RUN: triton-opt %s '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' --bubble-up-operation '--triton-to-linalg=compile-on-910-95=True compile-mode=simd_simt_template' | FileCheck %s

// Keep the full tile coordinates and disjoint row-loop contract through T2L.
// The backend must see the generated load iteration domain before distributing
// the load and a later consumer between sub-blocks.
// CHECK-LABEL: func.func @chunked_load_iteration_domain
// CHECK-NOT: tensor<128x256xi64>
// CHECK: %[[FULL:.*]] = scf.for
// CHECK: func.call @triton_indirect_load{{.*}}tensor<32x256xi64>{{.*}} -> tensor<32x256xi8>
// CHECK: tensor.insert_slice {{.*}}[{{.*}}, 0] [32, 256] [1, 1]
// CHECK: } {ExtractedLoadOrStore, hivm.parallel_loop}
// CHECK: bufferization.materialize_in_destination %[[FULL]]
// CHECK: return
tt.func public @chunked_load_iteration_domain(%src: !tt.ptr<i8>, %dst: !tt.ptr<i8>, %n: i32, %stride: i32) {
  %rows = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %cols = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %nvec = tt.splat %n : i32 -> tensor<256xi32>
  %wrapped = arith.remsi %cols, %nvec : tensor<256xi32>
  %row2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %strides = tt.splat %stride : i32 -> tensor<128x1xi32>
  %scaled = arith.muli %row2d, %strides : tensor<128x1xi32>
  %col2d = tt.expand_dims %wrapped {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
  %brows = tt.broadcast %scaled : tensor<128x1xi32> -> tensor<128x256xi32>
  %bcols = tt.broadcast %col2d : tensor<1x256xi32> -> tensor<128x256xi32>
  %offset = arith.addi %brows, %bcols : tensor<128x256xi32>
  %base = tt.splat %src : !tt.ptr<i8> -> tensor<128x256x!tt.ptr<i8>>
  %ptr = tt.addptr %base, %offset : tensor<128x256x!tt.ptr<i8>>, tensor<128x256xi32>
  %value = tt.load %ptr : tensor<128x256x!tt.ptr<i8>>
  %out_stride = arith.constant dense<256> : tensor<128x1xi32>
  %out_rows = arith.muli %row2d, %out_stride : tensor<128x1xi32>
  %out_cols = tt.expand_dims %cols {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
  %out_brows = tt.broadcast %out_rows : tensor<128x1xi32> -> tensor<128x256xi32>
  %out_bcols = tt.broadcast %out_cols : tensor<1x256xi32> -> tensor<128x256xi32>
  %out_offset = arith.addi %out_brows, %out_bcols : tensor<128x256xi32>
  %out_base = tt.splat %dst : !tt.ptr<i8> -> tensor<128x256x!tt.ptr<i8>>
  %out_ptr = tt.addptr %out_base, %out_offset : tensor<128x256x!tt.ptr<i8>>, tensor<128x256xi32>
  tt.store %out_ptr, %value : tensor<128x256x!tt.ptr<i8>>
  tt.return
}
