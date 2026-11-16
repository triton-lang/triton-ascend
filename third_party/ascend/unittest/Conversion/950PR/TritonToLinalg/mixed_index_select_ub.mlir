// RUN: triton-opt %s --triton-to-structured --triton-to-unstructure='compile-on-910-95=true compile-mode=simd' --bubble-up-operation --triton-to-structured --triton-to-linalg='compile-on-910-95=true compile-mode=simd_simt' | FileCheck %s --check-prefix=MIX
// RUN: triton-opt %s --triton-to-structured --triton-to-unstructure='compile-on-910-95=true compile-mode=simd' --bubble-up-operation --triton-to-structured --triton-to-linalg='compile-on-910-95=true compile-mode=simd' | FileCheck %s --check-prefix=SIMD
// RUN: triton-opt %s --triton-to-structured --triton-to-unstructure='compile-on-910-95=false compile-mode=simd' --bubble-up-operation --triton-to-structured --triton-to-linalg='compile-on-910-95=false compile-mode=simd' | FileCheck %s --check-prefix=OLD

// A row gather carries vector-produced indices into its extracted scalar loop.
// In Mixed mode its result must stay in UB until CV communication is inserted;
// inferring L1 from a later Cube consumer also moves the scalar index read to
// Cube and can produce an illegal internal-buffer load.
// MIX-LABEL: func.func @gather_rows
// MIX: memref.alloc() : memref<4x16xf32, #hivm.address_space<ub>>
// MIX: scf.for
// MIX: tensor.extract
// MIX: memref.copy
// SIMD-LABEL: func.func @gather_rows
// SIMD-NOT: #hivm.address_space<ub>
// SIMD: memref.alloc() : memref<4x16xf32>
// OLD-LABEL: func.func @gather_rows
// OLD-NOT: #hivm.address_space<ub>
// OLD: memref.alloc() : memref<4x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @gather_rows(%src: !tt.ptr<f32>, %indices: !tt.ptr<i32>,
                              %dst: !tt.ptr<f32>) {
    %rows = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %cols = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %idx_base = tt.splat %indices : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %idx_ptr = tt.addptr %idx_base, %rows : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %idx = tt.load %idx_ptr : tensor<4x!tt.ptr<i32>>
    %sixteen = arith.constant dense<16> : tensor<4xi32>
    %row_offset = arith.muli %idx, %sixteen : tensor<4xi32>
    %r = tt.expand_dims %row_offset {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %c = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %rb = tt.broadcast %r : tensor<4x1xi32> -> tensor<4x16xi32>
    %cb = tt.broadcast %c : tensor<1x16xi32> -> tensor<4x16xi32>
    %offset = arith.addi %rb, %cb : tensor<4x16xi32>
    %base = tt.splat %src : !tt.ptr<f32> -> tensor<4x16x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offset : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    %value = tt.load %ptr : tensor<4x16x!tt.ptr<f32>>
    %out_row = arith.muli %rows, %sixteen : tensor<4xi32>
    %out_r = tt.expand_dims %out_row {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %out_rb = tt.broadcast %out_r : tensor<4x1xi32> -> tensor<4x16xi32>
    %out_offset = arith.addi %out_rb, %cb : tensor<4x16xi32>
    %out_base = tt.splat %dst : !tt.ptr<f32> -> tensor<4x16x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_base, %out_offset : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    tt.store %out_ptr, %value : tensor<4x16x!tt.ptr<f32>>
    tt.return
  }
}
