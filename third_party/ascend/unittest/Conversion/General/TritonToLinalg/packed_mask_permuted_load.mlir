// RUN: triton-opt %s -triton-to-linalg="compile-on-910-95=true" -verify-each | FileCheck %s

// Exercise the memory-converter handoff after strided-access inspection.
// Packed views must not enter ordinary implicit-transpose annotation, even
// when a dot in the module enables that optimization for other loads.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @permuted_packed
  // CHECK-NOT: MayImplicitTransposeWithLastAxis
  // CHECK: memref.alloc() {{.*}}alignment = 32
  // CHECK: memref.view
  // CHECK: memref.subview
  // CHECK: hfusion.packed_mask_view
  // CHECK-NOT: MayImplicitTransposeWithLastAxis
  tt.func public @permuted_packed(%ptr: !tt.ptr<i8>) -> (tensor<32x128xi1>, tensor<16x16xf32>) {
    %rows = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %cols = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %c = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %stride = arith.constant dense<32> : tensor<1x16xi32>
    %co = arith.muli %c, %stride : tensor<1x16xi32>
    %rr = tt.broadcast %r : tensor<32x1xi32> -> tensor<32x16xi32>
    %cc = tt.broadcast %co : tensor<1x16xi32> -> tensor<32x16xi32>
    %off = arith.addi %rr, %cc : tensor<32x16xi32>
    %base = tt.splat %ptr : !tt.ptr<i8> -> tensor<32x16x!tt.ptr<i8>>
    %ptrs = tt.addptr %base, %off : tensor<32x16x!tt.ptr<i8>>, tensor<32x16xi32>
    %packed = tt.load %ptrs {InspectedByStridedLoadStoreRewrite} : tensor<32x16x!tt.ptr<i8>>
    %keep = hfusion.packed_mask_view %packed : tensor<32x16xi8> -> tensor<32x128xi1>
    %q = arith.constant dense<1.0> : tensor<16x16xf16>
    %acc = arith.constant dense<0.0> : tensor<16x16xf32>
    %score = tt.dot %q, %q, %acc : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    tt.return %keep, %score : tensor<32x128xi1>, tensor<16x16xf32>
  }

  // CHECK-LABEL: func.func @masked_permuted_packed
  // CHECK-NOT: MayImplicitTransposeWithLastAxis
  // CHECK: memref.alloc() {{.*}}alignment = 32
  // CHECK: memref.view
  // CHECK: memref.subview
  // CHECK: hfusion.packed_mask_view
  // CHECK-NOT: MayImplicitTransposeWithLastAxis
  tt.func public @masked_permuted_packed(%ptr: !tt.ptr<i8>) -> (tensor<32x128xi1>, tensor<16x16xf32>) {
    %rows = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %cols = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %c = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %stride = arith.constant dense<32> : tensor<1x16xi32>
    %co = arith.muli %c, %stride : tensor<1x16xi32>
    %rr = tt.broadcast %r : tensor<32x1xi32> -> tensor<32x16xi32>
    %cc = tt.broadcast %co : tensor<1x16xi32> -> tensor<32x16xi32>
    %off = arith.addi %rr, %cc : tensor<32x16xi32>
    %base = tt.splat %ptr : !tt.ptr<i8> -> tensor<32x16x!tt.ptr<i8>>
    %ptrs = tt.addptr %base, %off : tensor<32x16x!tt.ptr<i8>>, tensor<32x16xi32>
    %limit = arith.constant dense<16> : tensor<32x1xi32>
    %row_mask = arith.cmpi slt, %r, %limit : tensor<32x1xi32>
    %mask = tt.broadcast %row_mask : tensor<32x1xi1> -> tensor<32x16xi1>
    %zero = arith.constant dense<0> : tensor<32x16xi8>
    %packed = tt.load %ptrs, %mask, %zero {InspectedByStridedLoadStoreRewrite} : tensor<32x16x!tt.ptr<i8>>
    %keep = hfusion.packed_mask_view %packed : tensor<32x16xi8> -> tensor<32x128xi1>
    %q = arith.constant dense<1.0> : tensor<16x16xf16>
    %acc = arith.constant dense<0.0> : tensor<16x16xf32>
    %score = tt.dot %q, %q, %acc : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    tt.return %keep, %score : tensor<32x128xi1>, tensor<16x16xf32>
  }
}
