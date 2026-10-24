// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true compile-mode=simd_simt_template' --split-input-file --verify-each | FileCheck %s --check-prefix=UNSTRUCTURE
// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true compile-mode=simd_simt_template' --triton-to-linalg='compile-on-910-95=true compile-mode=simd_simt_template named-ops=true' --split-input-file --verify-each | FileCheck %s --check-prefix=LINALG

// An explicit volatile tt.load must retain its full snapshot when lowered
// through an indirect read. The nonvolatile control remains eligible for tiling.
// A nonmonotonic column permutation selects the indirect-load conversion.

// UNSTRUCTURE-LABEL: tt.func public @indirect_kernel_false
// UNSTRUCTURE: ascend.indirect_load {{.*}} {isVolatile = false} -> tensor<64x64xf16>
// LINALG-LABEL: func.func @indirect_kernel_false
// LINALG-NOT: call @triton_indirect_load{{.*}} -> tensor<64x64xf16>
// LINALG: scf.for
// LINALG: call @triton_indirect_load{{.*}} {isVolatile = false} : {{.*}} -> tensor<32x32xf16>
// LINALG: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
   tt.func public @indirect_kernel_false(%src: !tt.ptr<f16>, %dst: !tt.ptr<f32>) {
   %c64 = arith.constant dense<64> : tensor<64x1xi32>
   %c17 = arith.constant dense<17> : tensor<1x64xi32>
   %cmod = arith.constant dense<64> : tensor<1x64xi32>
   %o64 = arith.constant dense<64> : tensor<16x1xi32>
   %a = arith.constant dense<1.0> : tensor<16x64xf16>
   %c = arith.constant dense<0.0> : tensor<16x64xf32>
   %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
   %col = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
   %row = tt.expand_dims %r {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
   %rowoff = arith.muli %row, %c64 : tensor<64x1xi32>
   %mul = arith.muli %col, %c17 : tensor<1x64xi32>
   %perm = arith.remsi %mul, %cmod : tensor<1x64xi32>
   %cols = tt.broadcast %perm : tensor<1x64xi32> -> tensor<64x64xi32>
   %rows = tt.broadcast %rowoff : tensor<64x1xi32> -> tensor<64x64xi32>
   %off = arith.addi %rows, %cols : tensor<64x64xi32>
   %base = tt.splat %src : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
   %ptr = tt.addptr %base, %off : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
   %b = tt.load %ptr {isVolatile = false} : tensor<64x64x!tt.ptr<f16>>
   %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
   %m = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
   %mr = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
   %mo = arith.muli %mr, %o64 : tensor<16x1xi32>
   %mrwide = tt.broadcast %mo : tensor<16x1xi32> -> tensor<16x64xi32>
   %ncwide = tt.broadcast %col : tensor<1x64xi32> -> tensor<16x64xi32>
   %doff = arith.addi %mrwide, %ncwide : tensor<16x64xi32>
   %dbase = tt.splat %dst : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
   %dptr = tt.addptr %dbase, %doff : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
   tt.store %dptr, %d : tensor<16x64x!tt.ptr<f32>>
   tt.return
   }
}

// -----

// UNSTRUCTURE-LABEL: tt.func public @indirect_kernel_true
// The indirect op defaults to true, which its custom printer elides.
// UNSTRUCTURE: ascend.indirect_load {{.*}} : tensor<64x64xi64> -> tensor<64x64xf16>
// LINALG-LABEL: func.func @indirect_kernel_true
// LINALG: %[[SNAPSHOT:[a-zA-Z0-9_]+]] = call @triton_indirect_load{{.*}} {isVolatile = true} : {{.*}} -> tensor<64x64xf16>
// LINALG: scf.for
// LINALG: tensor.extract_slice %[[SNAPSHOT]]{{.*}} : tensor<64x64xf16> to tensor<32x32xf16>
// LINALG-NOT: call @triton_indirect_load
// LINALG: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
   tt.func public @indirect_kernel_true(%src: !tt.ptr<f16>, %dst: !tt.ptr<f32>) {
   %c64 = arith.constant dense<64> : tensor<64x1xi32>
   %c17 = arith.constant dense<17> : tensor<1x64xi32>
   %cmod = arith.constant dense<64> : tensor<1x64xi32>
   %o64 = arith.constant dense<64> : tensor<16x1xi32>
   %a = arith.constant dense<1.0> : tensor<16x64xf16>
   %c = arith.constant dense<0.0> : tensor<16x64xf32>
   %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
   %col = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
   %row = tt.expand_dims %r {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
   %rowoff = arith.muli %row, %c64 : tensor<64x1xi32>
   %mul = arith.muli %col, %c17 : tensor<1x64xi32>
   %perm = arith.remsi %mul, %cmod : tensor<1x64xi32>
   %cols = tt.broadcast %perm : tensor<1x64xi32> -> tensor<64x64xi32>
   %rows = tt.broadcast %rowoff : tensor<64x1xi32> -> tensor<64x64xi32>
   %off = arith.addi %rows, %cols : tensor<64x64xi32>
   %base = tt.splat %src : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
   %ptr = tt.addptr %base, %off : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
   %b = tt.load %ptr {isVolatile = true} : tensor<64x64x!tt.ptr<f16>>
   %d = tt.dot_scaled %a, %b, %c lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x64xf16> * tensor<64x64xf16> -> tensor<16x64xf32>
   %m = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
   %mr = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
   %mo = arith.muli %mr, %o64 : tensor<16x1xi32>
   %mrwide = tt.broadcast %mo : tensor<16x1xi32> -> tensor<16x64xi32>
   %ncwide = tt.broadcast %col : tensor<1x64xi32> -> tensor<16x64xi32>
   %doff = arith.addi %mrwide, %ncwide : tensor<16x64xi32>
   %dbase = tt.splat %dst : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
   %dptr = tt.addptr %dbase, %doff : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
   tt.store %dptr, %d : tensor<16x64x!tt.ptr<f32>>
   tt.return
   }
}
