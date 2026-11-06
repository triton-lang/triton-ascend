// === #2686 SelectedBlockScanRule (bit 65536) — design target ===
//
// What this is: the compiled gold standard for the rewritten form -- the
// Rule-A-only reference implementation of the NSA forward kernel, run through the
// same early TTIR pipeline that graph-optimize sees.  It is the artifact the
// rewrite is checked against, so the FileCheck assertions in
// nsa_selected_block_scan.mlir (prefix REWRITTEN / NOSCALAR) can be validated
// without running the scan rewrite at all.
// NOT a lit test: lit.cfg.py excludes Inputs/ (unittest/lit.cfg.py:38).
// Provenance of this copy: re-printed once by `triton-opt <file> -o <new>` WITHOUT
// --mlir-print-debuginfo.  It is the SAME IR, LOCATIONS DROPPED -- not a
// byte-for-byte copy of the reference artifact: every #loc alias is gone, and
// because that artifact carried name locations, the SSA value names they supplied
// are gone too (%q, %k_5, ... now print as %arg0, %6, ...).  The op stream, the
// types and the module attribute are unchanged.  The re-print is what keeps
// development-host absolute paths (and the reference artifact's own paths) out of
// the tree.
// Shapes: H=4 HQ=64 G=16 K=V=128 S=16 BS=64 BK=BV=128, dtype bf16; constant trip
// count (NS = S = 16); online-softmax with iter_args o/m/l; the index table read
// once as a vector, cdiv-based scan bound, per-token weight count + safe_m, and
// both dots keeping M = 16.
module attributes {hacc.target = #hacc.target<"Ascend910_95">} {
  tt.func public @parallel_nsa_fwd_kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>, %arg4: !tt.ptr<f32>, %arg5: f32, %arg6: !tt.ptr<i64>, %arg7: !tt.ptr<i64>, %arg8: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x128xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %cst_2 = arith.constant dense<0xFF800000> : tensor<16x64xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %c512_i64 = arith.constant 512 : i64
    %c1_i32 = arith.constant 1 : i32
    %cst_4 = arith.constant dense<16> : tensor<16xi32>
    %cst_5 = arith.constant dense<64> : tensor<16xi32>
    %cst_6 = arith.constant dense<0xFF800000> : tensor<16xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.divsi %2, %c4_i32 : i32
    %4 = arith.remsi %2, %c4_i32 : i32
    %5 = arith.muli %3, %arg8 : i32
    %6 = arith.muli %5, %c4_i32 : i32
    %7 = arith.addi %6, %4 : i32
    %8 = arith.muli %7, %c128_i32 : i32
    %9 = tt.addptr %arg1, %8 : !tt.ptr<bf16>, i32
    %10 = tt.addptr %arg2, %8 : !tt.ptr<bf16>, i32
    %11 = arith.addi %5, %0 : i32
    %12 = arith.muli %11, %c64_i32 : i32
    %13 = arith.muli %4, %c16_i32 : i32
    %14 = arith.addi %12, %13 : i32
    %15 = tt.addptr %arg6, %14 : !tt.ptr<i64>, i32
    %16 = arith.muli %11, %c8192_i32 : i32
    %17 = tt.addptr %arg0, %16 : !tt.ptr<bf16>, i32
    %18 = tt.make_tensor_ptr %17, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%13, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x128xbf16>>
    %19 = tt.load %18 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x128xbf16>>
    %20 = arith.extf %19 : tensor<16x128xbf16> to tensor<16x128xf32>
    %21 = tt.splat %arg5 : f32 -> tensor<16x128xf32>
    %22 = arith.mulf %20, %21 : tensor<16x128xf32>
    %23 = arith.truncf %22 : tensor<16x128xf32> to tensor<16x128xbf16>
    %24 = tt.addptr %arg3, %16 : !tt.ptr<bf16>, i32
    %25 = arith.muli %1, %c128_i32 : i32
    %26 = tt.make_tensor_ptr %24, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%13, %25] {order = array<i32: 1, 0>} : <tensor<16x128xbf16>>
    %27 = tt.addptr %arg4, %12 : !tt.ptr<f32>, i32
    %28 = tt.addptr %27, %13 : !tt.ptr<f32>, i32
    %29 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %30 = tt.splat %28 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %31 = tt.addptr %30, %29 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %32 = tt.splat %15 : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>>
    %33 = tt.addptr %32, %29 : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    %34 = tt.load %33 : tensor<16x!tt.ptr<i64>>
    %35 = arith.trunci %34 : tensor<16xi64> to tensor<16xi32>
    %36 = arith.muli %35, %cst_5 : tensor<16xi32>
    %37 = arith.cmpi slt, %29, %cst_4 : tensor<16xi32>
    %38 = arith.addi %0, %c64_i32 : i32
    %39 = arith.divsi %38, %c64_i32 : i32
    %40 = arith.extsi %arg8 : i32 to i64
    %41 = tt.expand_dims %36 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %42 = tt.expand_dims %37 {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %43 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %44 = tt.splat %0 : i32 -> tensor<64xi32>
    %45:3 = scf.for %arg9 = %c0_i32 to %39 step %c1_i32 iter_args(%arg10 = %cst_0, %arg11 = %cst_6, %arg12 = %cst_1) -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)  : i32 {
      %52 = arith.muli %arg9, %c64_i32 : i32
      %53 = tt.make_tensor_ptr %9, [%c128_i64, %40], [%c1_i64, %c512_i64], [%c0_i32, %52] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
      %54 = tt.make_tensor_ptr %10, [%40, %c128_i64], [%c512_i64, %c1_i64], [%52, %25] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %55 = tt.load %53 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
      %56 = tt.load %54 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
      %57 = tt.splat %52 : i32 -> tensor<1x16xi32>
      %58 = arith.cmpi eq, %41, %57 : tensor<1x16xi32>
      %59 = arith.andi %58, %42 : tensor<1x16xi1>
      %60 = arith.extui %59 : tensor<1x16xi1> to tensor<1x16xi32>
      %61 = "tt.reduce"(%60) <{axis = 1 : i32}> ({
      ^bb0(%arg13: i32, %arg14: i32):
        %96 = arith.addi %arg13, %arg14 : i32
        tt.reduce.return %96 : i32
      }) : (tensor<1x16xi32>) -> tensor<1xi32>
      %62 = arith.sitofp %61 : tensor<1xi32> to tensor<1xf32>
      %63 = tt.expand_dims %62 {axis = 1 : i32} : tensor<1xf32> -> tensor<1x1xf32>
      %64 = tt.broadcast %63 : tensor<1x1xf32> -> tensor<1x64xf32>
      %65 = tt.reshape %64 : tensor<1x64xf32> -> tensor<64xf32>
      %66 = tt.splat %52 : i32 -> tensor<64xi32>
      %67 = arith.addi %66, %43 : tensor<64xi32>
      %68 = arith.cmpf ogt, %65, %cst : tensor<64xf32>
      %69 = arith.cmpi sle, %67, %44 : tensor<64xi32>
      %70 = arith.andi %68, %69 : tensor<64xi1>
      %71 = tt.dot %23, %55, %cst_3 : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
      %72 = tt.expand_dims %70 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
      %73 = tt.broadcast %72 : tensor<1x64xi1> -> tensor<16x64xi1>
      %74 = arith.select %73, %71, %cst_2 : tensor<16x64xi1>, tensor<16x64xf32>
      %75 = "tt.reduce"(%74) <{axis = 1 : i32}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %96 = arith.maxnumf %arg13, %arg14 : f32
        tt.reduce.return %96 : f32
      }) : (tensor<16x64xf32>) -> tensor<16xf32>
      %76 = arith.maxnumf %arg11, %75 : tensor<16xf32>
      %77 = arith.cmpf oeq, %76, %cst_6 : tensor<16xf32>
      %78 = arith.select %77, %cst_1, %76 : tensor<16xi1>, tensor<16xf32>
      %79 = arith.subf %arg11, %78 : tensor<16xf32>
      %80 = math.exp %79 : tensor<16xf32>
      %81 = tt.expand_dims %78 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
      %82 = tt.broadcast %81 : tensor<16x1xf32> -> tensor<16x64xf32>
      %83 = arith.subf %74, %82 : tensor<16x64xf32>
      %84 = math.exp %83 : tensor<16x64xf32>
      %85 = tt.expand_dims %65 {axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
      %86 = tt.broadcast %85 : tensor<1x64xf32> -> tensor<16x64xf32>
      %87 = arith.mulf %84, %86 : tensor<16x64xf32>
      %88 = arith.mulf %arg12, %80 : tensor<16xf32>
      %89 = "tt.reduce"(%87) <{axis = 1 : i32}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %96 = arith.addf %arg13, %arg14 : f32
        tt.reduce.return %96 : f32
      }) : (tensor<16x64xf32>) -> tensor<16xf32>
      %90 = arith.addf %88, %89 : tensor<16xf32>
      %91 = tt.expand_dims %80 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
      %92 = tt.broadcast %91 : tensor<16x1xf32> -> tensor<16x128xf32>
      %93 = arith.mulf %arg10, %92 : tensor<16x128xf32>
      %94 = arith.truncf %87 : tensor<16x64xf32> to tensor<16x64xbf16>
      %95 = tt.dot %94, %56, %93 : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>
      scf.yield %95, %76, %90 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
    }
    %46 = tt.expand_dims %45#2 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %47 = tt.broadcast %46 : tensor<16x1xf32> -> tensor<16x128xf32>
    %48 = arith.divf %45#0, %47 : tensor<16x128xf32>
    %49 = math.log %45#2 : tensor<16xf32>
    %50 = arith.addf %45#1, %49 : tensor<16xf32>
    %51 = arith.truncf %48 : tensor<16x128xf32> to tensor<16x128xbf16>
    tt.store %26, %51 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x128xbf16>>
    tt.store %31, %50 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
