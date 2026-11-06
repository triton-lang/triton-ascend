// === #2686 SelectedBlockScanRule (bit 65536) — REAL kernel fixture ===
// Provenance: nsa_fwd.py -> ast_to_ttir -> make_ttir() (the TTIR graph-optimize
// sees: inliner/rewrite_tensor_descriptor/combine/canonicalize/reorder_broadcast/
// cse/licm/symbol_dce/loop_unroll already applied).  This copy is the same IR
// modulo (a) all loc() dropped (re-printed by triton-opt, which does not print
// debug locs) and (b) the module target attribute switched to the Ascend950PR
// spelling that hacc::utils::isAscend950() accepts (the frontend writes
// "Ascend910_95" on the dev host, which that helper rejects).
// Shapes: H=4 HQ=64 G=16 K=V=128 S=16 BS=64 BK=BV=128, dtype bf16.
// NOT a lit test: lit.cfg.py excludes Inputs/ (unittest/lit.cfg.py:38).
// Variant: NS = tt.load(block_counts + ...) -> runtime trip count, i64 induction var.
// D3 settled (2026-10-31): ACCEPTED — a non-constant upper bound that comes
// from a scalar tt.load(block_counts) is rewritten, so this is a POSITIVE case
// (group D of nsa_selected_block_scan.mlir, check prefix DYNREW; the surviving
// block_counts load before the table vector read rules out the plain REWRITTEN
// prefix — see the FileCheck note in that file).
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @parallel_nsa_fwd_kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>, %arg4: !tt.ptr<f32>, %arg5: f32, %arg6: !tt.ptr<i64>, %arg7: !tt.ptr<i64>, %arg8: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c0_i64 = arith.constant 0 : i64
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<16x64xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %c512_i64 = arith.constant 512 : i64
    %cst_3 = arith.constant dense<0xFF800000> : tensor<16xf32>
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
    %16 = arith.muli %11, %c4_i32 : i32
    %17 = tt.addptr %arg7, %16 : !tt.ptr<i64>, i32
    %18 = tt.addptr %17, %4 : !tt.ptr<i64>, i32
    %19 = tt.load %18 : !tt.ptr<i64>
    %20 = arith.muli %11, %c8192_i32 : i32
    %21 = tt.addptr %arg0, %20 : !tt.ptr<bf16>, i32
    %22 = tt.make_tensor_ptr %21, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%13, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x128xbf16>>
    %23 = tt.load %22 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x128xbf16>>
    %24 = arith.extf %23 : tensor<16x128xbf16> to tensor<16x128xf32>
    %25 = tt.splat %arg5 : f32 -> tensor<16x128xf32>
    %26 = arith.mulf %24, %25 : tensor<16x128xf32>
    %27 = arith.truncf %26 : tensor<16x128xf32> to tensor<16x128xbf16>
    %28 = tt.addptr %arg3, %20 : !tt.ptr<bf16>, i32
    %29 = arith.muli %1, %c128_i32 : i32
    %30 = tt.make_tensor_ptr %28, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%13, %29] {order = array<i32: 1, 0>} : <tensor<16x128xbf16>>
    %31 = tt.addptr %arg4, %12 : !tt.ptr<f32>, i32
    %32 = tt.addptr %31, %13 : !tt.ptr<f32>, i32
    %33 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %34 = tt.splat %32 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %35 = tt.addptr %34, %33 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %36:3 = scf.for %arg9 = %c0_i64 to %19 step %c1_i64 iter_args(%arg10 = %cst_0, %arg11 = %cst_3, %arg12 = %cst) -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)  : i64 {
      %43 = tt.addptr %15, %arg9 : !tt.ptr<i64>, i64
      %44 = tt.load %43 : !tt.ptr<i64>
      %45 = arith.trunci %44 : i64 to i32
      %46 = arith.muli %45, %c64_i32 : i32
      %47 = arith.cmpi sle, %46, %0 : i32
      %48 = arith.cmpi sge, %46, %c0_i32 : i32
      %49 = arith.andi %47, %48 : i1
      %50:3 = scf.if %49 -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>) {
        %51 = arith.extsi %arg8 : i32 to i64
        %52 = tt.make_tensor_ptr %9, [%c128_i64, %51], [%c1_i64, %c512_i64], [%c0_i32, %46] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
        %53 = tt.make_tensor_ptr %10, [%51, %c128_i64], [%c512_i64, %c1_i64], [%46, %29] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
        %54 = tt.load %52 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
        %55 = tt.load %53 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
        %56 = tt.dot %27, %54, %cst_2 : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
        %57 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %58 = tt.splat %46 : i32 -> tensor<64xi32>
        %59 = arith.addi %58, %57 : tensor<64xi32>
        %60 = tt.splat %0 : i32 -> tensor<64xi32>
        %61 = arith.cmpi sge, %60, %59 : tensor<64xi32>
        %62 = tt.expand_dims %61 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
        %63 = tt.broadcast %62 : tensor<1x64xi1> -> tensor<16x64xi1>
        %64 = arith.select %63, %56, %cst_1 : tensor<16x64xi1>, tensor<16x64xf32>
        %65 = "tt.reduce"(%64) <{axis = 1 : i32}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %81 = arith.maxnumf %arg13, %arg14 : f32
          tt.reduce.return %81 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %66 = arith.maxnumf %arg11, %65 : tensor<16xf32>
        %67 = arith.subf %arg11, %66 : tensor<16xf32>
        %68 = math.exp %67 : tensor<16xf32>
        %69 = tt.expand_dims %66 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %70 = tt.broadcast %69 : tensor<16x1xf32> -> tensor<16x64xf32>
        %71 = arith.subf %64, %70 : tensor<16x64xf32>
        %72 = math.exp %71 : tensor<16x64xf32>
        %73 = arith.mulf %arg12, %68 : tensor<16xf32>
        %74 = "tt.reduce"(%72) <{axis = 1 : i32}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %81 = arith.addf %arg13, %arg14 : f32
          tt.reduce.return %81 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %75 = arith.addf %73, %74 : tensor<16xf32>
        %76 = tt.expand_dims %68 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %77 = tt.broadcast %76 : tensor<16x1xf32> -> tensor<16x128xf32>
        %78 = arith.mulf %arg10, %77 : tensor<16x128xf32>
        %79 = arith.truncf %72 : tensor<16x64xf32> to tensor<16x64xbf16>
        %80 = tt.dot %79, %55, %78 : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>
        scf.yield %80, %66, %75 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
      } else {
        scf.yield %arg10, %arg11, %arg12 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
      }
      scf.yield %50#0, %50#1, %50#2 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
    }
    %37 = tt.expand_dims %36#2 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %38 = tt.broadcast %37 : tensor<16x1xf32> -> tensor<16x128xf32>
    %39 = arith.divf %36#0, %38 : tensor<16x128xf32>
    %40 = math.log %36#2 : tensor<16xf32>
    %41 = arith.addf %36#1, %40 : tensor<16xf32>
    %42 = arith.truncf %39 : tensor<16x128xf32> to tensor<16x128xbf16>
    tt.store %30, %42 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x128xbf16>>
    tt.store %35, %41 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
