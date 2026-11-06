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
// Variant: varlen (USE_OFFSETS=1) — extra scalar loads of token_indices/offsets.
// Negative case: RFC puts varlen out of scope for the scan rewrite.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @parallel_nsa_fwd_kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>, %arg4: !tt.ptr<f32>, %arg5: f32, %arg6: !tt.ptr<i64>, %arg7: !tt.ptr<i64>, %arg8: !tt.ptr<i64>, %arg9: !tt.ptr<i64>, %arg10: i32) attributes {noinline = false} {
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
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.remsi %2, %c4_i32 : i32
    %4 = arith.muli %0, %c2_i32 : i32
    %5 = tt.addptr %arg9, %4 : !tt.ptr<i64>, i32
    %6 = tt.load %5 : !tt.ptr<i64>
    %7 = arith.trunci %6 : i64 to i32
    %8 = tt.addptr %5, %c1_i32 : !tt.ptr<i64>, i32
    %9 = tt.load %8 : !tt.ptr<i64>
    %10 = arith.trunci %9 : i64 to i32
    %11 = tt.addptr %arg8, %7 : !tt.ptr<i64>, i32
    %12 = tt.load %11 : !tt.ptr<i64>
    %13 = arith.trunci %12 : i64 to i32
    %14 = tt.addptr %11, %c1_i32 : !tt.ptr<i64>, i32
    %15 = tt.load %14 : !tt.ptr<i64>
    %16 = arith.trunci %15 : i64 to i32
    %17 = arith.subi %16, %13 : i32
    %18 = arith.muli %13, %c4_i32 : i32
    %19 = arith.addi %18, %3 : i32
    %20 = arith.muli %19, %c128_i32 : i32
    %21 = tt.addptr %arg1, %20 : !tt.ptr<bf16>, i32
    %22 = tt.addptr %arg2, %20 : !tt.ptr<bf16>, i32
    %23 = arith.addi %13, %10 : i32
    %24 = arith.muli %23, %c64_i32 : i32
    %25 = arith.muli %3, %c16_i32 : i32
    %26 = arith.addi %24, %25 : i32
    %27 = tt.addptr %arg6, %26 : !tt.ptr<i64>, i32
    %28 = arith.muli %23, %c4_i32 : i32
    %29 = tt.addptr %arg7, %28 : !tt.ptr<i64>, i32
    %30 = tt.addptr %29, %3 : !tt.ptr<i64>, i32
    %31 = tt.load %30 : !tt.ptr<i64>
    %32 = arith.muli %23, %c8192_i32 : i32
    %33 = tt.addptr %arg0, %32 : !tt.ptr<bf16>, i32
    %34 = tt.make_tensor_ptr %33, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%25, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x128xbf16>>
    %35 = tt.load %34 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x128xbf16>>
    %36 = arith.extf %35 : tensor<16x128xbf16> to tensor<16x128xf32>
    %37 = tt.splat %arg5 : f32 -> tensor<16x128xf32>
    %38 = arith.mulf %36, %37 : tensor<16x128xf32>
    %39 = arith.truncf %38 : tensor<16x128xf32> to tensor<16x128xbf16>
    %40 = tt.addptr %arg3, %32 : !tt.ptr<bf16>, i32
    %41 = arith.muli %1, %c128_i32 : i32
    %42 = tt.make_tensor_ptr %40, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%25, %41] {order = array<i32: 1, 0>} : <tensor<16x128xbf16>>
    %43 = tt.addptr %arg4, %24 : !tt.ptr<f32>, i32
    %44 = tt.addptr %43, %25 : !tt.ptr<f32>, i32
    %45 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %46 = tt.splat %44 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %47 = tt.addptr %46, %45 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %48:3 = scf.for %arg11 = %c0_i64 to %31 step %c1_i64 iter_args(%arg12 = %cst_0, %arg13 = %cst_3, %arg14 = %cst) -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)  : i64 {
      %55 = tt.addptr %27, %arg11 : !tt.ptr<i64>, i64
      %56 = tt.load %55 : !tt.ptr<i64>
      %57 = arith.trunci %56 : i64 to i32
      %58 = arith.muli %57, %c64_i32 : i32
      %59 = arith.cmpi sle, %58, %10 : i32
      %60 = arith.cmpi sge, %58, %c0_i32 : i32
      %61 = arith.andi %59, %60 : i1
      %62:3 = scf.if %61 -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>) {
        %63 = arith.extsi %17 : i32 to i64
        %64 = tt.make_tensor_ptr %21, [%c128_i64, %63], [%c1_i64, %c512_i64], [%c0_i32, %58] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
        %65 = tt.make_tensor_ptr %22, [%63, %c128_i64], [%c512_i64, %c1_i64], [%58, %41] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
        %66 = tt.load %64 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
        %67 = tt.load %65 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
        %68 = tt.dot %39, %66, %cst_2 : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
        %69 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %70 = tt.splat %58 : i32 -> tensor<64xi32>
        %71 = arith.addi %70, %69 : tensor<64xi32>
        %72 = tt.splat %10 : i32 -> tensor<64xi32>
        %73 = arith.cmpi sge, %72, %71 : tensor<64xi32>
        %74 = tt.expand_dims %73 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
        %75 = tt.broadcast %74 : tensor<1x64xi1> -> tensor<16x64xi1>
        %76 = arith.select %75, %68, %cst_1 : tensor<16x64xi1>, tensor<16x64xf32>
        %77 = "tt.reduce"(%76) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %93 = arith.maxnumf %arg15, %arg16 : f32
          tt.reduce.return %93 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %78 = arith.maxnumf %arg13, %77 : tensor<16xf32>
        %79 = arith.subf %arg13, %78 : tensor<16xf32>
        %80 = math.exp %79 : tensor<16xf32>
        %81 = tt.expand_dims %78 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %82 = tt.broadcast %81 : tensor<16x1xf32> -> tensor<16x64xf32>
        %83 = arith.subf %76, %82 : tensor<16x64xf32>
        %84 = math.exp %83 : tensor<16x64xf32>
        %85 = arith.mulf %arg14, %80 : tensor<16xf32>
        %86 = "tt.reduce"(%84) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %93 = arith.addf %arg15, %arg16 : f32
          tt.reduce.return %93 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %87 = arith.addf %85, %86 : tensor<16xf32>
        %88 = tt.expand_dims %80 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %89 = tt.broadcast %88 : tensor<16x1xf32> -> tensor<16x128xf32>
        %90 = arith.mulf %arg12, %89 : tensor<16x128xf32>
        %91 = arith.truncf %84 : tensor<16x64xf32> to tensor<16x64xbf16>
        %92 = tt.dot %91, %67, %90 : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>
        scf.yield %92, %78, %87 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
      } else {
        scf.yield %arg12, %arg13, %arg14 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
      }
      scf.yield %62#0, %62#1, %62#2 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
    }
    %49 = tt.expand_dims %48#2 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %50 = tt.broadcast %49 : tensor<16x1xf32> -> tensor<16x128xf32>
    %51 = arith.divf %48#0, %50 : tensor<16x128xf32>
    %52 = math.log %48#2 : tensor<16xf32>
    %53 = arith.addf %48#1, %52 : tensor<16xf32>
    %54 = arith.truncf %51 : tensor<16x128xf32> to tensor<16x128xbf16>
    tt.store %42, %54 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x128xbf16>>
    tt.store %47, %53 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
