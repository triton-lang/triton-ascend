// === #2686 SelectedBlockScanRule (bit 65536) — REAL H=1 fixture ===
// Provenance: nsa_fwd.py -> ast_to_ttir -> make_ttir() (the TTIR graph-optimize
// sees).  This copy is that IR modulo all loc() dropped (re-printed by
// triton-opt); the module target stays the Ascend950PR_9579 spelling.
// Shapes: H=1 HQ=16 G=16 K=V=64 S=16 BS=32 BK=BV=64, dtype bf16.
// Variant: H == 1 (so i_h = i_bh % H folds away), dyn NS
// (NS = tt.load(block_counts)).  The row base then has NO i_h*S addend and the
// trip count is not a constant, so neither of the two old anchors could give the
// table length S: the matcher only fell back to "S == constant trip count".
// The row stride of block_indices is H*S == S when H == 1, and H itself is
// visible in the layout (H > 1 always leaves an `i_bh % H` remainder behind;
// this fixture has none), so S is read off the multiplier of the pid-carrying
// row offset.  That is what the HEAD1 anchor in the lit test asserts.
// POSITIVE case for H == 1 + dyn NS.
// NOT a lit test: lit.cfg.py excludes Inputs/ (unittest/lit.cfg.py:38).
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @parallel_nsa_fwd_kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>, %arg4: !tt.ptr<f32>, %arg5: f32, %arg6: !tt.ptr<i64>, %arg7: !tt.ptr<i32>, %arg8: i32, %arg9: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c1_i32 = arith.constant 1 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<16x32xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x32xf32>
    %c32_i32 = arith.constant 32 : i32
    %cst_3 = arith.constant dense<0xFF800000> : tensor<16xf32>
    %c0_i32 = arith.constant 0 : i32
    %c16_i64 = arith.constant 16 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %2, %arg8 : i32
    %4 = arith.extsi %3 : i32 to i64
    %5 = arith.muli %2, %arg9 : i32
    %6 = arith.extsi %5 : i32 to i64
    %7 = arith.subi %arg9, %arg8 : i32
    %8 = arith.muli %6, %c64_i64 : i64
    %9 = tt.addptr %arg1, %8 : !tt.ptr<bf16>, i64
    %10 = tt.addptr %arg2, %8 : !tt.ptr<bf16>, i64
    %11 = arith.extsi %0 : i32 to i64
    %12 = arith.addi %4, %11 : i64
    %13 = arith.muli %12, %c16_i64 : i64
    %14 = tt.addptr %arg6, %13 : !tt.ptr<i64>, i64
    %15 = tt.addptr %arg7, %12 : !tt.ptr<i32>, i64
    %16 = tt.load %15 : !tt.ptr<i32>
    %17 = arith.muli %12, %c1024_i64 : i64
    %18 = tt.addptr %arg0, %17 : !tt.ptr<bf16>, i64
    %19 = tt.make_tensor_ptr %18, [%c16_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xbf16>>
    %20 = tt.load %19 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x64xbf16>>
    %21 = arith.extf %20 : tensor<16x64xbf16> to tensor<16x64xf32>
    %22 = tt.splat %arg5 : f32 -> tensor<16x64xf32>
    %23 = arith.mulf %21, %22 : tensor<16x64xf32>
    %24 = arith.truncf %23 : tensor<16x64xf32> to tensor<16x64xbf16>
    %25 = tt.addptr %arg3, %17 : !tt.ptr<bf16>, i64
    %26 = arith.muli %1, %c64_i32 : i32
    %27 = tt.make_tensor_ptr %25, [%c16_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %26] {order = array<i32: 1, 0>} : <tensor<16x64xbf16>>
    %28 = tt.addptr %arg4, %13 : !tt.ptr<f32>, i64
    %29 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %30 = tt.splat %28 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %31 = tt.addptr %30, %29 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %32 = arith.addi %7, %0 : i32
    %33:3 = scf.for %arg10 = %c0_i32 to %16 step %c1_i32 iter_args(%arg11 = %cst_0, %arg12 = %cst_3, %arg13 = %cst) -> (tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>)  : i32 {
      %41 = tt.addptr %14, %arg10 : !tt.ptr<i64>, i32
      %42 = tt.load %41 : !tt.ptr<i64>
      %43 = arith.trunci %42 : i64 to i32
      %44 = arith.muli %43, %c32_i32 : i32
      %45 = arith.cmpi sle, %44, %32 : i32
      %46 = arith.cmpi sge, %44, %c0_i32 : i32
      %47 = arith.andi %45, %46 : i1
      %48:3 = scf.if %47 -> (tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>) {
        %49 = arith.extsi %arg9 : i32 to i64
        %50 = tt.make_tensor_ptr %9, [%c64_i64, %49], [%c1_i64, %c64_i64], [%c0_i32, %44] {order = array<i32: 0, 1>} : <tensor<64x32xbf16>>
        %51 = tt.make_tensor_ptr %10, [%49, %c64_i64], [%c64_i64, %c1_i64], [%44, %26] {order = array<i32: 1, 0>} : <tensor<32x64xbf16>>
        %52 = tt.load %50 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
        %53 = tt.load %51 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xbf16>>
        %54 = tt.dot %24, %52, %cst_2 : tensor<16x64xbf16> * tensor<64x32xbf16> -> tensor<16x32xf32>
        %55 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
        %56 = tt.splat %44 : i32 -> tensor<32xi32>
        %57 = arith.addi %56, %55 : tensor<32xi32>
        %58 = tt.splat %32 : i32 -> tensor<32xi32>
        %59 = arith.cmpi sge, %58, %57 : tensor<32xi32>
        %60 = tt.expand_dims %59 {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
        %61 = tt.broadcast %60 : tensor<1x32xi1> -> tensor<16x32xi1>
        %62 = arith.select %61, %54, %cst_1 : tensor<16x32xi1>, tensor<16x32xf32>
        %63 = "tt.reduce"(%62) <{axis = 1 : i32}> ({
        ^bb0(%arg14: f32, %arg15: f32):
          %79 = arith.maxnumf %arg14, %arg15 : f32
          tt.reduce.return %79 : f32
        }) : (tensor<16x32xf32>) -> tensor<16xf32>
        %64 = arith.maxnumf %arg12, %63 : tensor<16xf32>
        %65 = arith.subf %arg12, %64 : tensor<16xf32>
        %66 = math.exp %65 : tensor<16xf32>
        %67 = tt.expand_dims %64 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %68 = tt.broadcast %67 : tensor<16x1xf32> -> tensor<16x32xf32>
        %69 = arith.subf %62, %68 : tensor<16x32xf32>
        %70 = math.exp %69 : tensor<16x32xf32>
        %71 = arith.mulf %arg13, %66 : tensor<16xf32>
        %72 = "tt.reduce"(%70) <{axis = 1 : i32}> ({
        ^bb0(%arg14: f32, %arg15: f32):
          %79 = arith.addf %arg14, %arg15 : f32
          tt.reduce.return %79 : f32
        }) : (tensor<16x32xf32>) -> tensor<16xf32>
        %73 = arith.addf %71, %72 : tensor<16xf32>
        %74 = tt.expand_dims %66 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %75 = tt.broadcast %74 : tensor<16x1xf32> -> tensor<16x64xf32>
        %76 = arith.mulf %arg11, %75 : tensor<16x64xf32>
        %77 = arith.truncf %70 : tensor<16x32xf32> to tensor<16x32xbf16>
        %78 = tt.dot %77, %53, %76 : tensor<16x32xbf16> * tensor<32x64xbf16> -> tensor<16x64xf32>
        scf.yield %78, %64, %73 : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
      } else {
        scf.yield %arg11, %arg12, %arg13 : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
      }
      scf.yield %48#0, %48#1, %48#2 : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
    }
    %34 = tt.expand_dims %33#2 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %35 = tt.broadcast %34 : tensor<16x1xf32> -> tensor<16x64xf32>
    %36 = arith.divf %33#0, %35 : tensor<16x64xf32>
    %37 = math.log %33#2 : tensor<16xf32>
    %38 = arith.addf %33#1, %37 : tensor<16xf32>
    %39 = arith.truncf %36 : tensor<16x64xf32> to tensor<16x64xbf16>
    tt.store %27, %39 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x64xbf16>>
    %40 = arith.cmpi eq, %1, %c0_i32 : i32
    scf.if %40 {
      tt.store %31, %38 : tensor<16x!tt.ptr<f32>>
    }
    tt.return
  }
}
