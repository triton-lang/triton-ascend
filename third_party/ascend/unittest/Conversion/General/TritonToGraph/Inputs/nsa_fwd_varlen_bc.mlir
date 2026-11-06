// === #2686 SelectedBlockScanRule (bit 65536) — REAL varlen fixture ===
// Provenance: nsa_fwd.py -> ast_to_ttir -> make_ttir() (the TTIR graph-optimize
// sees).  This copy is that IR modulo all loc() dropped (re-printed by
// triton-opt); the module target stays the Ascend950PR_9579 spelling.
// Shapes: H=4 HQ=64 G=16 K=V=128 S=16 BS=64 BK=BV=128, dtype bf16.
// Variant: varlen (IS_VARLEN=1) AND block_counts present (USE_BLOCK_COUNTS=1),
// i.e. the production varlen dispatch: 2 token_indices_q + 2 cu_seqlens_q +
// 2 cu_seqlens_k + 1 block_counts = SEVEN loop-invariant scalar loads before the
// loop.  The skeleton budget used to be six (the varlen prefix only), so this
// shape was rejected before parseIndexTable ever ran.
// POSITIVE case for the scan rewrite's matcher.  The JOINT A+B gate still declines it
// because the row lift does not lift varlen (see the VARLEN group and negative (a)).
// NOT a lit test: lit.cfg.py excludes Inputs/ (unittest/lit.cfg.py:38).
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @parallel_nsa_fwd_kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>, %arg4: !tt.ptr<f32>, %arg5: f32, %arg6: !tt.ptr<i64>, %arg7: !tt.ptr<i32>, %arg8: !tt.ptr<i32>, %arg9: !tt.ptr<i32>, %arg10: !tt.ptr<i32>, %arg11: i32, %arg12: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c8192_i64 = arith.constant 8192 : i64
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<16x64xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %c512_i64 = arith.constant 512 : i64
    %cst_3 = arith.constant dense<0xFF800000> : tensor<16xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c4_i64 = arith.constant 4 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.remsi %2, %c4_i32 : i32
    %4 = arith.muli %0, %c2_i32 : i32
    %5 = tt.addptr %arg10, %4 : !tt.ptr<i32>, i32
    %6 = tt.load %5 : !tt.ptr<i32>
    %7 = tt.addptr %5, %c1_i32 : !tt.ptr<i32>, i32
    %8 = tt.load %7 : !tt.ptr<i32>
    %9 = tt.addptr %arg8, %6 : !tt.ptr<i32>, i32
    %10 = tt.load %9 : !tt.ptr<i32>
    %11 = arith.extsi %10 : i32 to i64
    %12 = tt.addptr %9, %c1_i32 : !tt.ptr<i32>, i32
    %13 = tt.load %12 : !tt.ptr<i32>
    %14 = arith.extsi %13 : i32 to i64
    %15 = tt.addptr %arg9, %6 : !tt.ptr<i32>, i32
    %16 = tt.load %15 : !tt.ptr<i32>
    %17 = arith.extsi %16 : i32 to i64
    %18 = tt.addptr %15, %c1_i32 : !tt.ptr<i32>, i32
    %19 = tt.load %18 : !tt.ptr<i32>
    %20 = arith.extsi %19 : i32 to i64
    %21 = arith.subi %14, %11 : i64
    %22 = arith.trunci %21 : i64 to i32
    %23 = arith.subi %20, %17 : i64
    %24 = arith.trunci %23 : i64 to i32
    %25 = arith.subi %24, %22 : i32
    %26 = arith.muli %17, %c4_i64 : i64
    %27 = arith.extsi %3 : i32 to i64
    %28 = arith.addi %26, %27 : i64
    %29 = arith.muli %28, %c128_i64 : i64
    %30 = tt.addptr %arg1, %29 : !tt.ptr<bf16>, i64
    %31 = tt.addptr %arg2, %29 : !tt.ptr<bf16>, i64
    %32 = arith.extsi %8 : i32 to i64
    %33 = arith.addi %11, %32 : i64
    %34 = arith.muli %33, %c64_i64 : i64
    %35 = arith.muli %3, %c16_i32 : i32
    %36 = arith.extsi %35 : i32 to i64
    %37 = arith.addi %34, %36 : i64
    %38 = tt.addptr %arg6, %37 : !tt.ptr<i64>, i64
    %39 = arith.muli %33, %c4_i64 : i64
    %40 = tt.addptr %arg7, %39 : !tt.ptr<i32>, i64
    %41 = tt.addptr %40, %3 : !tt.ptr<i32>, i32
    %42 = tt.load %41 : !tt.ptr<i32>
    %43 = arith.muli %33, %c8192_i64 : i64
    %44 = tt.addptr %arg0, %43 : !tt.ptr<bf16>, i64
    %45 = tt.make_tensor_ptr %44, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%35, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x128xbf16>>
    %46 = tt.load %45 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x128xbf16>>
    %47 = arith.extf %46 : tensor<16x128xbf16> to tensor<16x128xf32>
    %48 = tt.splat %arg5 : f32 -> tensor<16x128xf32>
    %49 = arith.mulf %47, %48 : tensor<16x128xf32>
    %50 = arith.truncf %49 : tensor<16x128xf32> to tensor<16x128xbf16>
    %51 = tt.addptr %arg3, %43 : !tt.ptr<bf16>, i64
    %52 = arith.muli %1, %c128_i32 : i32
    %53 = tt.make_tensor_ptr %51, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%35, %52] {order = array<i32: 1, 0>} : <tensor<16x128xbf16>>
    %54 = tt.addptr %arg4, %34 : !tt.ptr<f32>, i64
    %55 = tt.addptr %54, %35 : !tt.ptr<f32>, i32
    %56 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %57 = tt.splat %55 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %58 = tt.addptr %57, %56 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %59 = arith.addi %25, %8 : i32
    %60:3 = scf.for %arg13 = %c0_i32 to %42 step %c1_i32 iter_args(%arg14 = %cst_0, %arg15 = %cst_3, %arg16 = %cst) -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)  : i32 {
      %68 = tt.addptr %38, %arg13 : !tt.ptr<i64>, i32
      %69 = tt.load %68 : !tt.ptr<i64>
      %70 = arith.trunci %69 : i64 to i32
      %71 = arith.muli %70, %c64_i32 : i32
      %72 = arith.cmpi sle, %71, %59 : i32
      %73 = arith.cmpi sge, %71, %c0_i32 : i32
      %74 = arith.andi %72, %73 : i1
      %75:3 = scf.if %74 -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>) {
        %76 = arith.extsi %24 : i32 to i64
        %77 = tt.make_tensor_ptr %30, [%c128_i64, %76], [%c1_i64, %c512_i64], [%c0_i32, %71] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
        %78 = tt.make_tensor_ptr %31, [%76, %c128_i64], [%c512_i64, %c1_i64], [%71, %52] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
        %79 = tt.load %77 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
        %80 = tt.load %78 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
        %81 = tt.dot %50, %79, %cst_2 : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
        %82 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %83 = tt.splat %71 : i32 -> tensor<64xi32>
        %84 = arith.addi %83, %82 : tensor<64xi32>
        %85 = tt.splat %59 : i32 -> tensor<64xi32>
        %86 = arith.cmpi sge, %85, %84 : tensor<64xi32>
        %87 = tt.expand_dims %86 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
        %88 = tt.broadcast %87 : tensor<1x64xi1> -> tensor<16x64xi1>
        %89 = arith.select %88, %81, %cst_1 : tensor<16x64xi1>, tensor<16x64xf32>
        %90 = "tt.reduce"(%89) <{axis = 1 : i32}> ({
        ^bb0(%arg17: f32, %arg18: f32):
          %106 = arith.maxnumf %arg17, %arg18 : f32
          tt.reduce.return %106 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %91 = arith.maxnumf %arg15, %90 : tensor<16xf32>
        %92 = arith.subf %arg15, %91 : tensor<16xf32>
        %93 = math.exp %92 : tensor<16xf32>
        %94 = tt.expand_dims %91 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %95 = tt.broadcast %94 : tensor<16x1xf32> -> tensor<16x64xf32>
        %96 = arith.subf %89, %95 : tensor<16x64xf32>
        %97 = math.exp %96 : tensor<16x64xf32>
        %98 = arith.mulf %arg16, %93 : tensor<16xf32>
        %99 = "tt.reduce"(%97) <{axis = 1 : i32}> ({
        ^bb0(%arg17: f32, %arg18: f32):
          %106 = arith.addf %arg17, %arg18 : f32
          tt.reduce.return %106 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %100 = arith.addf %98, %99 : tensor<16xf32>
        %101 = tt.expand_dims %93 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %102 = tt.broadcast %101 : tensor<16x1xf32> -> tensor<16x128xf32>
        %103 = arith.mulf %arg14, %102 : tensor<16x128xf32>
        %104 = arith.truncf %97 : tensor<16x64xf32> to tensor<16x64xbf16>
        %105 = tt.dot %104, %80, %103 : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>
        scf.yield %105, %91, %100 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
      } else {
        scf.yield %arg14, %arg15, %arg16 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
      }
      scf.yield %75#0, %75#1, %75#2 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
    }
    %61 = tt.expand_dims %60#2 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %62 = tt.broadcast %61 : tensor<16x1xf32> -> tensor<16x128xf32>
    %63 = arith.divf %60#0, %62 : tensor<16x128xf32>
    %64 = math.log %60#2 : tensor<16xf32>
    %65 = arith.addf %60#1, %64 : tensor<16xf32>
    %66 = arith.truncf %63 : tensor<16x128xf32> to tensor<16x128xbf16>
    tt.store %53, %66 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x128xbf16>>
    %67 = arith.cmpi eq, %1, %c0_i32 : i32
    scf.if %67 {
      tt.store %58, %65 : tensor<16x!tt.ptr<f32>>
    }
    tt.return
  }
}
