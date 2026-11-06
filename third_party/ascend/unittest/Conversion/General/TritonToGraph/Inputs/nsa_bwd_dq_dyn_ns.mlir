// === #2686 SelectedBlockScanRule (bit 65536) — REAL bwd dq fixture ===
// Provenance: fla/ops/nsa/parallel.py::parallel_nsa_bwd_kernel_dq -> ast_to_ttir
// -> make_ttir() (the TTIR graph-optimize sees: inliner/rewrite_tensor_descriptor/
// combine/canonicalize/reorder_broadcast/cse/licm/symbol_dce/loop_unroll already
// applied).  This copy is that IR modulo all loc() dropped (re-printed by
// triton-opt, which does not print debug locs); the module target stays the
// Ascend950PR_9579 spelling the compile already used.
// Shapes: B=1 H=4 HQ=64 G=16 K=V=128 S=16 BS=64 BK=BV=128, dtype bf16.
// Variant: dyn NS (NS = tt.load(block_counts)), non-varlen, i64 induction
// variable, and the index address FOLDED as
//     tt.addptr %root, arith.addi(%rowOffset, %iv)
// instead of the dense spelling tt.addptr %rowBase, %iv, where %rowBase =
// tt.addptr %root, %rowOffset is hoisted out of the loop.  Both spellings are
// the same address; parseIndexTable used to insist that the offset be exactly
// the induction variable and therefore rejected this one, which left the
// block_counts production path of the bwd dq kernel silently unmatchable.
// POSITIVE case for the folded spelling.
// NOT a lit test: lit.cfg.py excludes Inputs/ (unittest/lit.cfg.py:38).
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @parallel_nsa_bwd_kernel_dq(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<bf16>, %arg6: !tt.ptr<bf16>, %arg7: f32, %arg8: !tt.ptr<i64>, %arg9: !tt.ptr<i64>, %arg10: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xbf16>
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x128xbf16>
    %cst_1 = arith.constant dense<64> : tensor<16x1xi64>
    %c8192_i64 = arith.constant 8192 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst_3 = arith.constant dense<128> : tensor<128x1xi32>
    %cst_4 = arith.constant dense<512> : tensor<1x64xi64>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<16x128xf32>
    %cst_6 = arith.constant dense<128> : tensor<16x1xi64>
    %cst_7 = arith.constant dense<128> : tensor<1x128xi32>
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i64 = arith.constant 64 : i64
    %c4_i64 = arith.constant 4 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.get_program_id y : i32
    %3 = tt.get_program_id z : i32
    %4 = arith.extsi %3 : i32 to i64
    %5 = arith.divsi %4, %c4_i64 : i64
    %6 = arith.remsi %4, %c4_i64 : i64
    %7 = arith.extsi %arg10 : i32 to i64
    %8 = arith.muli %5, %7 : i64
    %9 = arith.addi %8, %1 : i64
    %10 = arith.muli %9, %c8192_i64 : i64
    %11 = tt.addptr %arg0, %10 : !tt.ptr<bf16>, i64
    %12 = tt.addptr %arg5, %10 : !tt.ptr<bf16>, i64
    %13 = arith.muli %9, %c64_i64 : i64
    %14 = arith.extsi %2 : i32 to i64
    %15 = arith.muli %14, %7 : i64
    %16 = arith.addi %15, %8 : i64
    %17 = arith.addi %16, %1 : i64
    %18 = arith.muli %17, %c8192_i64 : i64
    %19 = tt.addptr %arg6, %18 : !tt.ptr<bf16>, i64
    %20 = arith.muli %6, %c16_i64 : i64
    %21 = arith.addi %13, %20 : i64
    %22 = arith.muli %9, %c4_i64 : i64
    %23 = arith.addi %22, %6 : i64
    %24 = tt.addptr %arg9, %23 : !tt.ptr<i64>, i64
    %25 = tt.load %24 : !tt.ptr<i64>
    %26 = arith.muli %8, %c4_i64 : i64
    %27 = arith.addi %26, %6 : i64
    %28 = arith.muli %27, %c128_i64 : i64
    %29 = tt.addptr %arg1, %28 : !tt.ptr<bf16>, i64
    %30 = tt.addptr %arg2, %28 : !tt.ptr<bf16>, i64
    %31 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %32 = arith.extsi %31 : tensor<16xi32> to tensor<16xi64>
    %33 = tt.splat %20 : i64 -> tensor<16xi64>
    %34 = arith.addi %33, %32 : tensor<16xi64>
    %35 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %36 = tt.expand_dims %34 {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
    %37 = arith.cmpi slt, %36, %cst_1 : tensor<16x1xi64>
    %38 = tt.expand_dims %35 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %39 = arith.cmpi slt, %38, %cst_7 : tensor<1x128xi32>
    %40 = tt.broadcast %37 : tensor<16x1xi1> -> tensor<16x128xi1>
    %41 = tt.broadcast %39 : tensor<1x128xi1> -> tensor<16x128xi1>
    %42 = arith.andi %40, %41 : tensor<16x128xi1>
    %43 = arith.muli %36, %cst_6 : tensor<16x1xi64>
    %44 = tt.splat %11 : !tt.ptr<bf16> -> tensor<16x1x!tt.ptr<bf16>>
    %45 = tt.addptr %44, %43 : tensor<16x1x!tt.ptr<bf16>>, tensor<16x1xi64>
    %46 = tt.broadcast %45 : tensor<16x1x!tt.ptr<bf16>> -> tensor<16x128x!tt.ptr<bf16>>
    %47 = tt.broadcast %38 : tensor<1x128xi32> -> tensor<16x128xi32>
    %48 = tt.addptr %46, %47 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32>
    %49 = tt.splat %19 : !tt.ptr<bf16> -> tensor<16x1x!tt.ptr<bf16>>
    %50 = tt.addptr %49, %43 : tensor<16x1x!tt.ptr<bf16>>, tensor<16x1xi64>
    %51 = tt.broadcast %50 : tensor<16x1x!tt.ptr<bf16>> -> tensor<16x128x!tt.ptr<bf16>>
    %52 = tt.addptr %51, %47 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32>
    %53 = tt.load %48, %42, %cst_0 : tensor<16x128x!tt.ptr<bf16>>
    %54 = arith.extf %53 : tensor<16x128xbf16> to tensor<16x128xf32>
    %55 = tt.splat %arg7 : f32 -> tensor<16x128xf32>
    %56 = arith.mulf %54, %55 : tensor<16x128xf32>
    %57 = arith.truncf %56 : tensor<16x128xf32> to tensor<16x128xbf16>
    %58 = arith.muli %2, %c128_i32 : i32
    %59 = tt.splat %58 : i32 -> tensor<128xi32>
    %60 = arith.addi %59, %35 : tensor<128xi32>
    %61 = tt.expand_dims %60 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %62 = arith.cmpi slt, %61, %cst_7 : tensor<1x128xi32>
    %63 = tt.broadcast %62 : tensor<1x128xi1> -> tensor<16x128xi1>
    %64 = arith.andi %40, %63 : tensor<16x128xi1>
    %65 = tt.splat %12 : !tt.ptr<bf16> -> tensor<16x1x!tt.ptr<bf16>>
    %66 = tt.addptr %65, %43 : tensor<16x1x!tt.ptr<bf16>>, tensor<16x1xi64>
    %67 = tt.broadcast %66 : tensor<16x1x!tt.ptr<bf16>> -> tensor<16x128x!tt.ptr<bf16>>
    %68 = tt.broadcast %61 : tensor<1x128xi32> -> tensor<16x128xi32>
    %69 = tt.addptr %67, %68 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32>
    %70 = tt.addptr %arg3, %21 : !tt.ptr<f32>, i64
    %71 = tt.splat %70 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %72 = tt.addptr %71, %31 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %73 = tt.addptr %arg4, %21 : !tt.ptr<f32>, i64
    %74 = tt.splat %73 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %75 = tt.addptr %74, %31 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %76 = tt.load %69, %64, %cst_0 : tensor<16x128x!tt.ptr<bf16>>
    %77 = tt.load %72 : tensor<16x!tt.ptr<f32>>
    %78 = tt.load %75 : tensor<16x!tt.ptr<f32>>
    %79 = scf.for %arg11 = %c0_i64 to %25 step %c1_i64 iter_args(%arg12 = %cst_5) -> (tensor<16x128xf32>)  : i64 {
      %82 = arith.addi %21, %arg11 : i64
      %83 = tt.addptr %arg8, %82 : !tt.ptr<i64>, i64
      %84 = tt.load %83 : !tt.ptr<i64>
      %85 = arith.muli %84, %c64_i64 : i64
      %86 = arith.cmpi sle, %85, %1 : i64
      %87 = arith.cmpi sge, %85, %c0_i64 : i64
      %88 = arith.andi %86, %87 : i1
      %89 = scf.if %88 -> (tensor<16x128xf32>) {
        %90 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %91 = arith.extsi %90 : tensor<64xi32> to tensor<64xi64>
        %92 = tt.splat %85 : i64 -> tensor<64xi64>
        %93 = arith.addi %92, %91 : tensor<64xi64>
        %94 = tt.splat %7 : i64 -> tensor<64xi64>
        %95 = arith.cmpi slt, %93, %94 : tensor<64xi64>
        %96 = tt.expand_dims %35 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %97 = tt.splat %29 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %98 = tt.addptr %97, %96 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %99 = tt.expand_dims %93 {axis = 0 : i32} : tensor<64xi64> -> tensor<1x64xi64>
        %100 = arith.muli %99, %cst_4 : tensor<1x64xi64>
        %101 = tt.broadcast %98 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x64x!tt.ptr<bf16>>
        %102 = tt.broadcast %100 : tensor<1x64xi64> -> tensor<128x64xi64>
        %103 = tt.addptr %101, %102 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi64>
        %104 = tt.expand_dims %60 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %105 = tt.splat %30 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %106 = tt.addptr %105, %104 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %107 = tt.broadcast %106 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x64x!tt.ptr<bf16>>
        %108 = tt.addptr %107, %102 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi64>
        %109 = arith.cmpi slt, %96, %cst_3 : tensor<128x1xi32>
        %110 = tt.expand_dims %95 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
        %111 = tt.broadcast %109 : tensor<128x1xi1> -> tensor<128x64xi1>
        %112 = tt.broadcast %110 : tensor<1x64xi1> -> tensor<128x64xi1>
        %113 = arith.andi %111, %112 : tensor<128x64xi1>
        %114 = tt.load %103, %113, %cst : tensor<128x64x!tt.ptr<bf16>>
        %115 = arith.cmpi slt, %104, %cst_3 : tensor<128x1xi32>
        %116 = tt.broadcast %115 : tensor<128x1xi1> -> tensor<128x64xi1>
        %117 = arith.andi %116, %112 : tensor<128x64xi1>
        %118 = tt.load %108, %117, %cst : tensor<128x64x!tt.ptr<bf16>>
        %119 = tt.dot %57, %114, %cst_2 : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
        %120 = tt.expand_dims %77 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %121 = tt.broadcast %120 : tensor<16x1xf32> -> tensor<16x64xf32>
        %122 = arith.subf %119, %121 : tensor<16x64xf32>
        %123 = math.exp %122 : tensor<16x64xf32>
        %124 = tt.splat %1 : i64 -> tensor<64xi64>
        %125 = arith.cmpi sge, %124, %93 : tensor<64xi64>
        %126 = tt.expand_dims %125 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
        %127 = tt.broadcast %126 : tensor<1x64xi1> -> tensor<16x64xi1>
        %128 = arith.select %127, %123, %cst_2 : tensor<16x64xi1>, tensor<16x64xf32>
        %129 = tt.dot %76, %118, %cst_2 : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
        %130 = tt.expand_dims %78 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %131 = tt.broadcast %130 : tensor<16x1xf32> -> tensor<16x64xf32>
        %132 = arith.subf %129, %131 : tensor<16x64xf32>
        %133 = arith.mulf %128, %132 : tensor<16x64xf32>
        %134 = arith.truncf %133 : tensor<16x64xf32> to tensor<16x64xbf16>
        %135 = tt.trans %114 {order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
        %136 = tt.dot %134, %135, %arg12 : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>
        scf.yield %136 : tensor<16x128xf32>
      } else {
        scf.yield %arg12 : tensor<16x128xf32>
      }
      scf.yield %89 : tensor<16x128xf32>
    }
    %80 = arith.mulf %79, %55 : tensor<16x128xf32>
    %81 = arith.truncf %80 : tensor<16x128xf32> to tensor<16x128xbf16>
    tt.store %52, %81, %42 : tensor<16x128x!tt.ptr<bf16>>
    tt.return
  }
}
