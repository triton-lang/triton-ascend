// === #2686 SelectedBlockScanRule (bit 65536) — NEGATIVE: the addptr
// tile load pads with a non-zero "other".
//
// The rewriter re-emits the tile load with other = 0 (the baseline's padding
// value), so a non-zero padding value cannot be reproduced: hasLoadShape requires
// MaskedZeroOther for this spelling.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @parallel_nsa_bwd_kernel_dq(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<bf16>, %arg6: !tt.ptr<bf16>, %arg7: f32, %arg8: !tt.ptr<i64>, %arg9: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xbf16>
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
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
    %7 = arith.extsi %arg9 : i32 to i64
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
    %22 = tt.addptr %arg8, %21 : !tt.ptr<i64>, i64
    %23 = arith.muli %8, %c4_i64 : i64
    %24 = arith.addi %23, %6 : i64
    %25 = arith.muli %24, %c128_i64 : i64
    %26 = tt.addptr %arg1, %25 : !tt.ptr<bf16>, i64
    %27 = tt.addptr %arg2, %25 : !tt.ptr<bf16>, i64
    %28 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %29 = arith.extsi %28 : tensor<16xi32> to tensor<16xi64>
    %30 = tt.splat %20 : i64 -> tensor<16xi64>
    %31 = arith.addi %30, %29 : tensor<16xi64>
    %32 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %33 = tt.expand_dims %31 {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
    %34 = arith.cmpi slt, %33, %cst_1 : tensor<16x1xi64>
    %35 = tt.expand_dims %32 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %36 = arith.cmpi slt, %35, %cst_7 : tensor<1x128xi32>
    %37 = tt.broadcast %34 : tensor<16x1xi1> -> tensor<16x128xi1>
    %38 = tt.broadcast %36 : tensor<1x128xi1> -> tensor<16x128xi1>
    %39 = arith.andi %37, %38 : tensor<16x128xi1>
    %40 = arith.muli %33, %cst_6 : tensor<16x1xi64>
    %41 = tt.splat %11 : !tt.ptr<bf16> -> tensor<16x1x!tt.ptr<bf16>>
    %42 = tt.addptr %41, %40 : tensor<16x1x!tt.ptr<bf16>>, tensor<16x1xi64>
    %43 = tt.broadcast %42 : tensor<16x1x!tt.ptr<bf16>> -> tensor<16x128x!tt.ptr<bf16>>
    %44 = tt.broadcast %35 : tensor<1x128xi32> -> tensor<16x128xi32>
    %45 = tt.addptr %43, %44 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32>
    %46 = tt.splat %19 : !tt.ptr<bf16> -> tensor<16x1x!tt.ptr<bf16>>
    %47 = tt.addptr %46, %40 : tensor<16x1x!tt.ptr<bf16>>, tensor<16x1xi64>
    %48 = tt.broadcast %47 : tensor<16x1x!tt.ptr<bf16>> -> tensor<16x128x!tt.ptr<bf16>>
    %49 = tt.addptr %48, %44 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32>
    %50 = tt.load %45, %39, %cst_0 : tensor<16x128x!tt.ptr<bf16>>
    %51 = arith.extf %50 : tensor<16x128xbf16> to tensor<16x128xf32>
    %52 = tt.splat %arg7 : f32 -> tensor<16x128xf32>
    %53 = arith.mulf %51, %52 : tensor<16x128xf32>
    %54 = arith.truncf %53 : tensor<16x128xf32> to tensor<16x128xbf16>
    %55 = arith.muli %2, %c128_i32 : i32
    %56 = tt.splat %55 : i32 -> tensor<128xi32>
    %57 = arith.addi %56, %32 : tensor<128xi32>
    %58 = tt.expand_dims %57 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %59 = arith.cmpi slt, %58, %cst_7 : tensor<1x128xi32>
    %60 = tt.broadcast %59 : tensor<1x128xi1> -> tensor<16x128xi1>
    %61 = arith.andi %37, %60 : tensor<16x128xi1>
    %62 = tt.splat %12 : !tt.ptr<bf16> -> tensor<16x1x!tt.ptr<bf16>>
    %63 = tt.addptr %62, %40 : tensor<16x1x!tt.ptr<bf16>>, tensor<16x1xi64>
    %64 = tt.broadcast %63 : tensor<16x1x!tt.ptr<bf16>> -> tensor<16x128x!tt.ptr<bf16>>
    %65 = tt.broadcast %58 : tensor<1x128xi32> -> tensor<16x128xi32>
    %66 = tt.addptr %64, %65 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32>
    %67 = tt.addptr %arg3, %21 : !tt.ptr<f32>, i64
    %68 = tt.splat %67 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %69 = tt.addptr %68, %28 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %70 = tt.addptr %arg4, %21 : !tt.ptr<f32>, i64
    %71 = tt.splat %70 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %72 = tt.addptr %71, %28 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %73 = tt.load %66, %61, %cst_0 : tensor<16x128x!tt.ptr<bf16>>
    %74 = tt.load %69 : tensor<16x!tt.ptr<f32>>
    %75 = tt.load %72 : tensor<16x!tt.ptr<f32>>
    %76 = scf.for %arg10 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg11 = %cst_5) -> (tensor<16x128xf32>)  : i32 {
      %79 = tt.addptr %22, %arg10 : !tt.ptr<i64>, i32
      %80 = tt.load %79 : !tt.ptr<i64>
      %81 = arith.muli %80, %c64_i64 : i64
      %82 = arith.cmpi sle, %81, %1 : i64
      %83 = arith.cmpi sge, %81, %c0_i64 : i64
      %84 = arith.andi %82, %83 : i1
      %85 = scf.if %84 -> (tensor<16x128xf32>) {
        %86 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %87 = arith.extsi %86 : tensor<64xi32> to tensor<64xi64>
        %88 = tt.splat %81 : i64 -> tensor<64xi64>
        %89 = arith.addi %88, %87 : tensor<64xi64>
        %90 = tt.splat %7 : i64 -> tensor<64xi64>
        %91 = arith.cmpi slt, %89, %90 : tensor<64xi64>
        %92 = tt.expand_dims %32 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %93 = tt.splat %26 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %94 = tt.addptr %93, %92 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %95 = tt.expand_dims %89 {axis = 0 : i32} : tensor<64xi64> -> tensor<1x64xi64>
        %96 = arith.muli %95, %cst_4 : tensor<1x64xi64>
        %97 = tt.broadcast %94 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x64x!tt.ptr<bf16>>
        %98 = tt.broadcast %96 : tensor<1x64xi64> -> tensor<128x64xi64>
        %99 = tt.addptr %97, %98 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi64>
        %100 = tt.expand_dims %57 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %101 = tt.splat %27 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %102 = tt.addptr %101, %100 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %103 = tt.broadcast %102 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x64x!tt.ptr<bf16>>
        %104 = tt.addptr %103, %98 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi64>
        %105 = arith.cmpi slt, %92, %cst_3 : tensor<128x1xi32>
        %106 = tt.expand_dims %91 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
        %107 = tt.broadcast %105 : tensor<128x1xi1> -> tensor<128x64xi1>
        %108 = tt.broadcast %106 : tensor<1x64xi1> -> tensor<128x64xi1>
        %109 = arith.andi %107, %108 : tensor<128x64xi1>
        %otherOne = arith.constant dense<1.000000e+00> : tensor<128x64xbf16>
        %110 = tt.load %99, %109, %otherOne : tensor<128x64x!tt.ptr<bf16>>
        %111 = arith.cmpi slt, %100, %cst_3 : tensor<128x1xi32>
        %112 = tt.broadcast %111 : tensor<128x1xi1> -> tensor<128x64xi1>
        %113 = arith.andi %112, %108 : tensor<128x64xi1>
        %114 = tt.load %104, %113, %cst : tensor<128x64x!tt.ptr<bf16>>
        %115 = tt.dot %54, %110, %cst_2 : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
        %116 = tt.expand_dims %74 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %117 = tt.broadcast %116 : tensor<16x1xf32> -> tensor<16x64xf32>
        %118 = arith.subf %115, %117 : tensor<16x64xf32>
        %119 = math.exp %118 : tensor<16x64xf32>
        %120 = tt.splat %1 : i64 -> tensor<64xi64>
        %121 = arith.cmpi sge, %120, %89 : tensor<64xi64>
        %122 = tt.expand_dims %121 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
        %123 = tt.broadcast %122 : tensor<1x64xi1> -> tensor<16x64xi1>
        %124 = arith.select %123, %119, %cst_2 : tensor<16x64xi1>, tensor<16x64xf32>
        %125 = tt.dot %73, %114, %cst_2 : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
        %126 = tt.expand_dims %75 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %127 = tt.broadcast %126 : tensor<16x1xf32> -> tensor<16x64xf32>
        %128 = arith.subf %125, %127 : tensor<16x64xf32>
        %129 = arith.mulf %124, %128 : tensor<16x64xf32>
        %130 = arith.truncf %129 : tensor<16x64xf32> to tensor<16x64xbf16>
        %131 = tt.trans %110 {order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
        %132 = tt.dot %130, %131, %arg11 : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>
        scf.yield %132 : tensor<16x128xf32>
      } else {
        scf.yield %arg11 : tensor<16x128xf32>
      }
      scf.yield %85 : tensor<16x128xf32>
    }
    %77 = arith.mulf %76, %52 : tensor<16x128xf32>
    %78 = arith.truncf %77 : tensor<16x128xf32> to tensor<16x128xbf16>
    tt.store %49, %78, %39 : tensor<16x128x!tt.ptr<bf16>>
    tt.return
  }
}
