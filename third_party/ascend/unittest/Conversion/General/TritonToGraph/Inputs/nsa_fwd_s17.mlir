// #2686 SelectedBlockScanRule: S = 17 (a NON power-of-two index-table row) (provenance)
//   source   : the pre-optimization NSA fwd kernel
//              (.tmp-analysis/ir/snapshots/nsa_fwd.py.20261103T1555, the only
//              surviving copy of the new-fla baseline) compiled with S=17,
//              IS_VARLEN=0, USE_BLOCK_COUNTS=0, BV=64.
//   produced : triton-opt <compiled IR> -o <this>   (locs dropped)
//   target   : Ascend950PR_9579
//   why it exists: the slot axis is padded to TW = next_pow2(S) = 32 and the
//     slot index is clamped to S-1, so the table read never leaves its row.
//     Without the padding, tt.make_range would be asked for a 17-wide range,
//     which its contract ("end must be a power of two") does not allow.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @parallel_nsa_fwd_kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>, %arg4: !tt.ptr<f32>, %arg5: f32, %arg6: !tt.ptr<i64>, %arg7: !tt.ptr<i32>, %arg8: !tt.ptr<i32>, %arg9: !tt.ptr<i32>, %arg10: !tt.ptr<i32>, %arg11: i32, %arg12: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c1_i32 = arith.constant 1 : i32
    %c8192_i64 = arith.constant 8192 : i64
    %c68_i64 = arith.constant 68 : i64
    %cst_0 = arith.constant dense<0xFF800000> : tensor<16x64xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %c512_i64 = arith.constant 512 : i64
    %cst_2 = arith.constant dense<0xFF800000> : tensor<16xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c17_i32 = arith.constant 17 : i32
    %c128_i64 = arith.constant 128 : i64
    %c4_i64 = arith.constant 4 : i64
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.divsi %2, %c4_i32 : i32
    %4 = arith.remsi %2, %c4_i32 : i32
    %5 = arith.muli %3, %arg11 : i32
    %6 = arith.extsi %5 : i32 to i64
    %7 = arith.muli %3, %arg12 : i32
    %8 = arith.extsi %7 : i32 to i64
    %9 = arith.subi %arg12, %arg11 : i32
    %10 = arith.muli %8, %c4_i64 : i64
    %11 = arith.extsi %4 : i32 to i64
    %12 = arith.addi %10, %11 : i64
    %13 = arith.muli %12, %c128_i64 : i64
    %14 = tt.addptr %arg1, %13 : !tt.ptr<bf16>, i64
    %15 = tt.addptr %arg2, %13 : !tt.ptr<bf16>, i64
    %16 = arith.extsi %0 : i32 to i64
    %17 = arith.addi %6, %16 : i64
    %18 = arith.muli %17, %c68_i64 : i64
    %19 = arith.muli %4, %c17_i32 : i32
    %20 = arith.extsi %19 : i32 to i64
    %21 = arith.addi %18, %20 : i64
    %22 = tt.addptr %arg6, %21 : !tt.ptr<i64>, i64
    %23 = arith.muli %17, %c8192_i64 : i64
    %24 = tt.addptr %arg0, %23 : !tt.ptr<bf16>, i64
    %25 = arith.muli %4, %c16_i32 : i32
    %26 = tt.make_tensor_ptr %24, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%25, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x128xbf16>>
    %27 = tt.load %26 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x128xbf16>>
    %28 = arith.extf %27 : tensor<16x128xbf16> to tensor<16x128xf32>
    %29 = tt.splat %arg5 : f32 -> tensor<16x128xf32>
    %30 = arith.mulf %28, %29 : tensor<16x128xf32>
    %31 = arith.truncf %30 : tensor<16x128xf32> to tensor<16x128xbf16>
    %32 = tt.addptr %arg3, %23 : !tt.ptr<bf16>, i64
    %33 = arith.muli %1, %c64_i32 : i32
    %34 = tt.make_tensor_ptr %32, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%25, %33] {order = array<i32: 1, 0>} : <tensor<16x64xbf16>>
    %35 = arith.muli %17, %c64_i64 : i64
    %36 = tt.addptr %arg4, %35 : !tt.ptr<f32>, i64
    %37 = tt.addptr %36, %25 : !tt.ptr<f32>, i32
    %38 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %39 = tt.splat %37 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %40 = tt.addptr %39, %38 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %41 = arith.addi %9, %0 : i32
    %42:3 = scf.for %arg13 = %c0_i32 to %c17_i32 step %c1_i32 iter_args(%arg14 = %cst_1, %arg15 = %cst_2, %arg16 = %cst) -> (tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>)  : i32 {
      %50 = tt.addptr %22, %arg13 : !tt.ptr<i64>, i32
      %51 = tt.load %50 : !tt.ptr<i64>
      %52 = arith.trunci %51 : i64 to i32
      %53 = arith.muli %52, %c64_i32 : i32
      %54 = arith.cmpi sle, %53, %41 : i32
      %55 = arith.cmpi sge, %53, %c0_i32 : i32
      %56 = arith.andi %54, %55 : i1
      %57:3 = scf.if %56 -> (tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>) {
        %58 = arith.extsi %arg12 : i32 to i64
        %59 = tt.make_tensor_ptr %14, [%c128_i64, %58], [%c1_i64, %c512_i64], [%c0_i32, %53] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
        %60 = tt.make_tensor_ptr %15, [%58, %c128_i64], [%c512_i64, %c1_i64], [%53, %33] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %61 = tt.load %59 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
        %62 = tt.load %60 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
        %63 = tt.dot %31, %61, %cst_1 : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
        %64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %65 = tt.splat %53 : i32 -> tensor<64xi32>
        %66 = arith.addi %65, %64 : tensor<64xi32>
        %67 = tt.splat %41 : i32 -> tensor<64xi32>
        %68 = arith.cmpi sge, %67, %66 : tensor<64xi32>
        %69 = tt.expand_dims %68 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
        %70 = tt.broadcast %69 : tensor<1x64xi1> -> tensor<16x64xi1>
        %71 = arith.select %70, %63, %cst_0 : tensor<16x64xi1>, tensor<16x64xf32>
        %72 = "tt.reduce"(%71) <{axis = 1 : i32}> ({
        ^bb0(%arg17: f32, %arg18: f32):
          %88 = arith.maxnumf %arg17, %arg18 : f32
          tt.reduce.return %88 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %73 = arith.maxnumf %arg15, %72 : tensor<16xf32>
        %74 = arith.subf %arg15, %73 : tensor<16xf32>
        %75 = math.exp %74 : tensor<16xf32>
        %76 = tt.expand_dims %73 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %77 = tt.broadcast %76 : tensor<16x1xf32> -> tensor<16x64xf32>
        %78 = arith.subf %71, %77 : tensor<16x64xf32>
        %79 = math.exp %78 : tensor<16x64xf32>
        %80 = arith.mulf %arg16, %75 : tensor<16xf32>
        %81 = "tt.reduce"(%79) <{axis = 1 : i32}> ({
        ^bb0(%arg17: f32, %arg18: f32):
          %88 = arith.addf %arg17, %arg18 : f32
          tt.reduce.return %88 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %82 = arith.addf %80, %81 : tensor<16xf32>
        %83 = tt.expand_dims %75 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %84 = tt.broadcast %83 : tensor<16x1xf32> -> tensor<16x64xf32>
        %85 = arith.mulf %arg14, %84 : tensor<16x64xf32>
        %86 = arith.truncf %79 : tensor<16x64xf32> to tensor<16x64xbf16>
        %87 = tt.dot %86, %62, %85 : tensor<16x64xbf16> * tensor<64x64xbf16> -> tensor<16x64xf32>
        scf.yield %87, %73, %82 : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
      } else {
        scf.yield %arg14, %arg15, %arg16 : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
      }
      scf.yield %57#0, %57#1, %57#2 : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
    }
    %43 = tt.expand_dims %42#2 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %44 = tt.broadcast %43 : tensor<16x1xf32> -> tensor<16x64xf32>
    %45 = arith.divf %42#0, %44 : tensor<16x64xf32>
    %46 = math.log %42#2 : tensor<16xf32>
    %47 = arith.addf %42#1, %46 : tensor<16xf32>
    %48 = arith.truncf %45 : tensor<16x64xf32> to tensor<16x64xbf16>
    tt.store %34, %48 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x64xbf16>>
    %49 = arith.cmpi eq, %1, %c0_i32 : i32
    scf.if %49 {
      tt.store %40, %47 : tensor<16x!tt.ptr<f32>>
    }
    tt.return
  }
}
