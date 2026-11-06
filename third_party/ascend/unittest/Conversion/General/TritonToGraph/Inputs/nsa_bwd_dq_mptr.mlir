// === #2686 SelectedBlockScanRule (bit 65536) — PRODUCTION bwd dq fixture ===
// Provenance: Q3TritonKernel/src/kernels/fla/ops/nsa/nsa_bwd_dq.py
// ::parallel_nsa_bwd_kernel_dq -> ast_to_ttir -> make_ttir() (the TTIR that
// graph-optimize sees; the two decorators @triton.heuristics/@triton.autotune
// are stripped, the kernel body is byte-identical).
// This copy is that IR modulo (a) all loc() dropped (re-printed by triton-opt,
// which does not print debug locs) and (b) the module target attribute switched
// to the Ascend950PR spelling isAscend950() accepts.
// Shapes: B=1 T=63 H=1 HQ=16 G=16 K=V=64 S=16 BS=32 BK=BV=64, dtype f16.
// Variant: const NS (=S=16), no varlen, no block_counts, i32 induction var.
// NOT a lit test: lit.cfg.py excludes Inputs/.
//
// WHY THIS FIXTURE EXISTS: the production kernel addresses its K/V tiles with
// tl.make_block_ptr, so the loop reads them through tt.make_tensor_ptr plus
// boundaryCheck.  Inputs/nsa_bwd_dq_const_ns.mlir came from fla/ops/nsa/
// parallel.py, which spells the same two loads as a two-level tt.addptr chain
// with an explicit mask.  The additive (bwd dq) branch of parseTiles used to
// accept only the addptr spelling ("The additive form: no mptr at all"), so the
// rule matched that fixture and never the real kernel.  Both spellings are
// covered now; this file is the must-match mptr one.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @parallel_nsa_bwd_kernel_dq(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f16>, %arg6: !tt.ptr<f16>, %arg7: f32, %arg8: !tt.ptr<i64>, %arg9: i32, %arg10: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x32xf32>
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c16_i64 = arith.constant 16 : i64
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %2, %arg10 : i32
    %4 = arith.extsi %3 : i32 to i64
    %5 = arith.extsi %0 : i32 to i64
    %6 = arith.addi %4, %5 : i64
    %7 = arith.muli %6, %c1024_i64 : i64
    %8 = tt.addptr %arg0, %7 : !tt.ptr<f16>, i64
    %9 = tt.addptr %arg5, %7 : !tt.ptr<f16>, i64
    %10 = arith.muli %6, %c16_i64 : i64
    %11 = tt.addptr %arg3, %10 : !tt.ptr<f32>, i64
    %12 = tt.addptr %arg4, %10 : !tt.ptr<f32>, i64
    %13 = arith.muli %1, %arg10 : i32
    %14 = arith.extsi %13 : i32 to i64
    %15 = arith.addi %14, %4 : i64
    %16 = arith.addi %15, %5 : i64
    %17 = arith.muli %16, %c1024_i64 : i64
    %18 = tt.addptr %arg6, %17 : !tt.ptr<f16>, i64
    %19 = tt.addptr %arg8, %10 : !tt.ptr<i64>, i64
    %20 = arith.muli %4, %c64_i64 : i64
    %21 = tt.addptr %arg1, %20 : !tt.ptr<f16>, i64
    %22 = tt.addptr %arg2, %20 : !tt.ptr<f16>, i64
    %23 = tt.make_tensor_ptr %8, [%c16_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf16>>
    %24 = tt.make_tensor_ptr %18, [%c16_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf16>>
    %25 = tt.load %23 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x64xf16>>
    %26 = arith.extf %25 : tensor<16x64xf16> to tensor<16x64xf32>
    %27 = tt.splat %arg7 : f32 -> tensor<16x64xf32>
    %28 = arith.mulf %26, %27 : tensor<16x64xf32>
    %29 = arith.truncf %28 : tensor<16x64xf32> to tensor<16x64xf16>
    %30 = arith.muli %1, %c64_i32 : i32
    %31 = tt.make_tensor_ptr %9, [%c16_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %30] {order = array<i32: 1, 0>} : <tensor<16x64xf16>>
    %32 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %33 = tt.splat %11 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %34 = tt.addptr %33, %32 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %35 = tt.splat %12 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %36 = tt.addptr %35, %32 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %37 = tt.load %31 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x64xf16>>
    %38 = tt.load %34 : tensor<16x!tt.ptr<f32>>
    %39 = tt.load %36 : tensor<16x!tt.ptr<f32>>
    %40 = scf.for %arg11 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg12 = %cst) -> (tensor<16x64xf32>)  : i32 {
      %43 = tt.addptr %19, %arg11 : !tt.ptr<i64>, i32
      %44 = tt.load %43 : !tt.ptr<i64>
      %45 = arith.trunci %44 : i64 to i32
      %46 = arith.muli %45, %c32_i32 : i32
      %47 = arith.cmpi sle, %46, %0 : i32
      %48 = arith.cmpi sge, %46, %c0_i32 : i32
      %49 = arith.andi %47, %48 : i1
      %50 = scf.if %49 -> (tensor<16x64xf32>) {
        %51 = arith.extsi %arg10 : i32 to i64
        %52 = tt.make_tensor_ptr %21, [%c64_i64, %51], [%c1_i64, %c64_i64], [%c0_i32, %46] {order = array<i32: 0, 1>} : <tensor<64x32xf16>>
        %53 = tt.make_tensor_ptr %22, [%c64_i64, %51], [%c1_i64, %c64_i64], [%30, %46] {order = array<i32: 0, 1>} : <tensor<64x32xf16>>
        %54 = tt.load %52 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16>>
        %55 = tt.load %53 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16>>
        %56 = tt.dot %29, %54, %cst_0 : tensor<16x64xf16> * tensor<64x32xf16> -> tensor<16x32xf32>
        %57 = tt.expand_dims %38 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %58 = tt.broadcast %57 : tensor<16x1xf32> -> tensor<16x32xf32>
        %59 = arith.subf %56, %58 : tensor<16x32xf32>
        %60 = math.exp %59 : tensor<16x32xf32>
        %61 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
        %62 = tt.splat %46 : i32 -> tensor<32xi32>
        %63 = arith.addi %62, %61 : tensor<32xi32>
        %64 = tt.splat %0 : i32 -> tensor<32xi32>
        %65 = arith.cmpi sge, %64, %63 : tensor<32xi32>
        %66 = tt.expand_dims %65 {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
        %67 = tt.broadcast %66 : tensor<1x32xi1> -> tensor<16x32xi1>
        %68 = arith.select %67, %60, %cst_0 : tensor<16x32xi1>, tensor<16x32xf32>
        %69 = tt.dot %37, %55, %cst_0 : tensor<16x64xf16> * tensor<64x32xf16> -> tensor<16x32xf32>
        %70 = tt.expand_dims %39 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %71 = tt.broadcast %70 : tensor<16x1xf32> -> tensor<16x32xf32>
        %72 = arith.subf %69, %71 : tensor<16x32xf32>
        %73 = arith.mulf %68, %72 : tensor<16x32xf32>
        %74 = arith.truncf %73 : tensor<16x32xf32> to tensor<16x32xf16>
        %75 = tt.trans %54 {order = array<i32: 1, 0>} : tensor<64x32xf16> -> tensor<32x64xf16>
        %76 = tt.dot %74, %75, %arg12 : tensor<16x32xf16> * tensor<32x64xf16> -> tensor<16x64xf32>
        scf.yield %76 : tensor<16x64xf32>
      } else {
        scf.yield %arg12 : tensor<16x64xf32>
      }
      scf.yield %50 : tensor<16x64xf32>
    }
    %41 = arith.mulf %40, %27 : tensor<16x64xf32>
    %42 = arith.truncf %41 : tensor<16x64xf32> to tensor<16x64xf16>
    tt.store %24, %42 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x64xf16>>
    tt.return
  }
}
