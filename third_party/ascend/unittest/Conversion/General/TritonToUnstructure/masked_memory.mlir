// RUN: split-file %s %t
// RUN: triton-opt --pass-pipeline='any(triton-control-flow-opt,triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},discrete-mask-access-conversion{compile-mode=simd_simt_template compile-on-910-95=true},triton-to-annotation,triton-to-unstructure{compile-mode=simd_simt_template compile-on-910-95=true})' --verify-each %t/masked_scalar_like_load.mlir | FileCheck %s --check-prefix=SCALAR --implicit-check-not=tt.load
// RUN: triton-opt %t/prefix_zero_load.mlir --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=PREFIX
// RUN: sed 's/, hacc.block_mapping_owner = "block_batching"//' %t/prefix_zero_load.mlir | triton-opt --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=UNBATCHED --implicit-check-not=linalg.fill

//--- masked_scalar_like_load.mlir
// SCALAR-LABEL: tt.func public @block_batching_memory(
// SCALAR-SAME: %[[INPUT:[^:]+]]: !tt.ptr<f32>
// The scalar fallback must bound the load loop by the original predicate.
// A false predicate gives zero iterations; the final select restores other=7.
// No load may escape the loop, even when followed by a select.
// SCALAR: %[[OFFSET:.*]] = arith.constant 1099511627776 : i64
// SCALAR: %[[ZERO:.*]] = arith.constant 0 : index
// SCALAR: %[[ONE:.*]] = arith.constant 1 : index
// SCALAR: %[[FOUR:.*]] = arith.constant 4 : index
// SCALAR: %[[OTHER:.*]] = arith.constant dense<7.000000e+00> : tensor<4xf32>
// SCALAR: %[[ENABLED:.*]] = arith.cmpi slt,
// SCALAR: %[[MASK:.*]] = tt.splat %[[ENABLED]] : i1 -> tensor<4xi1>
// SCALAR: %[[ACTIVE:.*]] = arith.index_castui %[[ENABLED]] : i1 to index
// SCALAR: %[[COUNT:.*]] = arith.muli %[[ACTIVE]], %[[FOUR]] : index
// SCALAR: %[[BOUND:.*]] = arith.minsi %[[COUNT]], %[[FOUR]] : index
// SCALAR: %[[RESULT:.*]] = scf.for %[[IV:.*]] = %[[ZERO]] to %[[BOUND]] step %[[ONE]] iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<4xf32>) {
// SCALAR-NEXT: %[[PTR:.*]] = tt.addptr %[[INPUT]], %[[OFFSET]] : !tt.ptr<f32>, i64
// SCALAR-NEXT: %[[VALUE:.*]] = tt.load %[[PTR]] {DiscreteMemAccess} : !tt.ptr<f32>
// SCALAR-NEXT: %[[INSERTED:.*]] = tensor.insert %[[VALUE]] into %[[ACC]][%[[IV]]] : tensor<4xf32>
// SCALAR-NEXT: scf.yield {DiscreteMemAccess} %[[INSERTED]] : tensor<4xf32>
// SCALAR-NEXT: } {ExtractedLoadOrStore}
// SCALAR-NEXT: %[[SELECTED:.*]] = arith.select %[[MASK]], %[[RESULT]], %[[OTHER]] {DiscreteMemAccess} : tensor<4xi1>, tensor<4xf32>
// SCALAR: tt.store %{{.*}}, %[[SELECTED]] : tensor<4x!tt.ptr<f32>>
// SCALAR: tt.return

module {
  tt.func public @block_batching_memory(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32, %k: i32) {
    %p = tt.get_program_id x : i32
    %one = arith.constant 1.0 : f32
    %zero_i = arith.constant 0 : i32
    %huge = arith.constant 1099511627776 : i64
    %invalid = tt.addptr %x, %huge : !tt.ptr<f32>, i64
    %xp = tt.splat %invalid : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %enabled = arith.cmpi slt, %k, %zero_i : i32
    %mask = tt.splat %enabled : i1 -> tensor<4xi1>
    %other = arith.constant dense<7.0> : tensor<4xf32>
    %value = tt.load %xp, %mask, %other : tensor<4x!tt.ptr<f32>>
    %four = arith.constant 4 : i32
    %base = arith.muli %p, %four : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %bv = tt.splat %base : i32 -> tensor<4xi32>
    %offset = arith.addi %bv, %range : tensor<4xi32>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ya = tt.addptr %yp, %offset : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %ya, %value : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

//--- prefix_zero_load.mlir
// PREFIX-LABEL: tt.func public @prefix_zero_load
// PREFIX: tensor.empty
// PREFIX: linalg.fill
// PREFIX: scf.for
// PREFIX: tt.load %{{[^,]+}}, %{{[^,]+}}, %{{[^ ]+}} {DiscreteMemAccess}
// PREFIX-NOT: arith.select
// PREFIX: tt.return
// UNBATCHED-LABEL: tt.func public @prefix_zero_load
// UNBATCHED: scf.for
// UNBATCHED: tt.load %{{[^,]+}}, %{{[^ ,]+}} {DiscreteMemAccess}
// UNBATCHED: arith.select
// UNBATCHED: tt.return
// Zero padding must be installed in the final loop destination. Retained
// axes still require a masked load with other=0, including empty prefixes.
// Unbatched kernels retain the outer select instead of destination prefill.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.block_mapping_owner = "block_batching"} {
  tt.func public @prefix_zero_load(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %rows: i32, %lower: i32, %upper: i32) {
    %x = tt.make_range {start = 0 : i32, end = 3 : i32} : tensor<3xi32>
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %two = arith.constant dense<2> : tensor<3xi32>
    %stride = arith.constant dense<16> : tensor<32xi32>
    %row = arith.remsi %x, %two : tensor<3xi32>
    %col = arith.muli %r, %stride : tensor<32xi32>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
    %col2d = tt.expand_dims %col {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %rowb = tt.broadcast %row2d : tensor<3x1xi32> -> tensor<3x32xi32>
    %colb = tt.broadcast %col2d : tensor<1x32xi32> -> tensor<3x32xi32>
    %offset = arith.addi %rowb, %colb : tensor<3x32xi32>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offset : tensor<3x32x!tt.ptr<f32>>, tensor<3x32xi32>
    %rowsb = tt.splat %rows : i32 -> tensor<3xi32>
    %xm = arith.cmpi slt, %x, %rowsb : tensor<3xi32>
    %lo = tt.splat %lower : i32 -> tensor<32xi32>
    %hi = tt.splat %upper : i32 -> tensor<32xi32>
    %rmlo = arith.cmpi sge, %r, %lo : tensor<32xi32>
    %rmhi = arith.cmpi slt, %r, %hi : tensor<32xi32>
    %rm = arith.andi %rmhi, %rmhi : tensor<32xi1>
    %xm2d = tt.expand_dims %xm {axis = 1 : i32} : tensor<3xi1> -> tensor<3x1xi1>
    %rm2d = tt.expand_dims %rm {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
    %xmb = tt.broadcast %xm2d : tensor<3x1xi1> -> tensor<3x32xi1>
    %rmb = tt.broadcast %rm2d : tensor<1x32xi1> -> tensor<3x32xi1>
    %mask = arith.andi %xmb, %rmb : tensor<3x32xi1>
    %other = arith.constant dense<0.0> : tensor<3x32xf32>
    %value = tt.load %ptr, %mask, %other : tensor<3x32x!tt.ptr<f32>>
    %outstride = arith.constant dense<32> : tensor<3xi32>
    %outrow = arith.muli %x, %outstride : tensor<3xi32>
    %outrow2d = tt.expand_dims %outrow {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
    %r2d = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %outrowb = tt.broadcast %outrow2d : tensor<3x1xi32> -> tensor<3x32xi32>
    %rb = tt.broadcast %r2d : tensor<1x32xi32> -> tensor<3x32xi32>
    %outoffset = arith.addi %outrowb, %rb : tensor<3x32xi32>
    %outbase = tt.splat %output : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
    %outptr = tt.addptr %outbase, %outoffset : tensor<3x32x!tt.ptr<f32>>, tensor<3x32xi32>
    tt.store %outptr, %value : tensor<3x32x!tt.ptr<f32>>
    tt.return
  }
}
