// RUN: triton-opt %s --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=T2U
// RUN: triton-opt %s --triton-to-unstructure --bubble-up-operation --triton-to-linalg --verify-each | FileCheck %s --check-prefix=LOWER
// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true compile-mode=simd_simt_template' --bubble-up-operation --triton-to-linalg='compile-on-910-95=true compile-mode=simd_simt_template' --verify-each | FileCheck %s --check-prefix=LOWER
// RUN: sed 's/%mask, %other/%mask/' %s | triton-opt --triton-to-unstructure --bubble-up-operation --triton-to-linalg --verify-each | FileCheck %s --check-prefix=LOWER
// RUN: sed 's/cmpi slt, %x/cmpi eq, %x/' %s | triton-opt --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=KEEP
// RUN: sed 's/cmpi slt, %x/cmpi sge, %x/' %s | triton-opt --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=KEEP
// RUN: sed 's/\(%value = tt.load [^,]*, \)%mask, %other/\1%xmb, %other/' %s | triton-opt --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=ROWONLY
// RUN: sed 's/%mask, %other/%mask/' %s | triton-opt --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=NOOTHER
// RUN: sed 's/%rowsb = tt.splat %rows : i32 -> tensor<3xi32>/%rowsb = arith.constant dense<0> : tensor<3xi32>/' %s | triton-opt --triton-to-unstructure --verify-each | FileCheck %s --check-prefix=EMPTY

// The row addresses are nonstructured, but columns have a fixed stride. Both
// mask bounds must guard memory access, even when the result is selected later.
// Runtime lower/upper bounds also cover nonzero starts and empty intersections.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // T2U-LABEL: tt.func public @mixed_axis_load_mask(
  // T2U: %[[COLMASK:.*]] = tt.broadcast %{{.*}} : tensor<1x32xi1> -> tensor<3x32xi1>
  // T2U: %[[FULLMASK:.*]] = arith.andi
  // T2U: scf.for
  // T2U: %[[MASK:.*]] = tensor.extract_slice %[[COLMASK]][%{{.*}}, 0] [1, 32] [1, 1] {DiscreteMemAccess} : tensor<3x32xi1> to tensor<1x32xi1>
  // T2U: tt.load %{{[^,]+}}, %[[MASK]], %{{.*}} {DiscreteMemAccess} : tensor<1x32x!tt.ptr<f32>>
  // T2U: arith.select %[[FULLMASK]],
  // KEEP-LABEL: tt.func public @mixed_axis_load_mask(
  // KEEP: %[[FULLMASK:.*]] = arith.andi %{{.*}}, %{{.*}} : tensor<3x32xi1>
  // KEEP: scf.for
  // KEEP: %[[MASK:.*]] = tensor.extract_slice %[[FULLMASK]][%{{.*}}, 0] [1, 32] [1, 1] {DiscreteMemAccess} : tensor<3x32xi1> to tensor<1x32xi1>
  // KEEP: tt.load %{{[^,]+}}, %[[MASK]], %{{.*}} {DiscreteMemAccess} : tensor<1x32x!tt.ptr<f32>>
  // LOWER-LABEL: func.func @mixed_axis_load_mask(
  // LOWER: scf.for
  // LOWER: %[[BASE:.*]] = memref.reinterpret_cast
  // LOWER: %[[SOURCE:.*]] = memref.subview %[[BASE]]
  // LOWER: memref.copy %[[SOURCE]], %{{.*}} : memref<1x?xf32,
  // LOWER: return
  // ROWONLY-LABEL: tt.func public @mixed_axis_load_mask(
  // ROWONLY-DAG: %[[ROWMASK:.*]] = tt.broadcast %{{.*}} : tensor<3x1xi1> -> tensor<3x32xi1>
  // ROWONLY-DAG: %[[OTHER:.*]] = arith.constant dense<-7.000000e+00> : tensor<3x32xf32>
  // ROWONLY: scf.for
  // ROWONLY: tt.load %{{[^, ]+}} {DiscreteMemAccess} : tensor<1x32x!tt.ptr<f32>>
  // ROWONLY: arith.select %[[ROWMASK]], %{{.*}}, %[[OTHER]]
  tt.func public @mixed_axis_load_mask(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %rows: i32, %lower: i32, %upper: i32) {
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
    %rm = arith.andi %rmlo, %rmhi : tensor<32xi1>
    %xm2d = tt.expand_dims %xm {axis = 1 : i32} : tensor<3xi1> -> tensor<3x1xi1>
    %rm2d = tt.expand_dims %rm {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
    %xmb = tt.broadcast %xm2d : tensor<3x1xi1> -> tensor<3x32xi1>
    %rmb = tt.broadcast %rm2d : tensor<1x32xi1> -> tensor<3x32xi1>
    %mask = arith.andi %xmb, %rmb : tensor<3x32xi1>
    %other = arith.constant dense<-7.0> : tensor<3x32xf32>
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

  // A row-only bound is completely enforced by the generated loop. Arbitrary
  // tensor padding still selects the original other value outside that loop.
  // T2U-LABEL: tt.func public @bounded_row_load_mask(
  // T2U-SAME: %[[OTHER:[a-zA-Z0-9_]+]]: tensor<3x32xf32>
  // T2U: %[[MASK:.*]] = tt.broadcast %{{.*}} : tensor<3x1xi1> -> tensor<3x32xi1>
  // T2U: scf.for
  // T2U: tt.load %{{[^, ]+}} {DiscreteMemAccess} : tensor<1x32x!tt.ptr<f32>>
  // T2U: arith.select %[[MASK]], %{{.*}}, %[[OTHER]]
  // T2U: tt.return
  // LOWER-LABEL: func.func @bounded_row_load_mask(
  // LOWER: scf.for
  // LOWER-NOT: linalg.fill
  // LOWER: %[[SOURCE:.*]] = memref.reinterpret_cast
  // LOWER-NOT: memref.subview %[[SOURCE]]
  // LOWER-NOT: linalg.fill
  // LOWER: memref.copy %[[SOURCE]], %{{.*}} : memref<1x32xf32,
  // LOWER: return
  // KEEP-LABEL: tt.func public @bounded_row_load_mask(
  // KEEP: scf.for
  // KEEP: tt.load %{{[^,]+}}, %{{[^,]+}}, %{{.*}} {DiscreteMemAccess} : tensor<1x32x!tt.ptr<f32>>
  // NOOTHER-LABEL: tt.func public @bounded_row_load_mask(
  // NOOTHER: scf.for
  // NOOTHER: tt.load %{{[^, ]+}} {DiscreteMemAccess} : tensor<1x32x!tt.ptr<f32>>
  // NOOTHER-NOT: arith.select
  // NOOTHER: tt.return
  // EMPTY-LABEL: tt.func public @bounded_row_load_mask(
  // EMPTY-SAME: %[[OTHER:[a-zA-Z0-9_]+]]: tensor<3x32xf32>
  // EMPTY-NOT: tt.load
  // EMPTY: %[[EMPTY_RESULT:.*]] = arith.select %{{.*}}, %{{.*}}, %[[OTHER]] {DiscreteMemAccess} : tensor<3x32xi1>, tensor<3x32xf32>
  // EMPTY-NOT: tt.load
  // EMPTY: tt.return %[[EMPTY_RESULT]] : tensor<3x32xf32>
  tt.func public @bounded_row_load_mask(%input: !tt.ptr<f32>, %rows: i32, %other: tensor<3x32xf32>) -> tensor<3x32xf32> {
    %x = tt.make_range {start = 0 : i32, end = 3 : i32} : tensor<3xi32>
    %c = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %two = arith.constant dense<2> : tensor<3xi32>
    %row = arith.remsi %x, %two : tensor<3xi32>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
    %col2d = tt.expand_dims %c {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %rowb = tt.broadcast %row2d : tensor<3x1xi32> -> tensor<3x32xi32>
    %colb = tt.broadcast %col2d : tensor<1x32xi32> -> tensor<3x32xi32>
    %offset = arith.addi %rowb, %colb : tensor<3x32xi32>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offset : tensor<3x32x!tt.ptr<f32>>, tensor<3x32xi32>
    %rowsb = tt.splat %rows : i32 -> tensor<3xi32>
    %xm = arith.cmpi slt, %x, %rowsb : tensor<3xi32>
    %xm2d = tt.expand_dims %xm {axis = 1 : i32} : tensor<3xi1> -> tensor<3x1xi1>
    %mask = tt.broadcast %xm2d : tensor<3x1xi1> -> tensor<3x32xi1>
    %result = tt.load %ptr, %mask, %other : tensor<3x32x!tt.ptr<f32>>
    tt.return %result : tensor<3x32xf32>
  }

  // A scalar enable predicate is not implied by a row bound.
  // T2U-LABEL: tt.func public @bounded_row_load_with_gate(
  // T2U-SAME: %[[ENABLED:[a-zA-Z0-9_]+]]: i1
  // T2U: %[[GATE:.*]] = tt.splat %[[ENABLED]] : i1 -> tensor<3x32xi1>
  // T2U: scf.for
  // T2U: %[[GATE_SLICE:.*]] = tensor.extract_slice %[[GATE]][%{{.*}}, 0] [1, 32] [1, 1] {DiscreteMemAccess} : tensor<3x32xi1> to tensor<1x32xi1>
  // T2U: tt.load %{{[^,]+}}, %[[GATE_SLICE]], %{{.*}} {DiscreteMemAccess} : tensor<1x32x!tt.ptr<f32>>
  // T2U: tt.return
  tt.func public @bounded_row_load_with_gate(%input: !tt.ptr<f32>, %rows: i32, %enabled: i1, %other: tensor<3x32xf32>) -> tensor<3x32xf32> {
    %x = tt.make_range {start = 0 : i32, end = 3 : i32} : tensor<3xi32>
    %c = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %two = arith.constant dense<2> : tensor<3xi32>
    %row = arith.remsi %x, %two : tensor<3xi32>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
    %col2d = tt.expand_dims %c {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %rowb = tt.broadcast %row2d : tensor<3x1xi32> -> tensor<3x32xi32>
    %colb = tt.broadcast %col2d : tensor<1x32xi32> -> tensor<3x32xi32>
    %offset = arith.addi %rowb, %colb : tensor<3x32xi32>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offset : tensor<3x32x!tt.ptr<f32>>, tensor<3x32xi32>
    %rowsb = tt.splat %rows : i32 -> tensor<3xi32>
    %xm = arith.cmpi slt, %x, %rowsb : tensor<3xi32>
    %xm2d = tt.expand_dims %xm {axis = 1 : i32} : tensor<3xi1> -> tensor<3x1xi1>
    %rowmask = tt.broadcast %xm2d : tensor<3x1xi1> -> tensor<3x32xi1>
    %gate = tt.splat %enabled : i1 -> tensor<3x32xi1>
    %mask = arith.andi %rowmask, %gate : tensor<3x32xi1>
    %result = tt.load %ptr, %mask, %other : tensor<3x32x!tt.ptr<f32>>
    tt.return %result : tensor<3x32xf32>
  }

  // An unknown lower bound must remain in the access mask. In particular,
  // clipping the loop start to zero does not prove the original start is zero.
  // T2U-LABEL: tt.func public @row_interval_load_mask(
  // T2U-SAME: %[[START_ARG:[a-zA-Z0-9_]+]]: i32, %{{[a-zA-Z0-9_]+}}: tensor<3x32xf32>
  // T2U: %[[LOWER_BOUND:.*]] = tt.splat %[[START_ARG]] : i32 -> tensor<3xi32>
  // T2U: %[[LOWER_MASK:.*]] = arith.cmpi sge, %{{.*}}, %[[LOWER_BOUND]] : tensor<3xi32>
  // T2U: %[[LOWER_2D:.*]] = tt.expand_dims %[[LOWER_MASK]]
  // T2U: %[[LOWER_BROADCAST:.*]] = tt.broadcast %[[LOWER_2D]] : tensor<3x1xi1> -> tensor<3x32xi1>
  // T2U: %[[FULLMASK:.*]] = arith.andi %{{.*}}, %[[LOWER_BROADCAST]] : tensor<3x32xi1>
  // T2U: scf.for
  // T2U: %[[MASK_SLICE:.*]] = tensor.extract_slice %[[FULLMASK]][%{{.*}}, 0] [1, 32] [1, 1] {DiscreteMemAccess} : tensor<3x32xi1> to tensor<1x32xi1>
  // T2U: tt.load %{{[^,]+}}, %[[MASK_SLICE]], %{{.*}} {DiscreteMemAccess} : tensor<1x32x!tt.ptr<f32>>
  // T2U: tt.return
  tt.func public @row_interval_load_mask(%input: !tt.ptr<f32>, %rows: i32, %lower: i32, %other: tensor<3x32xf32>) -> tensor<3x32xf32> {
    %x = tt.make_range {start = 0 : i32, end = 3 : i32} : tensor<3xi32>
    %c = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %two = arith.constant dense<2> : tensor<3xi32>
    %row = arith.remsi %x, %two : tensor<3xi32>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
    %col2d = tt.expand_dims %c {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %rowb = tt.broadcast %row2d : tensor<3x1xi32> -> tensor<3x32xi32>
    %colb = tt.broadcast %col2d : tensor<1x32xi32> -> tensor<3x32xi32>
    %offset = arith.addi %rowb, %colb : tensor<3x32xi32>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<3x32x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offset : tensor<3x32x!tt.ptr<f32>>, tensor<3x32xi32>
    %rowsb = tt.splat %rows : i32 -> tensor<3xi32>
    %xm = arith.cmpi slt, %x, %rowsb : tensor<3xi32>
    %xm2d = tt.expand_dims %xm {axis = 1 : i32} : tensor<3xi1> -> tensor<3x1xi1>
    %lo = tt.splat %lower : i32 -> tensor<3xi32>
    %lower_mask = arith.cmpi sge, %x, %lo : tensor<3xi32>
    %lower2d = tt.expand_dims %lower_mask {axis = 1 : i32} : tensor<3xi1> -> tensor<3x1xi1>
    %rowmask = tt.broadcast %xm2d : tensor<3x1xi1> -> tensor<3x32xi1>
    %lowerb = tt.broadcast %lower2d : tensor<3x1xi1> -> tensor<3x32xi1>
    %mask = arith.andi %rowmask, %lowerb : tensor<3x32xi1>
    %result = tt.load %ptr, %mask, %other : tensor<3x32x!tt.ptr<f32>>
    tt.return %result : tensor<3x32xf32>
  }
}
