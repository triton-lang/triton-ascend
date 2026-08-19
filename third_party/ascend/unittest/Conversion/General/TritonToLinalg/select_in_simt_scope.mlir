// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// A masked select inside an explicit SIMT scope must NOT be canonicalized into
// tensor.extract_slice/tensor.insert_slice: the SIMT lowering path needs the
// original arith.select.
//
// Every select result is consumed by a tt.store: the canonicalizer treats an
// unused tensor select as dead code and DCEs it (together with its operands),
// which would empty the function body before the CHECKs run.
// CHECK-LABEL: func.func @select_in_simt_scope
// CHECK-SAME: parallel_mode = "mix_simd_simt"
// CHECK: arith.select
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
tt.func public @select_in_simt_scope(%out: !tt.ptr<f32>) {
  %cst = arith.constant dense<1.000000e+00> : tensor<16xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %cst_1 = arith.constant dense<8> : tensor<16xi32>
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = arith.cmpi slt, %0, %cst_1 : tensor<16xi32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  scope.scope : () -> () {
    %2 = arith.select %1, %cst, %cst_0 : tensor<16xi1>, tensor<16xf32>
    tt.store %ptrs, %2 : tensor<16x!tt.ptr<f32>>
    scope.return
  } {vector_mode = "simt"}
  tt.return
}

// Control: the same select OUTSIDE any SIMT scope is still canonicalized to
// extract_slice + insert_slice. The extract_slice of a splat operand folds to
// a linalg.fill, so only the insert_slice materialization is asserted.
// CHECK-LABEL: func.func @select_outside_scope
// CHECK-NOT: arith.select
// CHECK: tensor.insert_slice
tt.func public @select_outside_scope(%out: !tt.ptr<f32>) {
  %cst = arith.constant dense<1.000000e+00> : tensor<16xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %cst_1 = arith.constant dense<8> : tensor<16xi32>
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = arith.cmpi slt, %0, %cst_1 : tensor<16xi32>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %2 = arith.select %1, %cst, %cst_0 : tensor<16xi1>, tensor<16xf32>
  tt.store %ptrs, %2 : tensor<16x!tt.ptr<f32>>
  tt.return
}

// Nested case: the select's DIRECT parent is scf.if, not the scope — the scope
// is an ancestor further up. isInsideSimtScope must still find the SIMT scope
// through the nesting and leave the select untouched.
// The scf.if condition is a runtime value (get_program_id) so canonicalization
// cannot fold the branch away (a constant-true condition would be folded and
// the CHECK below could never match).
// CHECK-LABEL: func.func @select_nested_in_simt_scope
// CHECK-SAME: parallel_mode = "mix_simd_simt"
// CHECK: scf.if
// CHECK: arith.select
// CHECK-NOT: tensor.extract_slice
tt.func public @select_nested_in_simt_scope(%out: !tt.ptr<f32>) {
  %cst = arith.constant dense<1.000000e+00> : tensor<16xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %cst_1 = arith.constant dense<8> : tensor<16xi32>
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = arith.cmpi slt, %0, %cst_1 : tensor<16xi32>
  %pid = tt.get_program_id x : i32
  %c0_i32 = arith.constant 0 : i32
  %cond = arith.cmpi eq, %pid, %c0_i32 : i32
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  scope.scope : () -> () {
    scf.if %cond {
      %2 = arith.select %1, %cst, %cst_0 : tensor<16xi1>, tensor<16xf32>
      tt.store %ptrs, %2 : tensor<16x!tt.ptr<f32>>
      scf.yield
    }
    scope.return
  } {vector_mode = "simt"}
  tt.return
}
