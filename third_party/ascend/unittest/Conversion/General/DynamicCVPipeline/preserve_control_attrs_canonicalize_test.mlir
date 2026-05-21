// RUN: triton-opt %s --preserve-control-attrs-canonicalize --mlir-print-op-generic | FileCheck %s

// CHECK-LABEL: "func.func"() <{{.*}}sym_name = "for_attr_survives_unused_iter_arg_canonicalize"
// CHECK: "scf.for"
// CHECK: }) {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 1 : i32

module {
  func.func @for_attr_survives_unused_iter_arg_canonicalize(%arg0: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %arg0, %dead = %c0_i32) -> (i32, i32) {
      %1 = arith.addi %acc, %c1_i32 : i32
      scf.yield %1, %dead : i32, i32
    } {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 1 : i32}
    return %0#0 : i32
  }
}

// -----

// CHECK-LABEL: "func.func"() <{{.*}}sym_name = "while_attr_survives_unused_result_canonicalize"
// CHECK: "scf.while"
// CHECK: }) {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 1 : i32} : (i32, i32) -> (i32, i32)

module {
  func.func @while_attr_survives_unused_result_canonicalize(%arg0: i32, %limit: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0:2 = scf.while (%acc = %arg0, %dead = %c0_i32) : (i32, i32) -> (i32, i32) {
      %1 = arith.cmpi slt, %dead, %limit : i32
      scf.condition(%1) %acc, %dead : i32, i32
    } do {
    ^bb0(%acc_iter: i32, %dead_iter: i32):
      %1 = arith.addi %acc_iter, %c1_i32 : i32
      %2 = arith.addi %dead_iter, %c1_i32 : i32
      scf.yield %1, %2 : i32, i32
    } attributes {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 1 : i32}
    return %0#0 : i32
  }
}

// -----

// CHECK-LABEL: "func.func"() <{{.*}}sym_name = "if_attr_survives_unused_result_canonicalize"
// CHECK-DAG: "arith.select"
// CHECK-DAG: "scf.if"
// CHECK: }) {hivm.unlikely_condition, ssbuffer.block_id = 11 : i32} : (i1) -> ()

module {
  func.func @if_attr_survives_unused_result_canonicalize(%cond: i1, %arg0: i32, %buf: memref<1xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0:2 = "scf.if"(%cond) ({
      memref.store %arg0, %buf[%c0] : memref<1xi32>
      "scf.yield"(%arg0, %c0_i32) : (i32, i32) -> ()
    }, {
      memref.store %c0_i32, %buf[%c0] : memref<1xi32>
      "scf.yield"(%c0_i32, %arg0) : (i32, i32) -> ()
    }) {hivm.unlikely_condition, ssbuffer.block_id = 11 : i32} : (i1) -> (i32, i32)
    return %0#0 : i32
  }
}

// -----

// CHECK-LABEL: "func.func"() <{{.*}}sym_name = "merged_zero_result_if_preserves_attrs"
// CHECK: "scf.if"
// CHECK: "memref.store"
// CHECK: "memref.store"
// CHECK: }) {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32} : (i1) -> ()
// CHECK-NOT: "scf.if"
// CHECK: "func.return"

module {
  func.func @merged_zero_result_if_preserves_attrs(%cond: i1, %lhs: memref<1xi32>, %rhs: memref<1xi32>) {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    scf.if %cond {
      memref.store %c1_i32, %lhs[%c0] : memref<1xi32>
    } {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32}
    scf.if %cond {
      memref.store %c2_i32, %rhs[%c0] : memref<1xi32>
    } {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32}
    return
  }
}