// RUN: triton-opt --process-args --split-input-file %s | FileCheck %s

// A tile's next row and column are different results of one conditional.
// Every pipeline block must advance its own complete pair of coordinates,
// including the values captured inside both branches of the conditional.
// CHECK-LABEL: func.func @for_coupled_state
// CHECK: iter_args(%[[FR:.*]] = %c0_i32, %[[FC:.*]] = %c1_i32, %[[FLR:.*]] = %c0_i32, %[[FLC:.*]] = %c1_i32)
// CHECK: %[[FNEXT:.*]]:2 = scf.if
// CHECK: %[[FSUM:.*]] = arith.addi %[[FR]], %[[FC]]
// CHECK: scf.yield %[[FC]], %[[FSUM]] : i32, i32
// CHECK: scf.yield %[[FC]], %[[FR]] : i32, i32
// CHECK: arith.addi %[[FLR]], %[[FLC]]
// CHECK: %[[FLOCAL:.*]]:2 = scf.if
// CHECK: %[[FLSUM:.*]] = arith.addi %[[FLR]], %[[FLC]]
// CHECK: scf.yield{{.*}} %[[FLC]], %[[FLSUM]] : i32, i32
// CHECK: scf.yield{{.*}} %[[FLC]], %[[FLR]] : i32, i32
// CHECK: scf.yield %[[FNEXT]]#0, %[[FNEXT]]#1, %[[FLOCAL]]#0, %[[FLOCAL]]#1 : i32, i32, i32, i32
module {
  func.func @for_coupled_state(%condition: i1) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %result:2 = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%row = %c0_i32, %col = %c1_i32) -> (i32, i32) {
      %next:2 = scf.if %condition -> (i32, i32) {
        %sum = arith.addi %row, %col {ssbuffer.block_id = 1 : i32} : i32
        scf.yield %col, %sum : i32, i32
      } else {
        scf.yield %col, %row : i32, i32
      } {ssbuffer.block_id = 1 : i32}
      %consumer = arith.addi %row, %col {ssbuffer.block_id = 2 : i32} : i32
      scf.yield %next#0, %next#1 : i32, i32
    } {ssbuffer.main_loop = 0 : i32}
    return %result#0, %result#1 : i32, i32
  }
}

// -----

// Exercise the same sharing through scf.while, independently of its exit
// condition. The condition here is loop invariant.
// CHECK-LABEL: func.func @while_coupled_state
// CHECK: ^bb0(%[[WR:.*]]: i32, %[[WC:.*]]: i32, %[[WLR:.*]]: i32, %[[WLC:.*]]: i32):
// CHECK: %[[WNEXT:.*]]:2 = scf.if
// CHECK: %[[WSUM:.*]] = arith.addi %[[WR]], %[[WC]]
// CHECK: scf.yield %[[WC]], %[[WSUM]] : i32, i32
// CHECK: scf.yield %[[WC]], %[[WR]] : i32, i32
// CHECK: arith.addi %[[WLR]], %[[WLC]]
// CHECK: %[[WLOCAL:.*]]:2 = scf.if
// CHECK: %[[WLSUM:.*]] = arith.addi %[[WLR]], %[[WLC]]
// CHECK: scf.yield{{.*}} %[[WLC]], %[[WLSUM]] : i32, i32
// CHECK: scf.yield{{.*}} %[[WLC]], %[[WLR]] : i32, i32
// CHECK: scf.yield %[[WNEXT]]#0, %[[WNEXT]]#1, %[[WLOCAL]]#0, %[[WLOCAL]]#1 : i32, i32, i32, i32
module {
  func.func @while_coupled_state(%condition: i1) -> (i32, i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result:2 = scf.while (%row = %c0, %col = %c1) : (i32, i32) -> (i32, i32) {
      scf.condition(%condition) %row, %col : i32, i32
    } do {
    ^bb0(%row: i32, %col: i32):
      %next:2 = scf.if %condition -> (i32, i32) {
        %sum = arith.addi %row, %col {ssbuffer.block_id = 1 : i32} : i32
        scf.yield %col, %sum : i32, i32
      } else {
        scf.yield %col, %row : i32, i32
      } {ssbuffer.block_id = 1 : i32}
      %consumer = arith.addi %row, %col {ssbuffer.block_id = 2 : i32} : i32
      scf.yield %next#0, %next#1 : i32, i32
    } attributes {ssbuffer.main_loop = 0 : i32}
    return %result#0, %result#1 : i32, i32
  }
}

// -----

// Both coordinates contribute to termination, so each block needs its own
// condition state as well as the correctly indexed conditional updates.
// CHECK-LABEL: func.func @while_condition_coupled_state
// CHECK: ^bb0(%[[CR:.*]]: i32, %[[CC:.*]]: i32, %[[C1R:.*]]: i32, %[[C2R:.*]]: i32, %[[C1C:.*]]: i32, %[[C2C:.*]]: i32, %[[CLR:.*]]: i32, %[[CLC:.*]]: i32):
// CHECK: %[[CNEXT:.*]]:2 = scf.if
// CHECK: arith.addi %[[CR]], %[[CC]]
// CHECK: %[[C1NEXT:.*]]:2 = scf.if
// CHECK: %[[C1SUM:.*]] = arith.addi %[[C1R]], %[[C1C]]
// CHECK: scf.yield{{.*}} %[[C1C]], %[[C1SUM]] : i32, i32
// CHECK: scf.yield{{.*}} %[[C1C]], %[[C1R]] : i32, i32
// CHECK: arith.addi %[[CLR]], %[[CLC]]
// CHECK: %[[C2NEXT:.*]]:2 = scf.if
// CHECK: %[[C2SUM:.*]] = arith.addi %[[C2R]], %[[C2C]]
// CHECK: scf.yield{{.*}} %[[C2C]], %[[C2SUM]] : i32, i32
// CHECK: scf.yield{{.*}} %[[C2C]], %[[C2R]] : i32, i32
// CHECK: %[[CLNEXT:.*]]:2 = scf.if
// CHECK: %[[CLSUM:.*]] = arith.addi %[[CLR]], %[[CLC]]
// CHECK: scf.yield{{.*}} %[[CLC]], %[[CLSUM]] : i32, i32
// CHECK: scf.yield{{.*}} %[[CLC]], %[[CLR]] : i32, i32
// CHECK: scf.yield %[[CNEXT]]#0, %[[CNEXT]]#1, %[[C1NEXT]]#0, %[[C2NEXT]]#0, %[[C1NEXT]]#1, %[[C2NEXT]]#1, %[[CLNEXT]]#0, %[[CLNEXT]]#1 : i32, i32, i32, i32, i32, i32, i32, i32
module {
  func.func @while_condition_coupled_state(%condition: i1, %limit: i32) -> (i32, i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result:2 = scf.while (%row = %c0, %col = %c1) : (i32, i32) -> (i32, i32) {
      %total = arith.addi %row, %col : i32
      %keep_going = arith.cmpi slt, %total, %limit : i32
      scf.condition(%keep_going) %row, %col : i32, i32
    } do {
    ^bb0(%row: i32, %col: i32):
      %next:2 = scf.if %condition -> (i32, i32) {
        %sum = arith.addi %row, %col {ssbuffer.block_id = 1 : i32} : i32
        scf.yield %col, %sum : i32, i32
      } else {
        scf.yield %col, %row : i32, i32
      } {ssbuffer.block_id = 1 : i32}
      %consumer = arith.addi %row, %col {ssbuffer.block_id = 2 : i32} : i32
      scf.yield %next#0, %next#1 : i32, i32
    } attributes {ssbuffer.main_loop = 0 : i32}
    return %result#0, %result#1 : i32, i32
  }
}

// -----

// A directly forwarded state has no defining operation. It still needs a
// yielded value for each block-local iter_arg introduced by ProcessArgs.
// CHECK-LABEL: func.func @for_forwarded_state
// CHECK-SAME: (%[[FINIT:.*]]: i32)
// CHECK: iter_args(%[[FORIGINAL:.*]] = %[[FINIT]], %[[FFORWARDED:.*]] = %[[FINIT]])
// CHECK: arith.addi %[[FORIGINAL]], %[[FORIGINAL]]
// CHECK: arith.addi %[[FFORWARDED]], %[[FFORWARDED]]
// CHECK: scf.yield %[[FORIGINAL]], %[[FFORWARDED]] : i32, i32
module {
  func.func @for_forwarded_state(%initial: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%state = %initial) -> (i32) {
      %owner = arith.addi %state, %state {ssbuffer.block_id = 1 : i32} : i32
      %consumer = arith.addi %state, %state {ssbuffer.block_id = 2 : i32} : i32
      scf.yield %state : i32
    } {ssbuffer.main_loop = 0 : i32}
    return %result : i32
  }
}

// -----

// CHECK-LABEL: func.func @while_forwarded_state
// CHECK: ^bb0(%[[WORIGINAL:.*]]: i32, %[[WFORWARDED:.*]]: i32):
// CHECK: arith.addi %[[WORIGINAL]], %[[WORIGINAL]]
// CHECK: arith.addi %[[WFORWARDED]], %[[WFORWARDED]]
// CHECK: scf.yield %[[WORIGINAL]], %[[WFORWARDED]] : i32, i32
module {
  func.func @while_forwarded_state(%initial: i32, %condition: i1) -> i32 {
    %result = scf.while (%state = %initial) : (i32) -> i32 {
      scf.condition(%condition) %state : i32
    } do {
    ^bb0(%state: i32):
      %owner = arith.addi %state, %state {ssbuffer.block_id = 1 : i32} : i32
      %consumer = arith.addi %state, %state {ssbuffer.block_id = 2 : i32} : i32
      scf.yield %state : i32
    } attributes {ssbuffer.main_loop = 0 : i32}
    return %result : i32
  }
}

// -----

// The consumer reads only the row. Advancing its row nevertheless requires
// the matching column state; borrowing the producer's column mixes tiles.
// CHECK-LABEL: func.func @for_state_dependency
// CHECK-SAME: (%[[DINIT:.*]]: i32)
// CHECK: iter_args(%[[DR:.*]] = %[[DINIT]], %[[DC:.*]] = %c1_i32, %[[DLR:.*]] = %[[DINIT]], %[[DLC:.*]] = %c1_i32)
// CHECK: %[[DNEXTR:.*]] = arith.addi %[[DR]], %[[DC]]
// CHECK: %[[DNEXTC:.*]] = arith.addi %[[DC]], %c1_i32
// CHECK: arith.addi %[[DLR]], %c1_i32
// CHECK-DAG: %[[DLNEXTR:.*]] = arith.addi %[[DLR]], %[[DLC]]
// CHECK-DAG: %[[DLNEXTC:.*]] = arith.addi %[[DLC]], %c1_i32
// CHECK: scf.yield %[[DNEXTR]], %[[DNEXTC]], %[[DLNEXTR]], %[[DLNEXTC]] : i32, i32, i32, i32
module {
  func.func @for_state_dependency(%initial: i32) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %result:2 = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%row = %initial, %col = %c1_i32) -> (i32, i32) {
      %next_row = arith.addi %row, %col {ssbuffer.block_id = 1 : i32} : i32
      %next_col = arith.addi %col, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      %consumer = arith.addi %row, %c1_i32 {ssbuffer.block_id = 2 : i32} : i32
      scf.yield %next_row, %next_col : i32, i32
    } {ssbuffer.main_loop = 0 : i32}
    return %result#0, %result#1 : i32, i32
  }
}

// -----

// Only the row contributes directly to the while condition. Its per-block
// update still depends on a block-local column and that column's update.
// CHECK-LABEL: func.func @while_condition_state_dependency
// CHECK: ^bb0(%[[QR:.*]]: i32, %[[QC:.*]]: i32, %[[Q1R:.*]]: i32, %[[Q2R:.*]]: i32, %[[Q1C:.*]]: i32, %[[Q2C:.*]]: i32, %[[QLR:.*]]: i32, %[[QLC:.*]]: i32):
// CHECK: %[[QNEXTR:.*]] = arith.addi %[[QR]], %[[QC]]
// CHECK: %[[QNEXTC:.*]] = arith.addi %[[QC]], %c1_i32
// CHECK-DAG: %[[Q1NEXTR:.*]] = arith.addi %[[Q1R]], %[[Q1C]]
// CHECK-DAG: %[[Q1NEXTC:.*]] = arith.addi %[[Q1C]], %c1_i32
// CHECK: arith.addi %[[QLR]], %c1_i32
// CHECK-DAG: %[[Q2NEXTR:.*]] = arith.addi %[[Q2R]], %[[Q2C]]
// CHECK-DAG: %[[Q2NEXTC:.*]] = arith.addi %[[Q2C]], %c1_i32
// CHECK-DAG: %[[QLNEXTR:.*]] = arith.addi %[[QLR]], %[[QLC]]
// CHECK-DAG: %[[QLNEXTC:.*]] = arith.addi %[[QLC]], %c1_i32
// CHECK: scf.yield %[[QNEXTR]], %[[QNEXTC]], %[[Q1NEXTR]], %[[Q2NEXTR]], %[[Q1NEXTC]], %[[Q2NEXTC]], %[[QLNEXTR]], %[[QLNEXTC]] : i32, i32, i32, i32, i32, i32, i32, i32
module {
  func.func @while_condition_state_dependency(%initial: i32, %limit: i32) -> (i32, i32) {
    %c1 = arith.constant 1 : i32
    %result:2 = scf.while (%row = %initial, %col = %c1) : (i32, i32) -> (i32, i32) {
      %keep_going = arith.cmpi slt, %row, %limit : i32
      scf.condition(%keep_going) %row, %col : i32, i32
    } do {
    ^bb0(%row: i32, %col: i32):
      %next_row = arith.addi %row, %col {ssbuffer.block_id = 1 : i32} : i32
      %next_col = arith.addi %col, %c1 {ssbuffer.block_id = 1 : i32} : i32
      %consumer = arith.addi %row, %c1 {ssbuffer.block_id = 2 : i32} : i32
      scf.yield %next_row, %next_col : i32, i32
    } attributes {ssbuffer.main_loop = 0 : i32}
    return %result#0, %result#1 : i32, i32
  }
}

// -----

// The first state consumes a value from a later block. Cloning its update for
// block 3 needs a local copy of the second state, but must preserve block 2 as
// the owner of the second state's original update.
// CHECK-LABEL: func.func @for_dependency_preserves_owner
// CHECK-SAME: (%[[PINIT:[a-zA-Z0-9_]+]]: i32)
// CHECK: iter_args(%[[PA:[a-zA-Z0-9_]+]] = %[[PINIT]], %[[PB:[a-zA-Z0-9_]+]] = %c1_i32, %[[PLA:[a-zA-Z0-9_]+]] = %[[PINIT]], %[[PLB:[a-zA-Z0-9_]+]] = %c1_i32)
// CHECK: arith.addi %[[PA]], %c1_i32 {ssbuffer.block_id = 1 : i32}
// CHECK: %[[PNEXTB:[a-zA-Z0-9_]+]] = arith.addi %[[PB]], %c1_i32 {ssbuffer.block_id = 2 : i32}
// CHECK: %[[PNEXTA:[a-zA-Z0-9_]+]] = arith.addi %[[PA]], %[[PNEXTB]] {ssbuffer.block_id = 1 : i32}
// CHECK: arith.addi %[[PLA]], %c1_i32
// CHECK: %[[PLNEXTB:[a-zA-Z0-9_]+]] = arith.addi %[[PLB]], %c1_i32
// CHECK: %[[PLNEXTA:[a-zA-Z0-9_]+]] = arith.addi %[[PLA]], %[[PLNEXTB]]
// CHECK: scf.yield %[[PNEXTA]], %[[PNEXTB]], %[[PLNEXTA]], %[[PLNEXTB]] : i32, i32, i32, i32
module {
  func.func @for_dependency_preserves_owner(%initial: i32) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %result:2 = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%a = %initial, %b = %c1_i32) -> (i32, i32) {
      %owner_a = arith.addi %a, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      %next_b = arith.addi %b, %c1_i32 {ssbuffer.block_id = 2 : i32} : i32
      %next_a = arith.addi %a, %next_b {ssbuffer.block_id = 1 : i32} : i32
      %consumer_a = arith.addi %a, %c1_i32 {ssbuffer.block_id = 3 : i32} : i32
      scf.yield %next_a, %next_b : i32, i32
    } {ssbuffer.main_loop = 0 : i32}
    return %result#0, %result#1 : i32, i32
  }
}

// -----

// B is read only by the loop yield, so it has no compute-block owner. The
// consumer's A update still needs its own B state, including B's initial
// value before the update to the loop-invariant constant takes effect.
// CHECK-LABEL: func.func @for_forwarded_dependency_without_owner
// CHECK-SAME: (%[[UINIT:[a-zA-Z0-9_]+]]: i32)
// CHECK: iter_args(%[[UA:[a-zA-Z0-9_]+]] = %[[UINIT]], %[[UB:[a-zA-Z0-9_]+]] = %c0_i32, %[[ULA:[a-zA-Z0-9_]+]] = %[[UINIT]], %[[ULB:[a-zA-Z0-9_]+]] = %c0_i32)
// CHECK: arith.addi %[[UA]], %c1_i32 {ssbuffer.block_id = 1 : i32}
// CHECK: arith.addi %[[ULA]], %c1_i32 {{.*}}ssbuffer.block_id = 3 : i32}
// CHECK: scf.yield %[[UB]], %c1_i32, %[[ULB]], %c1_i32 : i32, i32, i32, i32
module {
  func.func @for_forwarded_dependency_without_owner(%initial: i32) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %result:2 = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%a = %initial, %b = %c0_i32) -> (i32, i32) {
      %owner_a = arith.addi %a, %c1_i32 {ssbuffer.block_id = 1 : i32} : i32
      %consumer_a = arith.addi %a, %c1_i32 {ssbuffer.block_id = 3 : i32} : i32
      scf.yield %b, %c1_i32 : i32, i32
    } {ssbuffer.main_loop = 0 : i32}
    return %result#0, %result#1 : i32, i32
  }
}
