// RUN: triton-opt --process-args %s --allow-unregistered-dialect | FileCheck %s

// V scope : two i32 iter_args (%a, %b) are shared between
// block 5 (owner) and block 7 (ssbuffer.clone = 5). Their update chain is a
// single 2-result scf.if whose then branch cross-references the other shared
// arg (%nb = %b + %a). ProcessArgs must clone that scf.if once for block 7 and
// redirect EVERY shared arg in the clone to its own shadow, growing the loop
// from 2 to 4 iter_args. Before the fix only the arg being cloned was remapped,
// so the cloned else-yield kept the original block args and every shadow in
// scf.yield was wrongly taken from result #0 of its cloned scf.if.

// CHECK-LABEL: func.func @test_process_args_clone_if_two_scopes

// ===== V scope: the shared-arg loop is expanded 2 -> 4 results. Capture the
// two shadow block args positionally (IV + 2 originals + 2 shadows). =====
// CHECK: %{{.*}}:4 = scf.for %{{.*}} = {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %[[SA:.*]] = %{{.*}}, %[[SB:.*]] = %{{.*}}) -> (i32, i32, i32, i32)

// Two block-7 clones of the owner scf.if, one per shared arg. Both else-yields
// must forward the two SHADOW args rather than the migrated originals.
// CHECK: scf.yield {{.*}}%[[SA]]{{.*}}%[[SB]]{{.*}}: i32, i32
// CHECK: } {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 7 : i32}
// CHECK: scf.yield {{.*}}%[[SA]]{{.*}}%[[SB]]{{.*}}: i32, i32
// CHECK: } {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32}

// Decisive regression point: original two if results, then the two shadows.
// The second shadow reads result #1 of its OWN cloned if. The buggy pass
// emitted #0 for both shadows (#0, #1, #0, #0).
// CHECK: scf.yield %{{.*}}#0, %{{.*}}#1, %{{.*}}#0, %{{.*}}#1 : i32, i32, i32, i32

// The V scope closes with its VECTOR tcore type, then the C scope opens.
// CHECK: hivm.tcore_type = #hivm.tcore_type<VECTOR>
// C-scope main loop is untouched (its sole iter_arg is not shared): 1 result.
// CHECK: %{{.*}} = scf.for %{{.*}} = {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32)
// The C scope closes only after its loop, so its CUBE tcore type comes last.
// CHECK: hivm.tcore_type = #hivm.tcore_type<CUBE>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_process_args_clone_if_two_scopes() {
    // index constants: scf.for IV / bounds / step must be index type.
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i10 = arith.constant 10 : index
    // i32 constants: iter_arg initial values and integer arithmetic only.
    %c0 = arith.constant {ssbuffer.block_id = 4 : i32} 0 : i32
    %c1 = arith.constant {ssbuffer.block_id = 4 : i32} 1 : i32
    %c10 = arith.constant {ssbuffer.block_id = 4 : i32} 10 : i32
    %c96 = arith.constant {ssbuffer.block_id = 4 : i32} 96 : i32

    // ================= V scope (VECTOR): shared-arg main loop =================
    scope.scope : () -> () {
      %res:2 = scf.for %iv = %i0 to %i10 step %i1
          iter_args(%a = %c0, %b = %c0) -> (i32, i32) {

        // ---- block 5 (owner): one multi-result scf.if update chain ----
        %cond5 = arith.cmpi slt, %a, %c10 {ssbuffer.block_id = 5 : i32} : i32
        %upd:2 = scf.if %cond5 -> (i32, i32) {
          %na = arith.addi %a, %c96 {ssbuffer.block_id = 5 : i32} : i32
          // Cross-reference: updating %b reads the OTHER shared arg %a.
          %nb = arith.addi %b, %a {ssbuffer.block_id = 5 : i32} : i32
          scf.yield {ssbuffer.block_id = 5 : i32} %na, %nb : i32, i32
        } else {
          scf.yield {ssbuffer.block_id = 5 : i32} %a, %b : i32, i32
        } {ssbuffer.block_id = 5 : i32}

        // ---- block 7 (non-owner, clone of block 5): consumes BOTH args ----
        %u7a = arith.addi %a, %c1 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : i32
        %u7b = arith.addi %b, %a {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : i32

        scf.yield %upd#0, %upd#1 : i32, i32
      } {ssbuffer.block_id = 8 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    // ================= C scope (CUBE) =================
    scope.scope : () -> () {
      %cres = scf.for %civ = %i0 to %i10 step %i1
          iter_args(%ca = %c0) -> (i32) {
        %c5 = arith.addi %ca, %c1 {ssbuffer.block_id = 5 : i32} : i32
        %c3 = arith.addi %civ, %i1 {ssbuffer.block_id = 3 : i32, ssbuffer.clone = 5 : i32} : index
        scf.yield %c5 : i32
      } {ssbuffer.block_id = 8 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}

    return
  }
}
