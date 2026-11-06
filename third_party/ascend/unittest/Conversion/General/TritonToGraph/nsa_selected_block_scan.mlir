// === #2686 SelectedBlockScanRule (bit 65536) — POSITIVE case ===
//
// The scan rewrite never commits alone: the plan factory always hands its
// private sandbox to the row lift's real materialize entry (QueryRowBatching.h)
// and lets the row lift decide, so the form that lands in production is always
// the JOINT scan + row lift one.
// The A-only IR that the former test-only escape hatch (removed 2026-11-05)
// used to pin is no longer reachable through the pass, so the positive
// assertions in this file are the joint ones.
//
//  A. The DESIGN-TARGET assertions (FileCheck on Inputs/nsa_fwd_ruleA_design_target.mlir):
//     that fixture is the compiled gold standard (the Rule-A-only reference
//     implementation run through the early TTIR pipeline), re-printed once by
//     `triton-opt <file> -o <new>` without --mlir-print-debuginfo.  It is
//     therefore the SAME IR WITH THE LOCATIONS DROPPED -- not a byte-for-byte
//     copy of the reference artifact: every #loc alias is gone, and with it the
//     SSA value names the name locs used to supply (%q, %k_5, ... are now
//     %arg0, %6, ...).  The op stream, the types and the module attribute are
//     unchanged.  Keeping an implementation-independent artifact in the tree is
//     what makes these CHECK lines verifiable against something other than the
//     rule itself; otherwise every assertion about the rewritten form is
//     circular.
//
//  B. The JOINT assertions (`-graph-optimize='rule-mask=196608'` vs
//     `'rule-mask=65536'`): bit 131072 only decides whether the row lift is ALSO
//     scheduled standalone, so the two masks must produce identical IR (C below).
//
// What the fixture is: the REAL NSA forward kernel IR (nsa_fwd.py) after the early
// TTIR pipeline, i.e. exactly what graph-optimize sees in production
// (third_party/ascend/backend/compiler.py:309-327).  Shapes G=16, K=V=128, S=16,
// BS=64, bf16; constant trip count (NS = S = 16); online-softmax (Form 1:
// iter_args o/m/l).  See Inputs/nsa_fwd_const_ns.mlir for provenance.
//
// Why rule-mask=65536 and nothing else: attribution isolation, the same contract
// that convert_modulo_to_mask.mlir:8-9 states in prose.
// nsa_selected_block_scan_attribution.mlir proves that today's 3071 rules do not
// touch this fixture.
//
// Coverage note (2026-11-05, escape hatch removed): the A-only fingerprints
// (REWRITTEN/R2REW/DYNREW/BWDREW/VARLENREW/S17REW/VARLENBCREW) are gone with it,
// because that IR can no longer be produced by the pass.  The fixtures those
// groups covered are still exercised end to end by the joint smoke block at the
// end of this file (verify + idempotency); their exact joint fingerprints should
// be re-derived from a `triton-opt` dump before the MR.

// --- A. design target: the CHECK lines below, checked without the rule -----
// RUN: triton-opt %S/Inputs/nsa_fwd_ruleA_design_target.mlir -o %t.design
// RUN: FileCheck %s --check-prefix=REWRITTEN --implicit-check-not='scf.if' < %t.design
// RUN: FileCheck %s --check-prefix=NOSCALAR < %t.design

// --- control run: the fixture with the rule off -----------------------------
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.control

// --- The scan rewrite has no separate row-lift gate ------------------------------------
// It always runs the row lift's real implementation (the QueryRowBatching.h
// materialize entry) inside its own sandbox and lets the row lift decide, so bit
// 65536 alone must produce exactly the same IR as 65536|131072: the 131072 bit
// only decides whether the row lift is ALSO scheduled as a standalone rule, where
// it finds nothing left to lift.  (Bits 16/17 are fixed default-off, so the default
// mask stays 3071 and every run here enables them explicitly; see
// nsa_selected_block_scan_attribution.mlir for the baseline-vs-enabled split.)
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608' -canonicalize -cse -o %t.joint
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.gated
// RUN: diff %t.joint %t.gated
// ... and the joint rewrite must still differ from the rule-off run, so that
// equality cannot pass vacuously:
// RUN: not diff %t.control %t.gated

// ---------------------------------------------------------------------------
// REWRITTEN — the shape the rewrite must produce.  The design target
// (Inputs/nsa_fwd_ruleA_design_target.mlir) is the compiled gold standard with
// the locations dropped, so group A checks this fingerprint WITHOUT running the
// rule at all.  The assertions anchor semantic fingerprints only (vector table
// read / scan upper bound / one loop with 3 iter_args / weight reduce / safe_m /
// both dots keeping M = 16), never implementation detail.
// ---------------------------------------------------------------------------
// REWRITTEN-LABEL: tt.func public @parallel_nsa_fwd_kernel
// NEGATIVE 1: no branch may remain in the loop; NEGATIVE 2: no per-iteration
// scalar index read; NEGATIVE 3: the block-level i32 predicate must be gone
// (a token-level tensor<64xi32> sle is allowed)
// REWRITTEN-NOT: scf.if
// REWRITTEN-NOT: tt.load %{{.*}} : !tt.ptr<i64>
// REWRITTEN-NOT: arith.cmpi sle, %{{.*}}, %{{.*}} : i32
// POSITIVE 1: the whole index table is read once as a vector
// REWRITTEN: tt.load %{{.*}} : tensor<16x!tt.ptr<i64>>
// POSITIVE 2: the scan upper bound comes from cdiv(i_t + 1, BN)
// REWRITTEN: arith.divsi %{{.*}}, %{{.*}} : i32
// POSITIVE 3: still one scf.for, still three iter_args (o, m, l)
// REWRITTEN: %{{.*}}:3 = scf.for {{.*}} iter_args(
// POSITIVE 4: the weight reduce (along the S = 16 axis)
// REWRITTEN: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
// POSITIVE 5: the first dot still has M = 16 (not widened by row merging)
// REWRITTEN: tt.dot %{{.*}} : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
// POSITIVE 6: the safe_m guard is in place
// REWRITTEN: arith.cmpf oeq, %{{.*}}, %{{.*}} : tensor<16xf32>
// POSITIVE 7: the second dot still has M = 16
// REWRITTEN: tt.dot %{{.*}} : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>

// ---------------------------------------------------------------------------
// NOSCALAR — the three properties the row lift's matcher needs, expressed as "nothing
// forbidden from the loop onwards".  The NOT region starts at the `scf.for` match
// and runs to end of file (the loop is the last construct: in the after-IR and in
// any sane rewrite the only thing after it is the epilogue + tt.return).
// This also holds for the dynamic-NS fixtures: a surviving block_counts load
// sits BEFORE the first scf.for, outside the forbidden region.
// ---------------------------------------------------------------------------
// NOSCALAR-NOT: hacc.program_grid_transforms
// NOSCALAR-NOT: hacc.coalesce_factor
// NOSCALAR-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// NOSCALAR: scf.for
// NOSCALAR-NOT: : !tt.ptr<i64>
// NOSCALAR-NOT: scf.if
// NOSCALAR-NOT: scf.for

// ---------- joint golden standard: scan + row lift (JOINTREW) ---------------
// (2026-11-03) The joint flavour is the launch-contract one: the rewrite commits
// hacc.program_grid_transforms (the launcher divides the grid by the row factor)
// and the kernel keeps "program id == query row". The in-kernel flavour (flat
// program id + `if active`) that the hand-written reference `nsa_fwd_new.py`
// uses is a *different* plumbing; SelectedBlockScanRule must never emit both
// (see the mutual-exclusion check in the rule and 15-v3-realignment.md 0.1).
//
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608' -o %t.joint
// RUN: FileCheck %s --check-prefix=JOINTREW < %t.joint

// JOINTREW: hacc.program_grid_transforms = {extent_source = "runtime_original_grid"
// the joint function grows the hidden original-grid arguments the row lift appends
// JOINTREW: tt.func public @parallel_nsa_fwd_kernel(
// no per-block branch survives: the scan loop is the only loop
// JOINTREW-NOT: scf.if
// JOINTREW: scf.for
// the per-token weight reduce is present (axis 2, the lifted row axis), so A ran
// JOINTREW: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// and the rewritten body still ends with the masked stores
// JOINTREW: tt.store
// JOINTREW: tt.return

// ---------- bwd dq dyn NS: folded index address (BWDYNJOINT) ---------------
// (2026-11-03) The dyn-NS spelling of the bwd dq kernel folds the row advance
// into the in-loop index address:
//     tt.addptr %root, arith.addi(%rowOffset, %iv)
// The dense kernels keep it as a separate SSA value instead:
//     tt.addptr %rowBase, %iv      (%rowBase = tt.addptr %root, %rowOffset)
// The address is the same, but parseIndexTable used to insist that the offset be
// exactly the induction variable, so this spelling was rejected -- which left the
// block_counts production path of the bwd dq kernel silently unmatchable.
// Inputs/nsa_bwd_dq_dyn_ns.mlir is exactly that IR.  The rewrite re-creates the
// row base the vectorised read needs (tt.addptr %root, %rowOffset) in the
// prologue, so both spellings land on the same scan form.

// the joint gate accepts the same shape and publishes the runtime grid contract
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_dyn_ns.mlir -graph-optimize='rule-mask=196608' -o %t.bwdyndyn.joint
// RUN: FileCheck %s --check-prefix=BWDYNJOINT < %t.bwdyndyn.joint

// BWDYNJOINT: hacc.program_grid_transforms = {extent_source = "runtime_original_grid"
// BWDYNJOINT: tt.func public @parallel_nsa_bwd_kernel_dq(
// BWDYNJOINT-NOT: scf.if
// BWDYNJOINT: scf.for
// BWDYNJOINT: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// BWDYNJOINT: tt.store
// BWDYNJOINT: tt.return

// ---------- varlen + block_counts: 7 loop-invariant scalar loads ------------
// (2026-11-03) The production varlen dispatch also passes block_counts, so the
// loop-invariant scalar loads before the loop are 2 token_indices_q + 2
// cu_seqlens_q + 2 cu_seqlens_k + 1 block_counts = 7, not the 6 of the pure
// varlen prefix.  The skeleton budget used to be six and rejected the whole
// dispatch before parseIndexTable ran.  Inputs/nsa_fwd_varlen_bc.mlir is that IR.
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen_bc.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.varlenbc.ctl

// (meta) the joint gate must still refuse it: the row lift does not lift varlen, so
// offering only bit 65536 must leave the IR untouched.
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen_bc.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.varlenbc.gated
// RUN: diff %t.varlenbc.ctl %t.varlenbc.gated

// ---------- H == 1 + dyn NS: table length from the row stride (H1DYNREW) -----
// (2026-11-03) With H == 1 the head split folds away, so the row base loses its
// i_h*S addend and the row stride of block_indices is H*S == S.  The table
// length used to be recoverable only from a CONSTANT trip count, so the dyn-NS
// (block_counts) H == 1 decode shape never matched.  The matcher now reads S off
// the constant multiplier of the pid-carrying row offset, but only when the
// function carries no `i_bh % H` remainder -- H > 1 always leaves one, and with
// H > 1 that same constant would be H*S, i.e. a vector read spilling into the
// next head's row.
// Inputs/nsa_fwd_h1_dyn_ns.mlir is that IR (H=1 HQ=16 G=16 K=V=64 S=16 BS=32).
// The runtime decode offset uses a last-valid-row bound under the no-wrap input contract. The bounded variant also exercises the static-grid contract.
// RUN: triton-opt %S/Inputs/nsa_fwd_h1_dyn_ns.mlir -graph-optimize='rule-mask=196608' -canonicalize -cse -o %t.h1dyn.joint
// RUN: FileCheck %s --check-prefix=H1GROUP < %t.h1dyn.joint
// RUN: sed 's/%7 = arith.subi %arg9, %arg8 : i32/%7 = arith.constant 64 : i32/; s/module attributes {/module attributes {hacc.grid_num_tiles = 512 : i32,/' %S/Inputs/nsa_fwd_h1_dyn_ns.mlir > %t.h1bounded.mlir
// RUN: triton-opt %t.h1bounded.mlir -graph-optimize='rule-mask=196608' -o %t.h1bounded
// RUN: FileCheck %s --check-prefix=H1DYNREW < %t.h1bounded
// H1DYNREW: hacc.coalesce_factor = 32 : i32
// H1DYNREW: tt.func public @parallel_nsa_fwd_kernel(
// H1DYNREW-NOT: scf.if
// H1DYNREW: tensor<32x16x!tt.ptr<i64>>
// H1DYNREW: scf.for
// H1DYNREW: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// H1DYNREW: tt.store
// H1DYNREW: tt.return

// ---------- joint fingerprints (dumps taken from the bf34ac235 triton-opt) ---
// These stand in for the scan-only fingerprints that went away with the escape
// hatch: every line below is measured on the form production actually commits,
// i.e. the scan rewrite plus the row lift, with the row factor the row-count
// policy picks for this shape (R = 16 unless noted).

// --- dyn NS: the block_counts trip count survives outside the lifted rows ----
// RUN: triton-opt %S/Inputs/nsa_fwd_dyn_ns.mlir -graph-optimize='rule-mask=196608' -canonicalize -cse -o %t.dynj
// RUN: FileCheck %s --check-prefix=DYNJOINT < %t.dynj
// DYNJOINT: hacc.program_grid_transforms = {extent_source = "runtime_original_grid"
// DYNJOINT: tt.func public @parallel_nsa_fwd_kernel(
// DYNJOINT-NOT: scf.if
// DYNJOINT: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x16x!tt.ptr<i64>>
// DYNJOINT: %{{.*}}:4 = scf.for {{.*}} iter_args({{.*}}) -> (tensor<256x64xf32>, tensor<256x64xf32>, tensor<256xf32>, tensor<256xf32>)
// DYNJOINT: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// DYNJOINT: tt.dot %{{.*}} : tensor<256x128xbf16> * tensor<128x64xbf16> -> tensor<256x64xf32>
// DYNJOINT-COUNT-2: tt.dot %{{.*}} : tensor<256x64xbf16> * tensor<64x64xbf16> -> tensor<256x64xf32>
// DYNJOINT: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x16x64x!tt.ptr<bf16>>
// DYNJOINT: tt.return

// --- BS = 32 (R = BN/BS = 2 inside the scan rewrite): the R axis survives the lifting --
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs32.mlir -graph-optimize='rule-mask=196608' -canonicalize -cse -o %t.bs32j
// RUN: FileCheck %s --check-prefix=BS32JOINT < %t.bs32j
// BS32JOINT: hacc.program_grid_transforms = {extent_source = "runtime_original_grid"
// BS32JOINT: tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// BS32JOINT: tt.broadcast %{{.*}} : tensor<16x1x16xi32> -> tensor<16x2x16xi32>
// BS32JOINT-NOT: scf.if
// BS32JOINT: %{{.*}}:4 = scf.for {{.*}} iter_args({{.*}}) -> (tensor<256x64xf32>, tensor<256x64xf32>, tensor<256xf32>, tensor<256xf32>)
// BS32JOINT: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// BS32JOINT: tt.dot %{{.*}} : tensor<256x128xbf16> * tensor<128x64xbf16> -> tensor<256x64xf32>
// BS32JOINT: tt.return

// --- S = 17: the slot axis is padded to next_pow2(S) = 32 ------------------
// RUN: triton-opt %S/Inputs/nsa_fwd_s17.mlir -graph-optimize='rule-mask=196608' -canonicalize -cse -o %t.s17j
// RUN: FileCheck %s --check-prefix=S17JOINT < %t.s17j
// S17JOINT: hacc.program_grid_transforms = {extent_source = "runtime_original_grid"
// S17JOINT: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x32x!tt.ptr<i64>>
// S17JOINT-NOT: scf.if
// S17JOINT: %{{.*}}:3 = scf.for {{.*}} iter_args({{.*}}) -> (tensor<256x64xf32>, tensor<256xf32>, tensor<256xf32>)
// S17JOINT: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// S17JOINT: tt.dot %{{.*}} : tensor<256x128xbf16> * tensor<128x64xbf16> -> tensor<256x64xf32>
// S17JOINT: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x16x64x!tt.ptr<bf16>>
// S17JOINT: tt.return

// --- bwd dq additive form: R = 2, one carried accumulator, three dots ------
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_const_ns.mlir -graph-optimize='rule-mask=196608' -canonicalize -cse -o %t.bwdj
// RUN: FileCheck %s --check-prefix=BWDJOINT < %t.bwdj
// BWDJOINT: hacc.program_grid_transforms = {extent_source = "runtime_original_grid"
// BWDJOINT: factor = 2 : i64
// BWDJOINT: tt.func public @parallel_nsa_bwd_kernel_dq(
// BWDJOINT-NOT: scf.if
// BWDJOINT: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x16x!tt.ptr<i64>>
// BWDJOINT: %{{.*}} = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<32x128xf32>)
// BWDJOINT: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// BWDJOINT-COUNT-2: tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x128xbf16> * tensor<128x64xbf16> -> tensor<32x64xf32>
// BWDJOINT: tt.dot %{{.*}}, %{{.*}}, %arg13 : tensor<32x64xbf16> * tensor<64x128xbf16> -> tensor<32x128xf32>
// BWDJOINT: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x16x128x!tt.ptr<bf16>>
// BWDJOINT: tt.return

// --- non-950 target: the joint rule leaves the module alone ----------------
// The target gate lives in the row lift's scheduler entry (isAscend950), so with the
// joint mask the whole rule declines on a 910_95 module, prints nothing, and
// the IR stays byte-identical to the rule-off run.
// RUN: sed 's/Ascend950PR_9579/Ascend910_95/' %S/Inputs/nsa_fwd_const_ns.mlir > %t.a595.mlir
// RUN: triton-opt %t.a595.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.a595.ctl
// RUN: triton-opt %t.a595.mlir -graph-optimize='rule-mask=196608' -canonicalize -cse -o %t.a595.out 2> %t.a595.err
// RUN: diff %t.a595.ctl %t.a595.out
// RUN: echo -n > %t.empty
// RUN: diff %t.empty %t.a595.err

// ---------- joint smoke: verify + idempotency for the remaining fixtures ---
// i64_iv reuses the dense fwd joint form (same shapes, only the induction
// variable width differs) and the two varlen fixtures decline, so they get the
// structural assertion instead of a fingerprint: the pipeline verifies, a
// second application is a no-op, and the pass never fails.
// RUN: for f in nsa_fwd_const_ns_i64_iv nsa_fwd_varlen nsa_fwd_varlen_bc; do \
// RUN:   triton-opt %S/Inputs/$f.mlir -graph-optimize='rule-mask=196608' --verify-each -canonicalize -cse -o %t.$f.once && \
// RUN:   triton-opt %t.$f.once -graph-optimize='rule-mask=196608' --verify-each -canonicalize -cse -o %t.$f.twice && \
// RUN:   diff %t.$f.once %t.$f.twice || exit 1; done

// The H=1 dynamic-offset form uses a last-valid-row bound. The scan rewrite still
// commits only when its joint row-batching query succeeds.
// H1GROUP: hacc.program_grid_transforms
// H1GROUP-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// H1GROUP: arith.minsi
// H1GROUP-NOT: arith.maxsi {{.*}} : i32
// H1GROUP: scf.for
// H1GROUP: tt.dot
