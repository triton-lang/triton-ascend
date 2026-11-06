// === #2686 SelectedBlockScanRule (bit 65536) — attribution guard ===
//
// What this file proves: for every fixture the scan rewrite is pointed at, both the
// "graph-optimize enabled but no rule" baseline (rule-mask=0) and the 11
// baseline-bundle rules (rule-mask=3071, which is also the default mask, since the
// #2686 scan/row-lift bits are fixed default-off) leave the IR byte-for-byte
// unchanged.
// Therefore any diff a rewrite in the #2686 slot produces is attributable to
// it alone -- the same "isolate the rule so nothing can
// be blamed on another graph rule" contract that
// unittest/Conversion/General/TritonToGraph/convert_modulo_to_mask.mlir:8-9
// states in prose.
//
// The FileCheck blocks below are *not* about the rewrite.  They pin the
// pre-rewrite shape (scalar index-table load inside the loop, scf.if, K/V block
// pointers whose token offset is idx*BS, two tt.reduce) so that the "no change"
// diffs cannot pass vacuously: if a fixture stopped containing the pattern (e.g.
// after a frontend change), the diff would still be empty but the CHECKs would
// fail.  The identical CHECK bodies are reused verbatim in
// nsa_selected_block_scan_negative.mlir, where they are paired with the
// rule-mask=65536 runs.
//
// Fixture set covered here: const_ns (positive, R = BN/BS = 1), const_ns_bs32
// (positive, R = 2), dyn_ns (positive, runtime trip count), const_ns_i64_iv
// (positive, i64 induction variable), bwd_dq_const_ns (positive, additive got
// promoted to a positive case on 2026-11-02 -- see group G of
// nsa_selected_block_scan.mlir), varlen and const_ns_rhs_grid (negative), plus
// the BS = 48 and affine-offset/mask-diagonal/tile-width fixtures.
//
// Idiom choice: diff-empty for "unchanged" (repo precedent:
// row-coalescing-dot-fallback.mlir:1-7, row-coalescing-dot-basic.mlir:6-9) and
// FileCheck for "the pattern is really there".  `not` is not used for "no
// change" because in this repo `not` means "the tool must fail"
// (DiscreteMaskAccess/loop_mask_unsupported.mlir:1-2), which is the opposite
// meaning; `not` is used only in nsa_selected_block_scan_driver.mlir.

// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -o %t.const.bare
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=0' -o %t.const.off
// RUN: diff %t.const.bare %t.const.off
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.const.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=3071' -canonicalize -cse -o %t.const.on
// RUN: diff %t.const.ctl %t.const.on
// RUN: FileCheck %s --check-prefix=CONST < %t.const.on

// RUN: triton-opt %S/Inputs/nsa_fwd_varlen.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.varlen.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen.mlir -graph-optimize='rule-mask=3071' -canonicalize -cse -o %t.varlen.on
// RUN: diff %t.varlen.ctl %t.varlen.on
// RUN: FileCheck %s --check-prefix=VARLEN < %t.varlen.on

// RUN: triton-opt %S/Inputs/nsa_fwd_dyn_ns.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.dyns.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_dyn_ns.mlir -graph-optimize='rule-mask=3071' -canonicalize -cse -o %t.dyns.on
// RUN: diff %t.dyns.ctl %t.dyns.on
// RUN: FileCheck %s --check-prefix=DYNS < %t.dyns.on

// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_rhs_grid.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.rhs.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_rhs_grid.mlir -graph-optimize='rule-mask=3071' -canonicalize -cse -o %t.rhs.on
// RUN: diff %t.rhs.ctl %t.rhs.on
// RUN: FileCheck %s --check-prefix=RHSGRID < %t.rhs.on

// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs32.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.bs32.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs32.mlir -graph-optimize='rule-mask=3071' -canonicalize -cse -o %t.bs32.on
// RUN: diff %t.bs32.ctl %t.bs32.on
// RUN: FileCheck %s --check-prefix=BS32 < %t.bs32.on

// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs48.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.bs48.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs48.mlir -graph-optimize='rule-mask=3071' -canonicalize -cse -o %t.bs48.on
// RUN: diff %t.bs48.ctl %t.bs48.on
// RUN: FileCheck %s --check-prefix=BS48 < %t.bs48.on
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_i64_iv.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.i64.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_i64_iv.mlir -graph-optimize='rule-mask=3071' -canonicalize -cse -o %t.i64.on
// RUN: diff %t.i64.ctl %t.i64.on
// RUN: FileCheck %s --check-prefix=I64IV < %t.i64.on

// The additive (bwd dq) fixture joined the attributed set on 2026-11-02, when
// the scan rewrite's additive branch made it a POSITIVE case (group G of
// nsa_selected_block_scan.mlir, prefix BWDREW): the diff the scan rewrite produces there
// must still be attributable to the scan rewrite alone, so the 11 baseline rules have to
// leave the pre-rewrite additive loop exactly as they find it.
// BWDPRE below pins that pre-rewrite shape (scalar per-iteration table read,
// one iter_arg, scf.if, addptr + masked K/V tiles, three dots, tt.trans) so the
// diff-empty assertion cannot pass on a fixture that silently lost the shape;
// `--implicit-check-not='"tt.reduce"'` pins the scan-form weight reduce as
// absent, i.e. an un-rewritten module.
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_const_ns.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.bwd.ctl
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_const_ns.mlir -graph-optimize='rule-mask=3071' -canonicalize -cse -o %t.bwd.on
// RUN: diff %t.bwd.ctl %t.bwd.on
// RUN: FileCheck %s --check-prefix=BWDPRE --implicit-check-not='"tt.reduce"' < %t.bwd.on

// Target sensitivity: the fixture carries the Ascend950PR spelling because
// hacc::utils::isAscend950() accepts it.  The arch string the frontend emits on
// his host is "Ascend910_95", which isAscend950() REJECTS (measured), so with
// that string every 950-only rule stays shut and the IR must not change under
// any mask.  This run pins that fact.
// RUN: sed 's/Ascend950PR_9579/Ascend910_95/' %S/Inputs/nsa_fwd_const_ns.mlir > %t.a595.mlir
// RUN: triton-opt %t.a595.mlir -canonicalize -cse -o %t.a595.ctl
// RUN: triton-opt %t.a595.mlir -graph-optimize='rule-mask=3071' -canonicalize -cse -o %t.a595.on
// RUN: diff %t.a595.ctl %t.a595.on

// ---------------------------------------------------------------------------
// CONST — Inputs/nsa_fwd_const_ns.mlir : the POSITIVE case, untouched
// ---------------------------------------------------------------------------
// No program-grid contract, no coalesce factor: the scan rewrite must not publish one
// and nothing else did either.
// CONST-NOT: hacc.coalesce_factor
// CONST-NOT: hacc.program_grid_transforms
// CONST-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CONST-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// CONST: dense<0xFF800000> : tensor<16x64xf32>

// The loop: constant trip count 16 (= S), i32 induction variable, 3 iter_args
// (o, m, l) — the online-softmax (Form 1) carrying state.
// CONST: scf.for %{{.*}} = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)

// (1) per-iteration scalar index-table load — the op the scan rewrite exists to delete
// CONST: %[[IDXPTR:.*]] = tt.addptr %{{.*}}, %{{.*}} : !tt.ptr<i64>, i32
// CONST: %[[IDX:.*]] = tt.load %[[IDXPTR]] : !tt.ptr<i64>
// CONST: %[[IDX32:.*]] = arith.trunci %[[IDX]] : i64 to i32
// (2) token offset = block_index * BS (BS = 64)
// CONST: %[[TOK:.*]] = arith.muli %[[IDX32]], %c64_i32 : i32
// (3) the range/causal predicate and the scf.if it guards
// CONST: %[[INRANGE:.*]] = arith.cmpi sle, %[[TOK]], %{{.*}} : i32
// CONST: %[[GE0:.*]] = arith.cmpi sge, %[[TOK]], %c0_i32 : i32
// CONST: %[[OK:.*]] = arith.andi %[[INRANGE]], %[[GE0]] : i1
// CONST: %{{.*}} = scf.if %[[OK]] -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// (4) K/V block pointers whose token axis offset is exactly %TOK
// CONST: %{{.*}} = tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// CONST: %{{.*}} = tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %c128_i64], [%c512_i64, %c1_i64], [%[[TOK]], %{{.*}}] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// (5) scores, the causal select and the two reductions (max over s, sum over p)
// CONST: %{{.*}} = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
// CONST: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x64xi1>, tensor<16x64xf32>
// CONST: "tt.reduce"(
// CONST: "tt.reduce"(
// (6) epilogue: the two stores stay outside the loop
// CONST: tt.store %{{.*}} : tensor<16x!tt.ptr<f32>>
// CONST: tt.return

// ---------------------------------------------------------------------------
// VARLEN — Inputs/nsa_fwd_varlen.mlir : negative case, untouched
// ---------------------------------------------------------------------------
// Varlen signature: the trip count comes from a scalar i64 load and the
// induction variable is i64; block_indices/token_indices/offsets are four extra
// scalar loads before the loop.  RFC §6 puts varlen out of scope for the scan rewrite.
// VARLEN-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// VARLEN-COUNT-4: tt.load %{{.*}} : !tt.ptr<i64>
// VARLEN: %[[NS:.*]] = tt.load %{{.*}} : !tt.ptr<i64>
// VARLEN: scf.for %{{.*}} = %c0_i64 to %[[NS]] step %c1_i64
// VARLEN: %[[TOK:.*]] = arith.muli %{{.*}}, %c64_i32 : i32
// VARLEN: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// VARLEN: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// VARLEN: "tt.reduce"(

// ---------------------------------------------------------------------------
// DYNS — Inputs/nsa_fwd_dyn_ns.mlir : positive case (runtime trip count)
// ---------------------------------------------------------------------------
// NS = tt.load(block_counts + ...) -> runtime trip count, i64 induction var.
// Decision D3 (settled 2026-10-31) accepts this, so the fixture is a positive
// case in nsa_selected_block_scan.mlir (group D, prefix DYNREW); here it only
// has to stay untouched by the 11 baseline rules.
// DYNS-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// DYNS: %[[NS:.*]] = tt.load %{{.*}} : !tt.ptr<i64>
// DYNS: scf.for %{{.*}} = %c0_i64 to %[[NS]] step %c1_i64
// DYNS: %[[TOK:.*]] = arith.muli %{{.*}}, %c64_i32 : i32
// DYNS: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// DYNS: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// DYNS: "tt.reduce"(

// ---------------------------------------------------------------------------
// RHSGRID — Inputs/nsa_fwd_const_ns_rhs_grid.mlir : negative case, untouched
// ---------------------------------------------------------------------------
// K and V tile token offsets are program_id.x, i.e. the RHS is anchored on the
// grid axis instead of the index table.  The scalar load and the scf.if are
// still there (so the loop still looks like a gather loop) but the address
// chain no longer goes through %TOK — exactly the shape the scan rewrite must decline.
// RHSGRID-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// RHSGRID: %[[PIDX:.*]] = tt.get_program_id x : i32
// RHSGRID: %{{.*}} = scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// RHSGRID: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[PIDX]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// RHSGRID: tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %c128_i64], [%c512_i64, %c1_i64], [%[[PIDX]], %{{.*}}] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// RHSGRID: "tt.reduce"(

// ---------------------------------------------------------------------------
// I64IV — Inputs/nsa_fwd_const_ns_i64_iv.mlir : positive, i64 induction var
// ---------------------------------------------------------------------------
// Same rewritten shape as CONST (constant trip count 16) but with an i64
// induction variable, which is what the frontend emits as soon as the bound is
// i64.  Decision D4 (settled 2026-10-31) accepts it, so it is a positive case
// in nsa_selected_block_scan.mlir (group E); the 11 registered rules must not
// touch it.
// I64IV-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// I64IV: scf.for %{{.*}} = %c0_i64 to %c16_i64 step %c1_i64
// I64IV: %[[IDXPTR:.*]] = tt.addptr %{{.*}}, %{{.*}} : !tt.ptr<i64>, i64
// I64IV: %[[IDX:.*]] = tt.load %[[IDXPTR]] : !tt.ptr<i64>
// I64IV: %[[TOK:.*]] = arith.muli %{{.*}}, %c64_i32 : i32
// I64IV: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// I64IV: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// I64IV: "tt.reduce"(

// ---------------------------------------------------------------------------
// BS32 — Inputs/nsa_fwd_const_ns_bs32.mlir : positive case (R = 2), untouched
// ---------------------------------------------------------------------------
// The only fixture with R = BN/BS > 1: BS = 32, so BN = max(64, BS) = 64 is two
// blocks wide.  The scan rewrite is pointed at this file in nsa_selected_block_scan.mlir
// group F, so the default rules must leave it alone -- that is what the diff
// above asserts.  Pre-rewrite shape: 32-wide tile views, token offset
// idx * 32, one scf.if and the per-iteration scalar table read.
// BS32-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// BS32: scf.for %{{.*}} = %c0_i32 to %c16_i32 step %c1_i32
// BS32: %[[IDXPTR:.*]] = tt.addptr %{{.*}}, %{{.*}} : !tt.ptr<i64>, i32
// BS32: %[[IDX:.*]] = tt.load %[[IDXPTR]] : !tt.ptr<i64>
// BS32: %[[IDX32:.*]] = arith.trunci %[[IDX]] : i64 to i32
// BS32: %[[TOK:.*]] = arith.muli %[[IDX32]], %c32_i32 : i32
// BS32: %{{.*}} = scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// BS32: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK]]] {order = array<i32: 0, 1>} : <tensor<128x32xbf16>>
// BS32: tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %c128_i64], [%c512_i64, %c1_i64], [%[[TOK]], %{{.*}}] {order = array<i32: 1, 0>} : <tensor<32x128xbf16>>
// BS32: tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x128xbf16> * tensor<128x32xbf16> -> tensor<16x32xf32>
// BS32: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x32xi1>, tensor<16x32xf32>
// BS32: "tt.reduce"(

// BS48 — Inputs/nsa_fwd_const_ns_bs48.mlir : negative case, untouched
// (a 128x48 tile is legal IR only because this fork's build applies
// third_party/ascend/patch/triton-ascend-3.6.0.patch, which relaxes the
// verifyTensorSize power-of-two rule)
// ---------------------------------------------------------------------------
// BS = 48, not a power of two: BN % BS == 0 (the "one block index == BS tokens"
// weight semantics) cannot hold, so the scan rewrite must decline even though everything
// else about the loop is matchable.
// BS48-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// BS48: %[[TOK48:.*]] = arith.muli %{{.*}}, %c48_i32 : i32
// BS48: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// BS48: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK48]]] {order = array<i32: 0, 1>} : <tensor<128x48xbf16>>
// BS48: tt.make_range {end = 48 : i32, start = 0 : i32} : tensor<48xi32>
// BS48: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x48xi1>, tensor<16x48xf32>
// BS48: "tt.reduce"(

// ---------------------------------------------------------------------------
// BWDPRE — Inputs/nsa_bwd_dq_const_ns.mlir : POSITIVE case (additive), untouched
// by the 11 baseline rules
// ---------------------------------------------------------------------------
// Pre-rewrite shape of the bwd dq additive kernel, i.e. exactly what the scan rewrite's
// additive branch consumes: constant-trip-count i32 loop with ONE iter_arg (the
// 16x128 f32 dq accumulator), a per-iteration SCALAR index-table read plus its
// i64 block/token arithmetic, the scf.if guard, K/V read with tt.addptr and an
// explicit mask (including the `tokens < T` compare the rewrite must keep), the
// three dots and tt.trans as the third dot's B operand.  The scan-form weight
// count (`"tt.reduce"`) is absent -- that is the implicit check on the RUN line,
// and it is also the reason this CHECK body would fail on the REWRITTEN module.
// The rewritten form of the same fixture is asserted in group G of
// nsa_selected_block_scan.mlir (prefix BWDREW).
// BWDPRE-LABEL: tt.func public @parallel_nsa_bwd_kernel_dq(
// BWDPRE: scf.for %{{.*}} = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%{{.*}} = %{{.*}}) -> (tensor<16x128xf32>)
// BWDPRE: %[[IDXPTR:.*]] = tt.addptr %{{.*}}, %{{.*}} : !tt.ptr<i64>, i32
// BWDPRE: %[[IDX:.*]] = tt.load %[[IDXPTR]] : !tt.ptr<i64>
// BWDPRE: %[[TOK:.*]] = arith.muli %[[IDX]], %c64_i64 : i64
// BWDPRE: arith.cmpi sle, %[[TOK]], %{{.*}} : i64
// BWDPRE: %{{.*}} = scf.if %{{.*}} -> (tensor<16x128xf32>)
// BWDPRE: arith.cmpi slt, %{{.*}}, %{{.*}} : tensor<64xi64>
// BWDPRE: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x64x!tt.ptr<bf16>>
// BWDPRE: tt.dot %{{.*}} : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
// BWDPRE: tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// BWDPRE: tt.dot %{{.*}} : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>
// BWDPRE: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x128x!tt.ptr<bf16>>
// BWDPRE: tt.return
