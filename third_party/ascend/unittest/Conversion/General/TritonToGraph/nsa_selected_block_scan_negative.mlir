// === #2686 SelectedBlockScanRule (bit 65536) — NEGATIVE cases ===
//
// STATUS: post-implementation form; every RUN line below is green.  What the
// file now covers: (b) K/V anchored on the grid axis instead of the
// index table, (d) the table address is not the induction variable, (e) the
// causal mask carries a non-zero token-side shift, (f) the tile is not exactly
// one block wide on the token axis.  Sections kept the original lettering so the
// (c) hole below is visible rather than silently renumbered away.
//
// (g) is GONE (deleted 2026-11-02, not renumbered): it used to pin a refusal for
// the bwd dq additive form (Inputs/nsa_bwd_dq_const_ns.mlir).  The scan rewrite now
// ACCEPTS that shape -- the additive branch (single iter_arg carried state,
// tt.addptr + explicit-mask tiles, exp -> where -> *w -> (do v - delta) ->
// dot(ds, trans(k), acc)) landed -- so the fixture is a POSITIVE case whose
// assertions live in nsa_selected_block_scan.mlir, group G, check prefix
// BWDREW.  Nothing about it belongs in this file any more: "declines and is
// byte-identical" is the opposite of the current contract.
//
// What "must hold" means here, precisely: the output must be BYTE-IDENTICAL to
// the rule-off run, and triton-opt must not print a single line on stderr.
// Every fixture below is a shape the scan rewrite's matcher rejects, so the pass declines
// before the row lift is ever consulted; the two halves are asserted as:
//   * diff  -> "no change"            (repo precedent row-coalescing-dot-fallback.mlir:1-7)
//   * FileCheck -> "the pre-rewrite pattern is really there", so the diff cannot
//     pass because the fixture silently stopped containing the shape.
// Isolation note: the scan rewrite never commits alone (the plan factory always hands its
// sandbox to the row lift's real materialize entry), so "no change" is a joint verdict
// and cannot by itself prove which of the two matchers declined.  The former
// test-only escape hatch that isolated the scan rewrite is gone (2026-11-05); what keeps
// this group specific is that each fixture pins a defect the scan rewrite's parser owns
// (the FileCheck below shows the pre-rewrite shape really is there), and Rule
// B's matcher rejects the sparse gather form on its own account as well.
//
// The `echo -n` + `diff` pair is the "no diagnostics on the decline path" check:
// an unsupported rewrite must leave the input AND its diagnostics untouched.
// The empty file is created ONCE here, before the first section that needs it
// (section (c) below uses it too; it used to be created in section (a), i.e.
// after its first use, which only worked on a dirty Output dir).
// RUN: echo -n > %t.empty
//
// ---------- (c) BS is not a power of two -----------------------------------
// BS = 48 is not a power of two, so "one index-table entry == BS consecutive
// tokens" (the weight semantics) is not expressible and the rewrite must
// decline (matchSelectedBlockScan's isPowerOfTwo(tileBlockSize) guard).
//
// Why this fixture is legal HERE: a 128x48 tile has a non-power-of-two element
// count (6144) that OpTrait::impl::verifyTensorSize rejects -- but this fork's
// build applies third_party/ascend/patch/triton-ascend-3.6.0.patch, which
// relaxes exactly that check ("FIXME:patched triton community").  So judge this
// case on the patched tree (what setup.py / CI build); on a tree where someone
// ran restore_sources.py the fixture does not even parse.
//
// The sibling guard is unreachable for any legal BS: after isPowerOfTwo(:927)
// and BN = max(64, BS)(:929), `scanTileWidth % blockSize != 0`(:930) can never
// fire.  The BS < 64 positive side (R = BN/BS = 2) is group F of
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs48.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.bs48.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs48.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.bs48
// RUN: diff %t.bs48.ctl %t.bs48
// RUN: FileCheck %s --check-prefix=BS48 < %t.bs48
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs48.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.bs48.err
// RUN: diff %t.empty %t.bs48.err

// BS48-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// BS48: %[[TOK48:.*]] = arith.muli %{{.*}}, %c48_i32 : i32
// BS48: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// BS48: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK48]]] {order = array<i32: 0, 1>} : <tensor<128x48xbf16>>
// BS48: tt.make_range {end = 48 : i32, start = 0 : i32} : tensor<48xi32>
// BS48: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x48xi1>, tensor<16x48xf32>
// BS48: "tt.reduce"(

// ---------- (a) varlen: the joint gate still declines it -------------------
// (2026-11-03) The scan rewrite now RECOGNIZES the varlen dispatch: the row position
// (token_indices_q) and the K/V base (cu_seqlens) are loop-invariant scalar
// loads, which is exactly what matchRowPosition / checkAddressOperand accept
// after the varlen relaxation.  What must still hold is the JOINT contract:
// with only the scan rewrite's bit enabled the function is left BYTE-IDENTICAL,
// because the row lift declines varlen and the scan rewrite must never land alone
// (it trades O(NS) block reads for an O(TK) token scan).
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.varlen.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.varlen
// RUN: diff %t.varlen.ctl %t.varlen
// RUN: FileCheck %s --check-prefix=VARLEN < %t.varlen
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.varlen.err
// RUN: diff %t.empty %t.varlen.err

// VARLEN-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// VARLEN-COUNT-4: tt.load %{{.*}} : !tt.ptr<i64>
// VARLEN: %[[NS:.*]] = tt.load %{{.*}} : !tt.ptr<i64>
// VARLEN: scf.for %{{.*}} = %c0_i64 to %[[NS]] step %c1_i64
// VARLEN: %[[TOK:.*]] = arith.muli %{{.*}}, %c64_i32 : i32
// VARLEN: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// VARLEN: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// VARLEN: "tt.reduce"(

// ---------- (b) the RHS is anchored on the grid axis ------------------------
// K and V tile token offsets are program_id.x (= the query position) instead of
// the index-table entry: the address chain no longer goes through the scanned
// block index, so this is a dense/window read, not a selected-block scan.
// Downstream reason: Rules/DotRowCoalescing.cpp:294-306 (a lifted,
// pid-dependent dot B operand must be refused).
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_rhs_grid.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.rhs.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_rhs_grid.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.rhs
// RUN: diff %t.rhs.ctl %t.rhs
// RUN: FileCheck %s --check-prefix=RHSGRID < %t.rhs
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_rhs_grid.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.rhs.err
// RUN: diff %t.empty %t.rhs.err

// RHSGRID-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// RHSGRID: %[[PIDX:.*]] = tt.get_program_id x : i32
// RHSGRID: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// RHSGRID: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[PIDX]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// RHSGRID: tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %c128_i64], [%c512_i64, %c1_i64], [%[[PIDX]], %{{.*}}] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// RHSGRID: "tt.reduce"(

// ---------- (d) the index-table address is not the induction variable ------
// The per-iteration table entry is addressed as table[P + i_h*S + (i + 1)]:
// the offset operand of the scalar tt.addptr is arith.addi(%iv, %c1_i32), so
// the baseline reads table[1..S+1).  The scan rewrite reads the whole table
// once as table[0, S) and then indexes it by the iteration number, which is
// the same slot only when that offset is exactly the induction variable; an
// a*iv+b form (scale != 1 or bias != 0) therefore pins the wrong slots and is
// refused (matcher A2/B1).  Measured before the fix: the rewrite went through
// and dropped the +1 (163 diff lines, empty stderr).
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_affine_offset.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.aff.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_affine_offset.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.aff
// RUN: diff %t.aff.ctl %t.aff
// RUN: FileCheck %s --check-prefix=AFFOFF < %t.aff
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_affine_offset.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.aff.err
// RUN: diff %t.empty %t.aff.err

// AFFOFF-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// AFFOFF: %[[BIA:.*]] = arith.addi %{{.*}}, %c1_i32 : i32
// AFFOFF: %{{.*}} = tt.addptr %{{.*}}, %[[BIA]] : !tt.ptr<i64>, i32
// AFFOFF: %[[TOK:.*]] = arith.muli %{{.*}}, %c64_i32 : i32
// AFFOFF: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// AFFOFF: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// AFFOFF: "tt.reduce"(

// ---------- (e) the causal mask has a non-zero token offset ---------------
// The tensor-level causal mask compares (i_s + [0, BS) + 1) <= i_t, i.e. the
// token side carries a diagonal = 1 shift.  The rewrite rebuilds the token
// positions itself as token_start + [0, BN) with no offset, so the shift would
// be dropped (and with it one token per block moves in/out of the causal
// window).  The scan rewrite now requires the token side to be exactly
// splat(token_offset) + tt.make_range{start = 0, end = BS} over the same SSA
// value as a tile token offset (matcher A8 site 3 / B2); any extra arithmetic
// is refused instead of normalized.  Measured before the fix: rewrite went
// through, +1 gone (164 diff lines, empty stderr).
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_mask_diagonal.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.maskd.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_mask_diagonal.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.maskd
// RUN: diff %t.maskd.ctl %t.maskd
// RUN: FileCheck %s --check-prefix=MASKD < %t.maskd
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_mask_diagonal.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.maskd.err
// RUN: diff %t.empty %t.maskd.err

// MASKD-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// MASKD: %[[DIDX:.*]] = arith.trunci %{{.*}} : i64 to i32
// MASKD: %[[DTOK:.*]] = arith.muli %[[DIDX]], %c64_i32 : i32
// MASKD: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// MASKD: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[DTOK]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// MASKD: %[[DRNG:.*]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// MASKD: %[[DSPL:.*]] = tt.splat %[[DTOK]] : i32 -> tensor<64xi32>
// MASKD: %[[DBASE:.*]] = arith.addi %[[DSPL]], %[[DRNG]] : tensor<64xi32>
// MASKD: %[[DSHIFT:.*]] = arith.addi %[[DBASE]], %{{.*}} : tensor<64xi32>
// MASKD: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x64xi1>, tensor<16x64xf32>
// MASKD: "tt.reduce"(

// ---------- (f) the tile is not one block wide on the token axis -----------
// The K/V tile views (and the tensors mirroring them) are 32 tokens wide while
// the token offset still advances by BS = 64 per iteration: a legal module
// whose width/stride agreement is broken.  blockSize is read off the
// token-offset multiplier, and the rewrite then pins the tile token extent to
// BN = 64 unconditionally, so it would widen the tile and change how many
// tokens a selected block accumulates.  The scan rewrite now requires the pointee
// shape[tokenAxis] of both tile pointers to equal blockSize (matcher A4/B3).
// Measured before the fix: rewrite went through, '<tensor<128x32xbf16>>'
// became '<tensor<128x64xbf16>>' (162 diff lines, empty stderr).
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_tile_width.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.tilew.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_tile_width.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.tilew
// RUN: diff %t.tilew.ctl %t.tilew
// RUN: FileCheck %s --check-prefix=TILEW < %t.tilew
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_tile_width.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.tilew.err
// RUN: diff %t.empty %t.tilew.err

// TILEW-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// TILEW: %[[WIDX:.*]] = arith.trunci %{{.*}} : i64 to i32
// TILEW: %[[WTOK:.*]] = arith.muli %[[WIDX]], %c64_i32 : i32
// TILEW: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// TILEW: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[WTOK]]] {order = array<i32: 0, 1>} : <tensor<128x32xbf16>>
// TILEW: tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %c128_i64], [%c512_i64, %c1_i64], [%[[WTOK]], %{{.*}}] {order = array<i32: 1, 0>} : <tensor<32x128xbf16>>
// TILEW: tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// TILEW: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x32xi1>, tensor<16x32xf32>
// TILEW: "tt.reduce"(

// ---------- (h) the layout says H > 1 but the row base has no i_h*S term ----
// nsa_fwd_h1_dyn_ns.mlir is accepted because H == 1 folds the i_h*S slot addend
// away, which makes the pid row-offset multiplier equal to S.  The same IR with
// an `i_bh % H` remainder (H = 2) present is the opposite layout: there the
// multiplier is H*S, so a vector read of that width would run past the row into
// the next head.  S cannot be recovered from the IR in that case, so the matcher
// must decline (loopCarriesHeadSplit in deriveTableLength) instead of guessing.
// The remainder is a dead op -- it exists only to state H > 1, as it does in the
// real kernel before folding.
// RUN: triton-opt %S/Inputs/nsa_fwd_h1_dyn_ns_headsplit.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.headsplit.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_h1_dyn_ns_headsplit.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.headsplit
// RUN: diff %t.headsplit.ctl %t.headsplit
// RUN: FileCheck %s --check-prefix=HEADSPLIT < %t.headsplit
// RUN: triton-opt %S/Inputs/nsa_fwd_h1_dyn_ns_headsplit.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.headsplit.err
// RUN: diff %t.empty %t.headsplit.err

// HEADSPLIT-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// The gather loop, its scalar table read and the guarded region must all
// survive: canonicalize drops the dead `remsi` that states H > 1, so assert the
// shape the rule refused to rewrite instead.
// HEADSPLIT: scf.for
// HEADSPLIT: tt.load %{{.*}} : !tt.ptr<i64>
// HEADSPLIT: scf.if

// ---------- (i) an extra tensor load inside the gather loop ---------------
// The rewrite replaces the whole loop body, so it can only accept bodies whose
// every load it rebuilds: one scalar index read plus the two K/V tiles. The
// extra load here would be dropped silently (parseIndexTable: load census).
// RUN: triton-opt %S/Inputs/nsa_fwd_extra_loop_load.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.extra.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_extra_loop_load.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.extra
// RUN: diff %t.extra.ctl %t.extra
// RUN: FileCheck %s --check-prefix=EXTRA < %t.extra
// RUN: triton-opt %S/Inputs/nsa_fwd_extra_loop_load.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.extra.err
// RUN: diff %t.empty %t.extra.err

// EXTRA-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// EXTRA: scf.for
// EXTRA: tt.load %{{.*}} : !tt.ptr<i64>
// EXTRA: scf.if

// ---------- (j) the scalar index read carries a mask ----------------------
// A masked entry load is not the pattern: the vector read the rewrite emits
// has no per-iteration predicate to carry (hasLoadShape: Unmasked).
// RUN: triton-opt %S/Inputs/nsa_fwd_masked_index_load.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.idxmask.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_masked_index_load.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.idxmask
// RUN: diff %t.idxmask.ctl %t.idxmask
// RUN: FileCheck %s --check-prefix=IDXMASK < %t.idxmask
// RUN: triton-opt %S/Inputs/nsa_fwd_masked_index_load.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.idxmask.err
// RUN: diff %t.empty %t.idxmask.err

// IDXMASK-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// IDXMASK: tt.load %{{.*}}, %{{.*}} : !tt.ptr<i64>
// IDXMASK: scf.if

// ---------- (k) a second scf.if in the gather loop ------------------------
// The rewrite replaces the single guarded region with a per-token weight, so a
// body with two conditions has no expression in the scan form (parseBlockGuard
// requires exactly one scf.if at loop-body top level).
// RUN: triton-opt %S/Inputs/nsa_fwd_two_guards.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.twoguards.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_two_guards.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.twoguards
// RUN: diff %t.twoguards.ctl %t.twoguards
// RUN: FileCheck %s --check-prefix=TWOGUARDS < %t.twoguards
// RUN: triton-opt %S/Inputs/nsa_fwd_two_guards.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.twoguards.err
// RUN: diff %t.empty %t.twoguards.err

// The dead second guard is folded by canonicalize, so assert the shape the rule
// refused to rewrite: the gather loop with its scalar table read and the one
// guarded region.
// TWOGUARDS-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// TWOGUARDS: scf.for
// TWOGUARDS: tt.load %{{.*}} : !tt.ptr<i64>
// TWOGUARDS: scf.if

// ---------- (l) the index table is read flat (no row base) ----------------
// The rewrite re-creates `root + rowOffset` outside the loop and indexes that
// vector by the iteration number. addptr(root, iv) has no row offset, so
// table[0, S) is not a row of the table and parseTableAddress refuses.
// RUN: triton-opt %S/Inputs/nsa_fwd_flat_index_table.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.flat.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_flat_index_table.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.flat
// RUN: diff %t.flat.ctl %t.flat
// RUN: FileCheck %s --check-prefix=FLAT < %t.flat
// RUN: triton-opt %S/Inputs/nsa_fwd_flat_index_table.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.flat.err
// RUN: diff %t.empty %t.flat.err

// FLAT-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// FLAT: scf.for
// FLAT: tt.addptr %arg6, %{{.*}} : !tt.ptr<i64>
// FLAT: scf.if

// ---------- (m) the row base depends on two axis-0 program ids ------------
// One query position has to be chosen, because the same SSA value is required
// by the causal predicates and the tensor mask; two reaching program ids make
// that choice ambiguous and the matcher refuses to pick.
// RUN: triton-opt %S/Inputs/nsa_fwd_ambiguous_pid.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.twopid.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_ambiguous_pid.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.twopid
// RUN: diff %t.twopid.ctl %t.twopid
// RUN: FileCheck %s --check-prefix=TWOPID < %t.twopid
// RUN: triton-opt %S/Inputs/nsa_fwd_ambiguous_pid.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.twopid.err
// RUN: diff %t.empty %t.twopid.err

// CSE merges the two identical program-id ops, so assert the shape the rule
// refused to rewrite instead of the duplicated op.
// TWOPID-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// TWOPID: tt.get_program_id x : i32
// TWOPID: scf.for
// TWOPID: tt.load %{{.*}} : !tt.ptr<i64>

// ---------- (n) the forward loop body contains a tt.trans -----------------
// tt.trans belongs to the additive template's vocabulary (it replays the order
// verbatim); the online-softmax template has no transpose step, so a body that
// carries one cannot be rebuilt and checkLoopVocabulary refuses it.
// RUN: triton-opt %S/Inputs/nsa_fwd_fwd_trans.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.fwdtrans.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_fwd_trans.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.fwdtrans
// RUN: diff %t.fwdtrans.ctl %t.fwdtrans
// RUN: FileCheck %s --check-prefix=FWDTRANS < %t.fwdtrans
// RUN: triton-opt %S/Inputs/nsa_fwd_fwd_trans.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.fwdtrans.err
// RUN: diff %t.empty %t.fwdtrans.err

// The dead transpose is folded away, so assert the shape the rule refused.
// FWDTRANS-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// FWDTRANS: scf.for
// FWDTRANS: tt.load %{{.*}} : !tt.ptr<i64>
// FWDTRANS: scf.if

// ---------- (o) the addptr tile pads with a non-zero "other" --------------
// The rewriter re-emits the tile load with other = 0 (the baseline's padding),
// so a non-zero padding cannot be reproduced (hasLoadShape: MaskedZeroOther).
// RUN: triton-opt %S/Inputs/nsa_bwd_nonzero_other.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.other.ctl
// RUN: triton-opt %S/Inputs/nsa_bwd_nonzero_other.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.other
// RUN: diff %t.other.ctl %t.other
// RUN: FileCheck %s --check-prefix=OTHER < %t.other
// RUN: triton-opt %S/Inputs/nsa_bwd_nonzero_other.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.other.err
// RUN: diff %t.empty %t.other.err

// OTHER-LABEL: tt.func public @parallel_nsa_bwd_kernel_dq(
// OTHER: scf.for
// OTHER: scf.if

// ---------- (p) the table row is wider than the 4096-slot ceiling ---------
// S drives the width of the vector read emitted before the loop, so an unbounded
// S would materialize an arbitrarily wide read; the i_h*S addend and the trip
// count are raised together so the cap is what refuses.
// RUN: triton-opt %S/Inputs/nsa_fwd_big_s.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.bigs.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_big_s.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.bigs
// RUN: diff %t.bigs.ctl %t.bigs
// RUN: FileCheck %s --check-prefix=BIGS < %t.bigs
// RUN: triton-opt %S/Inputs/nsa_fwd_big_s.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.bigs.err
// RUN: diff %t.empty %t.bigs.err

// BIGS-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// BIGS: scf.for
// BIGS: tt.load %{{.*}} : !tt.ptr<i64>

// ---------- (q) the constant trip count disagrees with S ------------------
// A constant trip count is the special case NS == S, and the matcher cross-checks
// the two derivations of the table width (checkCrossStageAgreements).
// RUN: triton-opt %S/Inputs/nsa_fwd_trip_mismatch.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.trip.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_trip_mismatch.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.trip
// RUN: diff %t.trip.ctl %t.trip
// RUN: FileCheck %s --check-prefix=TRIP < %t.trip
// RUN: triton-opt %S/Inputs/nsa_fwd_trip_mismatch.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.trip.err
// RUN: diff %t.empty %t.trip.err

// TRIP-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// TRIP: scf.for
// TRIP: tt.load %{{.*}} : !tt.ptr<i64>

// ---------- (t) the tile-max reduce combines with min, not max ------------
// The recurrence is recognised by its combine ops: the running maximum reduces
// with maxnumf, the normaliser with addf. A different combiner is a different
// recurrence even when every shape matches.
// RUN: triton-opt %S/Inputs/nsa_fwd_min_combiner.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.minc.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_min_combiner.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.minc
// RUN: diff %t.minc.ctl %t.minc
// RUN: FileCheck %s --check-prefix=MINC < %t.minc
// RUN: triton-opt %S/Inputs/nsa_fwd_min_combiner.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.minc.err
// RUN: diff %t.empty %t.minc.err

// MINC-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// MINC: "tt.reduce"
// MINC: arith.minnumf
