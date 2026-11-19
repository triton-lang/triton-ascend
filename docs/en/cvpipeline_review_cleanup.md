# Production review merged with attention cleanup

Date: 2026-11-12. Branch: `ebraheema/flashAtt/0.3.cleanup`.

This worktree starts from the validated cleanup and imports the committed
production review changes. The goal is to retain the common recipe's numerical
correctness and kernel timing. Fresh hardware measurements are complete for all
eleven production cases. Correctness passed and generated kernel binaries match
the cleanup reference for every case. The independent full-matrix evidence
audit passed; commit-to-build identities are recorded separately below.

## Source provenance

| Input | Triton parent | AscendNPU-IR |
| --- | --- | --- |
| Cleanup baseline | `67122fc8e16fcabf17d67c155f11d0fe2eda0c17` | `ad4a204994946ba03c72f9b3f9c0f639e9ecf18f` |
| Production review | `e3273bef38359b6146bf0ff6fa6bd5d9180be681` | `7d640fe3c1c171145456870d420e00c10f7e7d26` |
| Original integration used to isolate review changes | `c646a47a7f3d619c7ac495bd1bc2f8dc2696194e` | `b8f0c8ac0b9868aa29ad045e2407217fc4deb060` |

The resulting native commit is
`f659233f4412301da6458653eceb283d3462da15`. Its 5,243 source files match the
frozen build manifest exactly, including the seven changed C++/header files
reformatted before committing. The parent commit records the matching native
gitlink, frontend, migrated tests and this report together. Final source and
artifact identities are retained in `output/FINAL-SOURCE-BINDING.json` under the
private experiment directory below.

The review was imported as a delta against the original integration, rather than
as another complete integration commit. Both original worktrees remain source
inputs; the new worktree has an independent native checkout and private build,
package, cache and output directories.

The captured review commit remains the input even though the production native
checkout subsequently moved back to its parent-recorded commit. Its review
branch and the complete input patch remain preserved.

## Merge decisions

The Python frontend retains the review's table-driven native options and one
compiler-capability probe. The four query forwarding controls used by the
measured common recipe remain declared, typed, packed through the real JIT and
forwarded through that table. The canonical invariant-cache overlap control also
remains available. The implicit DCVP VF-substitution setting preserves its
existing safety fallback independently of explicit capability requests.

Cross-option attention legality is owned by the native frontend. Python checks
types, supported enum values, compiler availability, public target/SIMT routing
and query-buffer protection. Tests were migrated to those boundaries: native
dependency failures are exercised through the selected native compiler, while
Python tests check that general policy values and omitted values are preserved.
Obsolete internal Python helpers were not restored to satisfy old tests.

The native merge retains generic `rows`, `cols` and `reduction_dim` resource
geometry and the shared L1 matmul implementation. Cache-specific version tags
and the `_v1` cache-helper suffix are removed consistently from the retained
lowering and normal device library. Cached-B, co-resident storage, banked
readiness and complete resource/credit/drain checks remain the supported path.

Five review-edited fixtures belong to implementations already removed by the
cleanup and remain deleted. Their complete review patch is retained. Refactoring
of retired cached-A, standalone and neutral-cache branches is not used to restore
those implementations. Existing retained resource-plan assertions remain.

All seven obsolete attention compatibility names remain absent from active
option declarations and routing. Rows2, boundary specialization, half-P,
four-chain and query-Unroll implementations remain removed. Generic Stage 6
ownership/footprint features remain excluded.

## CPU and artifact validation

| Check | Current result |
| --- | --- |
| Merged Python source/toolchain checks | 398 passed; 28 existing skips |
| Actual installed-option and JIT packing checks | 143 passed |
| Standalone frontend boundary checks | 17 passed |
| Benchmark helper checks | 54 passed |
| Real native dependency integration | 32 cases included in the source suite; passed with the selected new native frontend |
| Native build and normal device libraries | Build exit 0; nine libraries produced |
| Native C++ unit suites | 100 passed |
| Focused attention lit | 96 passed |
| Full `check-bishengir` | Rerun exit 0: 938 passed; 140 unsupported; one expected failure; zero unexpected failures |
| Full `check-mlir` | Exit 1: 2287 passed; 606 unsupported; one expected failure; sole SCF failure reproduced with identical retained baseline output |
| Preliminary hardware probes | Both versions passed all ten small/extreme/odd-even probes; independent interim audit checked 104 comparisons and eight native payloads |
| Production hardware correctness and paired timing | All eleven cases completed; correctness passed; eight paired rounds per case independently checked |

The first full BishengIR run had 934 passes, 140 unsupported cases, one expected
failure and four failures. Three could not find `mlir-opt`; one mutation helper
refused a generated destination retained from the previous focused run. The
failed run is preserved. After building the missing tool and isolating test
outputs, the full rerun passed with the same 938/140/1 counts as the cleanup
baseline. No expectation or unsupported annotation was weakened to bypass the
initial failures.

The first full MLIR attempt stopped during the CPU async-runtime DSO link, before
tests executed: the B060 device wrapper interpreted `ALL` from the original GNU
`-exclude-libs,ALL` option as an input filename. The prior target-only host-linker
launcher is applied to this private build. It preserves the complete original
host-link arguments and symbol-hiding policy; LLVM and SDK sources remain
unchanged. The subsequent full suite completed with the same 2287/606/1/1 counts
as the cleanup baseline. Its sole failure remains
`Dialect/SCF/one-shot-bufferize.mlir::elide_copy_of_non_writing_scf_if`, at the
allocation/copy exclusions on lines 912 and 913. The new build-tree and installed
readers, and stock `mlir-opt`, produce the exact previously recorded failing
function body: SHA256
`72f7b2e26688e6fde32c3fba1bdfd2aed3f2dab1d0e5346c2d81d139ea822c41`,
with three allocations and one copy. The stock tool and original test-source
hashes also match the baseline. The failed full-suite exit remains recorded;
no LLVM test expectation or unsupported annotation was changed.

The installed candidate frontend hash is
`9e035e762c6c826285573b070c25fc99a57c15351b4a5c23ae1f143072671e2b`;
the reader hash is
`0f0d0dfd806e71baf56807c10983b0b1429a29bb38d6f468f3ce3b71b53a4fb7`.
These are installed files with their final RPATH, rather than the differently
hashed build-tree executables.

Independent normal-library inspection uses the matching B060 SDK reader.
`LocalMmad.cpp` belongs to the `aic_c310` build group, so
`meta_op.aic.c310.bc` defines the two retained no-v1 cache helpers. Other
host/vector/C220/mixed libraries have distinct roles. All nine selected
libraries were inspected; no retired v1 cache-helper definition remains.

## Hardware preservation contract

The first runtime preflight exited before kernel validation because the staged
Triton translator files had mode 0600. Their contents match the previously
qualified runtime wheel exactly. Fresh private runtime copies restored mode 0755
for `triton-opt` and `triton-mlir-opt`; the failed package, inputs and exit records
remain retained. This permission repair changes no compiler code or kernel
option and supplies no correctness or timing credit by itself.

The fresh second cohort uses the corrected private runtime copies. Both versions
completed all ten preliminary probes. Independent interim review checked their
104 comparison records, matching input fingerprints, exact common-option types,
selected frontend identities and eight native payload hashes. These preliminary
results supplement the completed production matrix below.

All eleven production cases passed an independent read-only audit of all 176
primary timing CSVs and 836 numerical/reuse comparisons. Each used eight paired
rounds. The difference is candidate/reference minus one; positive is slower:

| Case | Reference median (ms) | Candidate median (ms) | Difference | Native binaries |
| --- | ---: | ---: | ---: | --- |
| N32768/HD64 | 1643.625310 | 1644.223018 | +0.036365% | Identical |
| N16384/HD64 | 379.033774 | 379.013547 | -0.005336% | Identical |
| N16384/HD128 | 531.908808 | 532.047135 | +0.026006% | Identical |
| N8192/HD64 | 84.074163 | 84.091693 | +0.020851% | Identical |
| N8192/HD128 | 113.323203 | 113.334222 | +0.009724% | Identical |
| N4096/HD64 | 21.030983 | 21.029974 | -0.004797% | Identical |
| N4096/HD128 | 30.104091 | 30.167102 | +0.209310% | Identical |
| N2048/HD64 | 5.273780 | 5.273979 | +0.003788% | Identical |
| N2048/HD128 | 7.880545 | 7.876639 | -0.049566% | Identical |
| N1024/HD64 | 1.329474 | 1.329562 | +0.006630% | Identical |
| N1024/HD128 | 2.131935 | 2.131408 | -0.024715% | Identical |

Observed median differences range from -0.049566% to +0.209310%; no case reaches
the predeclared 1% slowdown threshold for a separate repeat. These measurements
support preservation of the measured kernel timing. They do not measure Python
compilation latency or prove performance for unmeasured configurations.

The complete per-case data is also recorded in
[the timing CSV](cvpipeline_review_cleanup_timings.csv).

The final runtime check compared every file in both deployed private stacks to
the trusted pretransfer manifest: 293 files, all nine normal device libraries
and four executable modes per version passed. Only the originally documented
translator mode repair was applied; no runtime payload changed.

The independent selected-evidence transfer verified all 2,725 files against a
source manifest recorded before transfer. Its SHA256 is
`934ad8ac538c45a20422e78f84cec5a947452c91924b922c01b1056d7cfa507e`.
The strict full-matrix auditor then reread all 176 primary timing blocks,
twenty-two native payloads, probe/checker records, numerical/reuse records,
metadata, input fingerprints and exact recipe/runtime identities. Its report
is `output/PRESERVATION-AUDIT-01/audit.json` in the private experiment directory.

Physical NPU0 had no visible workload in the healthy, complete prelaunch sample,
which preceded launch by 0.416 seconds. All 917 samples during the run were
healthy and complete, with at most two visible workloads and a maximum sample
gap of 6.517 seconds. The device was empty in the first healthy sample after the
run. These are observations on a shared device, not a global exclusivity claim.
Raw physical PIDs lack container labels; their association with the resident
pair is inferred from the empty-before/two-workers/empty-after observations.
Exact owned container worker identities and exits were checked separately.

The coordinator and both workers exited with code zero, the process group was
empty, and both cohort reservations were released. Private builds, runtime
copies, caches, raw evidence and the failed initial preflight remain retained.

Both private runtime stacks use the same canonical 27-option common recipe.
The control is the canonical cleanup version, rather than the older 31-option
compatibility version. Production shapes remain BS128/H8, FP16 BNSD, noncausal,
scale 0.5, BM128, with BN256 for HD64 and BN128 for HD128.

The existing preservation workflow checks full O/LSE against fresh AscendC for
seeds 12/0/42, selected FP32 CPU reference rows, unchanged input fingerprints,
small normal/zero/repeated/extreme inputs, odd/even key tails and repeated-forward
reuse behavior. Absolute and relative tolerances remain 0.005. Each case must
pass its correctness checks before its timing blocks.

Timing uses one excluded mapping block and eight balanced alternating paired
rounds, with ten active hardware tasks per block. The aggregate must identify
the exact `_attn_fwd` operation, `AI_CORE`, selected physical device and Count10.
Reference and candidate profile serially in independent private processes on
the same physical NPU. Final reporting must include all eleven cases and all
176 primary timing blocks, raw means, medians, paired ratios, code identities
and observed sharing.

New workers require BZ reservations and a fresh physical-host health/workload
snapshot. Partial case rounds are never pooled across retries. Completed cases
remain retained regardless of gain; independent repeats are reported separately.
Any unexplained systematic regression remains unresolved.

Experiment ledger: `bz-20261111T135448Z-a85445820bb1`.
Private records: `/home/ebraheema/bz-workflow/fa-03-review-cleanup-20261112`.
The complete source/commit binding is `output/FINAL-SOURCE-BINDING.json`; the
operational retention and final lifecycle records remain in the same ledger and
private experiment directory. The full MLIR suite's existing SCF failure remains
the validation limitation described above.
