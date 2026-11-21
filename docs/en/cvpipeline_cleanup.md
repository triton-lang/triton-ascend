# CVPipeline unused-code cleanup

Status: the November 11 cleanup passed native/hardware preservation. The November 12 follow-up removes attention compatibility options; its validation is recorded separately below. One unchanged baseline MLIR test failure remains documented.

Branch: `ebraheema/flashAtt/0.2.cleanup` in a new dedicated worktree and an independent AscendNPU-IR checkout. Starting revisions are Triton `0d518923a11945728703aeafc558414b4ee7d4d3` and AscendNPU-IR `b8f0c8ac0b9868aa29ad045e2407217fc4deb060`.

November 11 AscendNPU-IR cleanup commit: `449a6b3f9cfc5bc6eba9e40e920137a98d4256bf`. Its file contents match the validated source; the original worktrees remain preserved.

## Preservation target

Preserve the latest November 10 BZ common-switch **compiler** comparison, not the older Windows A/B/C selection. The measured November 10/11 recipe used the same 31 explicitly requested options and fixed host settings for all eleven cases, BM128, BN256 for HD64 and BN128 for HD128. The canonical-only follow-up uses 27 options after deleting four obsolete false controls. The sealed recipe SHA256 is `ffe620bc4bb60f6ad6f58e6b989751c27008691c23f042be1210124e3d0def37`.

The common policy retains key-major dense max/P/state, score rematerialization/max scaling, two reduction chains with explicit sum reassociation, Skew/U1 query streaming and all four forwarding controls, VF/FMA, protected query reuse, and automatic invariant caching with banked readiness/overlap/drain. Automatic resource refusal/fallback for HD128 is part of this behavior. No numerical permissions, tiling policy, scheduler, or optimization is added.

The current recipe explicitly requests `cvp_invariant_cache_mode="auto"`. Python's unspecified value is `None` and the native default is `none`. The historical `cvp_q_cache_mode` and `cvp_q_cache_consider_overlapping` spellings are removed, including alias normalization, JIT fields and mirrored metadata.

The latest common policy itself has some slower small-HD128 results than its historical per-case controls. This cleanup preserves that policy; it does not re-enable boundary optimizations or choose different flags by case to claim an improvement. Historical/prior controls require the retained original compiler artifacts.

## Removed implementation surface

| Removed surface | Reason it is outside the latest compiler policy |
| --- | --- |
| Rows2 pass | Explicitly false in every case; unrelated to the retained two reduction chains |
| First-key output/state and last-key output specialization | All three attention-boundary switches are explicitly false; normal query initialization, finalization, O/M stores and first/last predicates remain |
| Boundary-only forwarding branches | Specialized first/last helper signatures are no longer emitted; ordinary output forwarding keeps its full row/chunk, alias, ordering and final-reader proofs |
| Half-P streaming pass | Not requested; ordinary dense-P packing/publication remains |
| Ordinary-layout score rewrite | The selected score rewrites use the key-major QueryVector path |
| Four-chain implementation and public partial `p-state` mode | The selected producer is `max-p-state`, with two chains; its shared probability/state helpers remain necessary |
| Attention-specific Unroll emission and logical-lane APIs | Every selected request uses Skew/U1; ordinary upstream Unroll for other kernels remains |
| Neutral, cached-A and invariant-only cache variants | The selected cache is automatic B/co-resident caching; unsupported geometries retain safe fallback |
| Unbanked cache emission and its helper entries | The selected cache path requests banked readiness; unready cache IR remains an intermediate form and is refused if lowered prematurely |
| Dedicated obsolete fixtures/helpers | Their implementation is removed; retained resource, alias, generation, credit, drain and backup checks are preserved |

The November 12 follow-up deletes all five compatibility-only option fields and their native command-line switches: Rows2, first output, last output, first state and half-P streaming. They are absent from recipes and metadata; even False/None is no longer accepted by the public constructor or JIT. The generic unknown-keyword validation rejects old launch names.

Generic Stage 6 changes remain excluded. Stock planning/synchronization, transactional replacement, filtered-backup restoration, query storage protection, checked resource arithmetic and complete handoff/drain proofs remain dependencies of the selected implementation.

## Validation status

The November 11 benchmark manifest selected the common 31-switch policy. The current manifest selects its canonical 27-option equivalent, omitting four false fields and renaming the cache key. The ordered native feature arguments and protected reuse policy match the pre-cleanup source for every sealed case.

All eleven saved native `bcmlir` inputs were copied unchanged from the normal JIT archives, after its bytecode/reader roundtrip. With the exact B060 downstream tools and original library bitcode, all eleven cleanup kernel ELF files are byte-identical to the fresh reference files **and to the kernel hashes recorded in the latest timing run**. This comparison does not require an IR rewrite or a reader change.

| Validation | Observed result |
| --- | --- |
| C++ unit suites | 100 passed |
| Focused attention lit | 96 passed |
| Preload/pipeline/sync/backup regressions | 80 passed; 18 unsupported |
| Full `check-bishengir` | 938 passed; 140 unsupported; 1 expected failure; exit 0 |
| Standalone frontend contracts | 45 passed |
| Source-loaded frontend/toolchain pytest | 472 passed; 28 existing skips |
| Installed compiler-options pytest | 127 passed |
| Benchmark CPU tests | 54 passed |
| Full `check-mlir` | 2287 passed; 606 unsupported; 1 expected failure; 1 unchanged SCF bufferization failure reproduced on the original reference |
| Fresh hardware correctness/stress/timing | All eleven cases passed numerical/stress checks and completed eight paired timing rounds |

The MLIR CPU async-runtime test library needed a private build launcher that sends the original host link arguments directly to B060's own `ld.lld`, bypassing its device wrapper's unsupported option parsing. LLVM/SDK sources, symbol-hiding options, and deployed NPU compiler artifacts remain unchanged.

The remaining `SCF/one-shot-bufferize.mlir` FileCheck failure is pre-existing. The exact failing split section, `elide_copy_of_non_writing_scf_if`, produces an identical function on the original published reader, the cleanup reader, and the stock `mlir-opt` from the pinned LLVM tree. All three retain the same extra allocation/copy and fail the same checks. The full suite's exit remains nonzero; neither its expectations nor unsupported annotations were changed. The first whole-file comparison hit separate reader crashes before this section; those diagnostics are also retained.

The hardware comparison used separate verified reference/cleanup packages, identical input fingerprints and options, full O/LSE checks for seeds 12/0/42, FP32 reference rows, and 64-forward stress. Both versions also passed the small normal/zero/repeated/extreme and odd/even-tail probes. All numerical tolerances remained 0.005 absolute/relative. The first NPU5 attempt stopped when other workloads exceeded admission limits; its evidence was preserved. The NPU7 retry completed successfully, with both workers and the coordinator exiting 0 and their process group empty.

## Measured timing preservation

Reference means the published pre-cleanup compiler running the same common recipe. Values are milliseconds: each is the median of eight hardware-profiler block means, with ten active tasks per block and alternating reference/cleanup order. Mapping blocks are excluded. Positive change means cleanup is slower.

| Case | Reference ms | Cleanup ms | Change % |
| --- | ---: | ---: | ---: |
| n1024-hd64 | 1.328544 | 1.329140 | +0.045 |
| n1024-hd128 | 2.133034 | 2.131047 | -0.093 |
| n2048-hd64 | 5.264806 | 5.264662 | -0.003 |
| n2048-hd128 | 7.895191 | 7.882731 | -0.158 |
| n4096-hd64 | 21.002619 | 20.995749 | -0.033 |
| n4096-hd128 | 29.971733 | 30.065652 | +0.313 |
| n8192-hd64 | 85.746748 | 85.752969 | +0.007 |
| n8192-hd128 | 116.894137 | 116.943154 | +0.042 |
| n16384-hd64 | 419.393566 | 419.201318 | -0.046 |
| n16384-hd128 | 592.426523 | 592.538868 | +0.019 |
| n32768-hd64 | 1796.099138 | 1796.771853 | +0.037 |

The matched timing changes range from -0.158% to +0.313%. Every fresh current-JIT native kernel also matches the latest historical compiler hash. This table uses fresh same-device controls on NPU7; the November 10 absolute timings were measured on NPU5 under different run conditions.

The run was nonexclusive: additional resident PIDs appeared in 167 of 956 host observations during the accepted trial. Maximum observed workload count was three and sampled health stayed OK. The launch observation was 2.95 seconds old and showed no existing workload. These observations are snapshots, not a cross-user guarantee. Both BZ reservations were released after actual group quiescence. Monitor termination was observed after its stop marker; its wrapper did not capture an OS exit code.

The copied evidence manifest SHA256 is `b8c5ac6240449f5b906ab068880a758c85c5eb6959407983b84a4eca2247df2f`. Independent audit re-read 176 measured CSV blocks and all 22 native files, recomputed medians, and checked inputs, options, complete numerical/stress records and exit receipts. Evidence is under `results-server113-final-01/`; the parent audit is `audit-parent-02/`.

Experiment/evidence ID: `bz-20261111T135448Z-a85445820bb1`. Private run root: `/home/ebraheema/bz-workflow/fa-02-cleanup-cde9-20261111`. Original worktrees, historical timing data, build artifacts and retention decisions are preserved.

## Canonical-only follow-up (November 12)

AscendNPU-IR canonical-only commit: `ad4a204994946ba03c72f9b3f9c0f639e9ecf18f`.

The attention compatibility layer has been removed from Python and native option declarations. Existing general backend compatibility predating this integration is outside this follow-up. The current common recipe uses 27 explicit options, with `cvp_invariant_cache_mode="auto"`; tiling, arithmetic permissions, scheduling and optimization behavior are unchanged in intent.

The four obsolete explicitly false controls (Rows2, first output, last output and first state) and the omitted half-P control are deleted. There are no attention compatibility fields left to delete later. Query state/final/output/scalar forwarding remains enabled and implemented.

The canonical-only source passed the following checks:

| Follow-up validation | Result |
| --- | --- |
| Native build and C++ units | Build exit0; 100 passed |
| Focused attention / moved-helper wrappers | 96 passed / 2 passed |
| Full `check-bishengir` | 938 passed; 140 unsupported; 1 expected failure; exit0 |
| Standalone frontend contracts | 41 passed after removing obsolete compatibility tests |
| Source-loaded frontend/toolchain contracts | 398 passed; 28 existing skips |
| Installed compiler options and real JIT packing | 143 passed, including all seven removed names rejected for None/False/True |
| Benchmark CPU tests | 54 passed |
| Native replay of the eleven common cases | All33 emitted objects rehashed equal to the saved reference; all11 kernel payloads equal the latest measured hashes |
| Full `check-mlir` | 2287 passed; 606 unsupported; 1 expected failure; the same1baseline SCF failure |

The stock `mlir-opt` hash and the original SCF test source hash are unchanged from the prior exact-split reproduction. The sole failure is again the allocation/copy exclusions at lines912/913 of `elide_copy_of_non_writing_scf_if`. The failed suite exit remains recorded; no expectations or unsupported annotations changed.

Independent follow-up review is `canonical-only-20261112/parent-canonical-validation-review.json`, with source/test manifest `source-inputs-02.json`. Its only amendment to the generation-time manifest changes two test files. Runtime, C++ and recipe bytes are unchanged by that amendment. The benchmark phase passed in the original frontend job; its stderr is preserved in the durable outer log, while the two failed compatibility test suites were corrected and rerun separately.

The hardware table above belongs to the November 11 source and is not relabelled as a new run. The user requested fresh feature-by-feature timings with no correctness or stress phase. The profiler CSV selector was corrected to the observed `AI_CORE` aggregate, and five completed cases have independent interim audits. Server113 SSH access subsequently failed, leaving the remaining remote states unknown. See [feature gains and attribution limits](cvpipeline_feature_gains.md) for the separately labelled fresh results and remaining work.
