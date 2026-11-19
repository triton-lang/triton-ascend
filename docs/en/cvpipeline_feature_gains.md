# CVPipeline feature gains and removal candidates

Date: 2026-11-12. Scope: the attention implementation retained on `ebraheema/flashAtt/0.2.cleanup`. This report separates historical measurements from interim results of the requested fresh timing-only feature-ablation sweep. The full eleven-case report remains incomplete. No retained optimization is removed for performance in this step.

The strongest individual historical gains are output forwarding, two reduction chains, scalar-input forwarding and FMA. Direct O forwarding has a smaller independently repeated benefit, about 0.9–1.0%. Several other features have no isolated measurement. Missing attribution is not evidence of zero benefit.

## How to read the numbers

Gain means time reduction, `100 * (control_ms - candidate_ms) / control_ms`. Positive is faster; negative is slower. Each comparison uses its own same-device, same-cohort control. Do not add these percentages or compare absolute milliseconds across cohorts. The effects depend on the parent implementation, shape and interacting features.

The historical compiler measurements below use the original attention Python, B060, BS128/H8/BM128, FP16 BNSD, scale0.5, noncausal. HD64 uses BN256 and HD128 uses BN128 unless stated. The statistic is the median of six block means, ten actual hardware tasks per block, with balanced arm order and mapping blocks excluded. Reports retain numerical/stress qualification and sharing limitations. Paired wins are not a statistical significance test.

The locally accessible evidence for most individual contrasts is a preserved scientific-result record, including original evidence hashes. The original per-feature CSV packets are not in the transferred full-eleven timing packet. These are reported historical measurements, not a new independent re-audit of those CSVs. The current BZ full-eleven and cleanup comparisons have their own raw evidence, but compare complete recipes rather than individual features.

## Fresh timing-only sweep

The user selected fresh feature-by-feature timings wherever an existing switch permits them, explicitly excluding correctness and stress testing. The prepared matrix covers all eleven original production shapes with the canonical common policy and unchanged tiling. It contains seventeen contrasts: fourteen single-switch contrasts (some on a shared parent with dependent features disabled), a cache/readiness protocol group, a key-major/vector group and a complete attention-group comparison.

The screen uses six balanced forward/reverse rounds, ten active hardware tasks per block and the existing profiler schedule. Compile-only eligibility preserves actual unsupported-combination diagnostics; it does not run a numerical test. Effective options, native hashes, inputs, raw CSVs and sharing/admission observations are retained. Identical native images share a timing series and are labelled identical, rather than presenting timing noise as a feature effect. There is no AscendC/FP32 reference or 64-forward stress phase.

The CSV selector has been corrected to the observed `AI_CORE` aggregate. Five cases have completed and been independently audited; the remaining cases are pending. Partial cohorts stopped by resource admission guards remain preserved and are not pooled with replacement cohorts. Its sealed plan and private driver are under `/home/ebraheema/bz-workflow/fa-02-cleanup-cde9-20261111/feature-ablation-20261112/`. New timing-only variants will be labelled as such; historical correctness qualification will not be claimed as a new test of them.

Access status: server113 SSH began returning connection refused after the last successful observation at 13:35:04 UTC on November 12. At that observation, N2048/HD128 and N16384/HD64 were still running, with the other four uncompleted cases queued. Their current state is unknown. The five results below remain the last independently verified set; the full CSV/native/sharing/lifecycle audit and final report are pending restored access. Reservations and all evidence remain retained.

## Interim fresh result: N16384 / HD128

**Scope: one of eleven cases. The full matrix is still running.** This case completed all six balanced rounds on NPU2. An independent interim audit re-read 84 raw CSV blocks and 18 native payloads, checked exact typed options and recomputed the reported gains. There was no correctness or stress phase.

| Feature / group | On ms | Off ms | Enabled time reduction | Paired wins | Parent / qualification |
| --- | ---: | ---: | ---: | ---: | --- |
| maximum scaling | 582.763532 | 596.191327 | +2.252% | 6/6 | Common parent; only the stated switch differs. |
| two versus one reduction chain | 582.763532 | 615.496792 | +5.318% | 6/6 | Common parent; only the stated switch differs. |
| direct persistent O update / copyback removal | 582.763532 | 645.218409 | +9.680% | 6/6 | Common parent; only the stated switch differs. |
| scalar m/l input forwarding | 582.763532 | 623.102265 | +6.474% | 6/6 | Common parent; only the stated switch differs. |
| FMA | 582.763532 | 591.142907 | +1.417% | 6/6 | Common parent; only the stated switch differs. |
| Cube overlap | 582.763532 | 582.763532 | 0: identical native | — | One shared timing series, not two independent measurements.  |
| derived Cube drain | 582.763532 | 582.763532 | 0: identical native | — | One shared timing series, not two independent measurements.  |
| O-state/input forwarding | — | — | Unavailable | — | state-input-off |
| final FP16 output forwarding | 645.218409 | 741.618327 | +12.999% | 6/6 | Conditional parent: direct-output forwarding off in both arms. |
| VF operand substitution | — | — | Unavailable | — | vf-substitution-off |
| max/P/state producer fusion | 648.548456 | 722.757526 | +10.267% | 6/6 | Conditional parent: one chain and scalar-input forwarding disabled in both arms; compile eligibility decides shape support. |
| dense P packing | 615.496792 | 699.428605 | +12.000% | 6/6 | Conditional parent: one reduction chain in both arms. Native compile decides shape support. |
| score rematerialization | — | — | Unavailable | — | score-rematerialization-off |
| invariant cache and required banked readiness | 582.763532 | 582.763532 | 0: identical native | — | One shared timing series, not two independent measurements. Two-option protocol group: unbanked caching was removed, so a cache-only contrast is unavailable. |
| continuous query streaming | — | — | Unavailable | — | query-streaming-on |
| key-major layout and dependent vector transforms | 660.788166 | 608.379731 | -8.614% | 0/6 | Dependent vector group on a query-disabled parent; individual layout-only native form may be unsupported. |
| complete attention optimization group | 582.763532 | 608.379731 | +4.211% | 6/6 | Contextual total; same FMA/VF and host route in both arms; not individual attribution. |

The rematerialization-off configuration reports UB demand of 276 KiB against 248 KiB available in this case; the declared N32768/HD64 off configuration needs 308 KiB. Rematerialization must therefore remain enabled for the current recipe and tiling. Disabling it would require another change that reduces memory use, followed by fresh timing; this is a memory-feasibility dependency even though an isolated speedup cannot be measured here. The other unavailable HD128 modes report the closed key-major tensor VF-pair/protected-scope requirement. These are refusals under the declared parents, not zero gains or proof that every possible parent fails.

In particular, the fresh direct-O result is 9.680% on this HD128 parent. The older approximately 0.9–1.0% HD64 evidence below does not quantify this parent. Removal priorities must wait for the full fresh matrix.

Interim evidence: `/home/ebraheema/bz-workflow/fa-02-cleanup-cde9-20261111/feature-ablation-20261112/parent-interim-n16384-hd128-audit.json`. No completed case is selected or discarded based on whether its gain is favorable.

## Second interim fresh result: N8192 / HD128

**Scope: two of eleven cases are now complete.** The second case also passed an independent interim audit of 84 raw timing blocks and 18 native payloads. Both runs used six balanced rounds and no correctness/stress phase.

| Feature, with the common parent | N16384/HD128 gain | N8192/HD128 gain |
| --- | ---: | ---: |
| Two reduction chains | +5.318% | +6.119% |
| Direct O forwarding | +9.680% | +9.680% |
| Scalar-input forwarding | +6.474% | +6.684% |
| FMA | +1.417% | +0.845% |

All these on/off comparisons won six of six paired blocks in their own case. Cache/Cube controls again produced identical native code in N8192/HD128. These are per-case effects, not additive gains or full-matrix removal recommendations.

Interim evidence: `feature-ablation-20261112/parent-interim-n8192-hd128-audit.json` under the experiment root.

## Third interim fresh result: N8192 / HD64

**Scope: three of eleven cases are now complete and independently audited.** This HD64 case completed all six rounds on NPU2; the interim audit re-read 126 raw CSV blocks and 21 native payloads. No correctness or stress phase was run.

| Feature / group | Enabled time reduction | Paired wins | Parent / qualification |
| --- | ---: | ---: | --- |
| maximum scaling | -0.880% | 0/6 | Common parent; only the stated switch differs. |
| two versus one reduction chain | +7.499% | 6/6 | Common parent; only the stated switch differs. |
| direct persistent O update / copyback removal | +3.062% | 6/6 | Common parent; only the stated switch differs. |
| scalar m/l input forwarding | +2.267% | 6/6 | Common parent; only the stated switch differs. |
| FMA | -0.562% | 0/6 | Common parent; only the stated switch differs. |
| Cube overlap | +13.166% | 6/6 | Common parent; only the stated switch differs. |
| derived Cube drain | +0.064% | 3/6 | Common parent; only the stated switch differs. |
| O-state/input forwarding | +8.899% | 6/6 | Conditional parent: direct-output forwarding off in both arms. |
| final FP16 output forwarding | +5.395% | 6/6 | Conditional parent: direct-output forwarding off in both arms. |
| VF operand substitution | +13.661% | 6/6 | Conditional parent: dependent direct-output and scalar-input forwarding off in both arms. |
| max/P/state producer fusion | +11.175% | 6/6 | Conditional parent: one chain and scalar-input forwarding disabled in both arms; compile eligibility decides shape support. |
| dense P packing | +11.606% | 6/6 | Conditional parent: one reduction chain in both arms. Native compile decides shape support. |
| score rematerialization | Unavailable | — | score-rematerialization-off |
| invariant cache and required banked readiness | +0.150% | 6/6 | Two-option protocol group: unbanked caching was removed, so a cache-only contrast is unavailable. |
| continuous query streaming | -10.329% | 0/6 | Conditional parent disables all dependent forwarding/cache/Cube extensions in both arms. |
| key-major layout and dependent vector transforms | -0.399% | 0/6 | Dependent vector group on a query-disabled parent; individual layout-only native form may be unsupported. |
| complete attention optimization group | +11.906% | 6/6 | Contextual total; same FMA/VF and host route in both arms; not individual attribution. |

FMA and max scaling regress this HD64 case while improving the two completed HD128 cases. Derived drain is small and mixed across paired blocks here. These are candidates for later review, not current deletion decisions.

The cache/readiness group is only +0.150% with Cube overlap and drain held off in both arms. Cache/readiness are also prerequisites for the separately measured overlap path, whose gain is +13.166%; the small direct group gain does not establish that those prerequisites can be removed.

The query switch alone regresses on its deliberately unforwarded/cache-off parent. The predeclared query-family group—common versus query-off, including dependent forwarding/cache/Cube controls—improves this case by +12.256% and wins six of six pairs. Corresponding audited derived groups are +11.808% for N16384/HD128 and +5.534% for N8192/HD128. These are group effects, not isolated query-pipeline gains.

Interim raw-audit evidence: `feature-ablation-20261112/parent-interim-n8192-hd64-audit.json`; derived calculation: `parent-interim-query-family-derived.json`. Full-matrix results remain pending.

## Fourth interim fresh result: N32768 / HD64

**Scope: four of eleven cases are complete and independently audited.** The largest case completed all six rounds on NPU6. Its interim audit re-read 126 raw CSV blocks and checked 21 native payloads and their typed options. No correctness or stress phase was run.

| Feature / group | Enabled time reduction | Paired wins | Parent / qualification |
| --- | ---: | ---: | --- |
| maximum scaling | +1.824% | 6/6 | Common parent. |
| two versus one reduction chain | +5.654% | 6/6 | Common parent. |
| direct persistent O update / copyback removal | +6.824% | 6/6 | Common parent. |
| scalar m/l input forwarding | +2.275% | 6/6 | Common parent. |
| FMA | +0.071% | 3/6 | Common parent; small and mixed. |
| Cube overlap | +6.364% | 6/6 | Common parent. |
| derived Cube drain | +0.241% | 4/6 | Common parent; small and mixed. |
| O-state/input forwarding | +3.279% | 6/6 | Direct-output forwarding off in both configurations. |
| final FP16 output forwarding | +0.788% | 5/6 | Direct-output forwarding off in both configurations. |
| VF operand substitution | +9.490% | 6/6 | Direct-output and scalar-input forwarding off in both configurations. |
| max/P/state producer fusion | +8.731% | 6/6 | One chain and scalar-input forwarding off in both configurations. |
| dense P packing | +8.270% | 6/6 | One reduction chain in both configurations. |
| score rematerialization | Unavailable | — | The declared off configuration needs 308 KiB UB; 248 KiB is available. |
| invariant cache and required banked readiness | +0.405% | 6/6 | Cube overlap and drain off in both configurations; required protocol group. |
| continuous query streaming | -11.674% | 0/6 | Dependent forwarding/cache/Cube extensions off in both configurations. |
| key-major layout and dependent vector transforms | +1.698% | 6/6 | Query-disabled parent; dependent vector group. |
| complete attention optimization group | +7.802% | 6/6 | Contextual group total; same FMA/VF and host route. |

The predeclared query-family group improves this case by +6.210%, with six of six paired wins. Its query-switch-only regression therefore does not establish a loss for the complete streaming family. FMA and derived drain again show small effects here; their individual paired results do not establish a repeatable benefit at this shape. Maximum scaling improves this case but regressed N8192/HD64, so a global removal decision remains premature.

Interim evidence: `feature-ablation-20261112/parent-interim-n32768-hd64-audit.json` and `parent-interim-query-family-derived-02.json`. The remaining seven cases are pending; no partial case is combined with a replacement cohort.

## Fifth interim fresh result: N4096 / HD64

**Scope: five of eleven cases are complete and independently audited.** This case completed all six rounds on NPU1; the interim audit re-read 126 raw CSV blocks and checked 21 native payloads and their typed options. Admission showed one pre-existing workload. Full sharing review remains pending. No correctness or stress phase was run.

| Feature / group | Enabled time reduction | Paired wins | Parent / qualification |
| --- | ---: | ---: | --- |
| maximum scaling | -2.866% | 0/6 | Common parent. |
| two versus one reduction chain | +8.140% | 6/6 | Common parent. |
| direct persistent O update / copyback removal | +1.341% | 5/6 | Common parent. |
| scalar m/l input forwarding | +0.731% | 5/6 | Common parent. |
| FMA | -2.077% | 0/6 | Common parent. |
| Cube overlap | +13.511% | 6/6 | Common parent. |
| derived Cube drain | -0.794% | 2/6 | Common parent. |
| O-state/input forwarding | +10.503% | 6/6 | Conditional parent: direct-output forwarding off in both arms. |
| final FP16 output forwarding | +6.631% | 6/6 | Conditional parent: direct-output forwarding off in both arms. |
| VF operand substitution | +16.390% | 6/6 | Conditional parent: dependent direct-output and scalar-input forwarding off in both arms. |
| max/P/state producer fusion | +12.546% | 6/6 | Conditional parent: one chain and scalar-input forwarding disabled in both arms; compile eligibility decides shape support. |
| dense P packing | +12.801% | 6/6 | Conditional parent: one reduction chain in both arms. Native compile decides shape support. |
| score rematerialization | Unavailable | — | Declared off configuration fails compilation. |
| invariant cache and required banked readiness | -0.902% | 1/6 | Cube overlap and drain off in both configurations; required protocol group. |
| continuous query streaming | -8.717% | 0/6 | Conditional parent disables all dependent forwarding/cache/Cube extensions in both arms. |
| key-major layout and dependent vector transforms | -0.975% | 0/6 | Dependent vector group on a query-disabled parent; individual layout-only native form may be unsupported. |
| complete attention optimization group | +13.867% | 6/6 | Contextual total; same FMA/VF and host route in both arms; not individual attribution. |

Maximum scaling and FMA regress this case by 2.866% and 2.077%, respectively, and each loses all six pairs. Cube overlap and two chains retain material gains. Direct-O and scalar forwarding have smaller effects here than in the completed large cases. These are timing-screen observations; retained-feature removal and any required independent repeats remain later work.

The query-family group improves this case by +14.699%, with six of six paired wins. The negative cache/readiness contrast holds overlap and drain off in both configurations, so it does not establish that the protocol required by the +13.511% overlap path can be removed.

Interim evidence: `feature-ablation-20261112/parent-interim-n4096-hd64-audit.json` and `parent-interim-query-family-derived-03.json`. The six remaining cases are pending.

## Historical gains for retained features

| Current feature/control | Best attributable saved comparison | Assessment for later removal |
| --- | --- | --- |
| `enable_cvp_query_state_copy_forward` — readonly O input/state snapshot forwarding | N16384/HD64: 512.029839 → 493.713699 ms, **3.577%**; 6/6 wins [S34] | Material benefit on the measured parent. Retain pending a current-parent ablation. |
| `enable_cvp_query_final_output_forward` — final FP16 output forwarding | Added to input forwarding: 493.713699 → 442.435417 ms, **10.386%**. Both together improve 13.592% versus 512.029839 ms; 6/6 combined wins [S34] | Material conditional benefit. The 13.592% is the pair's total, not this feature alone. |
| `cvp_attention_reduction_chains=2` | N16384/HD64: 444.109804 → 419.818737 ms, **5.470%**, 6/6 wins; dense P/fused producer and existing output forwarding fixed [S35] | Material benefit on the measured parent. This is two chains, not removed Rows2 scheduling or four chains. |
| `enable_cvp_query_output_copy_forward` — direct persistent O update/copyback removal | N16384/HD64: 414.543815 → 410.431404 ms, **0.992%**; independent confirmation 414.739029 → 410.934985 ms, **0.917%**; both 6/6 wins [S37] | Small, repeatable historical benefit. Good complexity-versus-gain review candidate. This includes changed stock GSS/output placement, not just a copy instruction. |
| `enable_cvp_query_scalar_input_forward` — six readonly m/l input snapshots | N16384/HD64: 407.065573 → 397.225444 ms, **2.417%**. All six matched HD64 medians improve; separate focused confirmation recorded [S41] | Useful measured effect; the record does not provide an isolated gain for every HD128 shape. |
| `enable_fused_multiply_add` | N16384/HD128, otherwise identical fused bundle: 555.582673 → 548.908928 ms, **1.201%** [S52]. Ordinary small-HD128 FMA-only discovery: **0.445%, 0.337%, 0.878%** at N4096/2048/1024 [S54] | Modest, shape-dependent. The isolated FMA discovery has no independent FMA-only repeat; later confirmation changed the whole bundle. |
| `enable_vf_operand_substitution` | Original N16384/HD128: 575.171161 → 575.280663 ms, **−0.019%**, 3/6 wins [S51] | No noticeable isolated timing benefit in that control. It also enables memory feasibility and forwarding; it cannot safely be removed solely from this timing. |
| `enable_cvp_attention_score_rematerialization` | No isolated result for the retained key-major implementation. Older ordinary-layout remat + max-scale discovery: 2.049373 → 2.045649 ms, **0.182%**, 4/6 wins at N1024/HD128 [S55] | Unknown current contribution. The old ordinary-layout implementation was removed; its small gain does not quantify the retained path. |
| `enable_cvp_attention_max_scaling` | Measured with rematerialization in the preceding row; no separate isolated estimate [S55] | Unknown current contribution. Treat dependencies and numerical permission explicitly. |
| `enable_cvp_attention_key_major` | No isolated result for the current fused dense/two-chain implementation. Older key-major/P-state group lost **7.064%** versus its ordinary control at N1024/HD128 [S55] | That older group used a different fusion/chain policy, now removed. It does not prove today's layout is harmful. |
| `enable_cvp_attention_dense_p` | No isolated estimate. Changed together with max/P/state fusion, two chains, scalar forwarding and FMA in the HD128 bundle [S53] | Unknown individual contribution. Needed by the retained two-chain route. |
| `cvp_attention_producer_fusion="max-p-state"` | No isolated estimate. The HD128 fused bundle improves original by **5.017% / 3.190%** at N16384/8192, but regresses **4.598% / 4.745% / 3.362%** at N4096/2048/1024 [S53] | Useful large-shape bundle; smaller shapes deserve a legal grouped ablation. Do not credit or blame fusion alone. |
| `enable_cvp_query_pipeline` — across-query streaming | No isolated gain on the current common parent. Older N1024/HD128 V8: query-on 2.819556 versus query-off 2.107482 ms, a **25.255%** reduction by turning it off [S46]. Older N16384/HD128 query/storage group adds **2.171%** time [S51] | High-priority current-parent ablation, especially small HD128. Old regressions involve different math/storage policies and do not establish a current universal loss. |
| `cvp_invariant_cache_mode="auto"` — invariant Q cache | No isolated cache-on/off hardware gain in accessible records. Cache is used within combined HD64 recipes; HD128 uses the safe resource fallback [S64, BZ10] | Measure cache on/off before removal. A requested `auto` is not proof that caching applied on a particular shape. No qualified HD128 cache-on gain is available. |
| `enable_cvp_cube_banked_ready` | No isolated estimate. It supplies the readiness protocol for cached overlap/drain | Protocol dependency; do not delete required waits or banking to estimate performance. |
| `enable_cvp_cube_overlap` | Historical N16384/HD64 combined Cube events/overlap/tail change: 475.335323 → 433.027520 ms, **8.901%** [P] | Material bundle evidence, not an isolated current overlap gain. Measure it against the current parent with legal synchronization. |
| `enable_cvp_cube_derived_drain` | Included in that **8.901%** combined change; no independent drain-only/tail-only estimate [P] | Unknown separate benefit. Do not attribute all 8.901% to trimming drain iterations. |
| `enable_cvp_attention_analysis` | Recognition/validation foundation; no standalone speedup established | Judge by required analysis and safety, not an isolated runtime gain. |

FMA-only small-shape percentages are recomputed from S54's discovery controls: 28.725321→28.597562, 7.511091→7.485797, and 2.049298→2.031306 ms. Independent confirmations in S54 validate FMA plus boundary/state changes together, not FMA alone. The old boundary/state implementations are already removed.

The earlier input-forwarding diagnostic showing 0.177% was rejected after an NPU health alarm [S18]. It is excluded from accepted performance evidence; S34 supplies the later qualified comparison.

## Required policy and correctness dependencies

`allow_cvp_attention_sum_reassociation` permits the selected reduction tree; it is not a separate optimization with an additive gain. `disable_tightly_coupled_buffer_reuse=True` repaired actual incorrect outputs on the query path. Query state/output storage, alias/last-reader checks, transactional replacement, backup restoration and stock synchronization/planning are correctness dependencies, not optional speedups to remove when their direct gain is unmeasured.

`cvp_query_pipeline_mode="skew"` and `cvp_query_key_unroll=1` select the supported schedule; there is no current supported Unroll alternative to benchmark by changing them. Mixed CV, preload, multibuffering, automatic subblock binding, synchronization solver, explicit `unit_flag=False` and enabled block synchronization define the qualified host/compiler route. Their individual gains are not isolated in these attention experiments.

Changing VF substitution can invalidate output/scalar forwarding and, in older HD128 native cases, overflow stock UB capacity. Turning off the query pipeline also requires turning off dependent state/final/output/scalar forwarding and the invariant cache/Cube extensions. Such comparisons must be labelled as groups, with every effective option and applied rewrite recorded.

Score rematerialization is also an explicit compiler prerequisite for two-chain reduction, maximum fusion and maximum scaling (`CVPAttentionQueryVector.cpp`). The declared isolated rematerialization contrast disables those dependent features in both configurations first; its off configuration still exceeds UB capacity in the observed HD64/HD128 cases. Thus a direct toggle is unsupported in the common recipe, and the less demanding declared parent also has a measured compile-time memory refusal. Neither observation is a runtime gain estimate.

## Current common recipe: total effect, not feature attribution

The November 10 BZ run is closer to the current common policy than the older profile-selected Windows tables. Its same-run compiler-versus-baseline reductions were [BZ10]:

| N | HD64 time reduction | HD128 time reduction |
| ---: | ---: | ---: |
| 1024 | +18.154% | −1.921% |
| 2048 | +15.529% | −3.560% |
| 4096 | +13.704% | −3.007% |
| 8192 | +10.799% | +2.805% |
| 16384 | +8.899% | +4.606% |
| 32768 | +8.921% | Not in the eleven-case matrix |

The baseline has head-dimension variants; these totals cannot be assigned to one switch. Some cases had sampled sharing. The November 11 cleanup then preserved the latest kernels byte-for-byte and measured −0.158% to +0.313% against a fresh same-device reference. That is a cleanup-preservation check, not a measurement of any feature's benefit.

## Implementation boundaries for later removal

The feature switches do not correspond to independent files. A later removal should follow these shared implementation boundaries and its legal parent configurations; file size alone does not estimate removable code.

| Feature family | Main implementation under `bishengir/lib/Dialect/HIVM/` | Dependency to preserve or replace |
| --- | --- | --- |
| Score rematerialization, maximum scaling, producer fusion, dense P and reduction chains | `Transforms/regbase/CVPAttentionQueryVector.cpp` | Shared tensor matching and rewriting; the common two-chain route requires dense P, fused maximum/P/state, rematerialization and sum-reassociation permission. |
| Key-major layout | `Transforms/regbase/CVPAttentionLayout.cpp` | Supplies the layout expected by the retained query-vector transformations. |
| State, final-output, direct-O and scalar-input forwarding | `Transforms/regbase/CVPQueryStateCopyForward.cpp` | Four separate passes share query-storage, ownership and ordering contracts. Turning off one switch does not make the whole file unused. |
| Continuous query streaming | `Transforms/regbase/CVPQueryPipeline.cpp`, `CVPQuerySchedule.cpp`, `CVPQueryStateStorage.cpp`, `CVPQueryOutputStorage.cpp` | Forwarding and cache/Cube extensions depend on the streaming/storage schedule. |
| Invariant cache, banked readiness, Cube overlap and derived drain | `Transforms/regbase/CVPInvariantCache.cpp`; contract declarations in `IR/CVPInvariantCacheContract.h` | Cache fallback and complete readiness/credit/drain proofs are shared. An identical HD128 fallback image does not make the implementation unused for HD64. |
| Recognition and plan validation | `Analysis/CVPAttentionPlan.cpp` | Foundation for the retained transforms; no separate runtime gain is attributed. |

Paths in the table are relative to the AscendNPU-IR submodule; the cache contract header is under `bishengir/include/bishengir/Dialect/HIVM/`. FMA and VF substitution also affect kernels outside this attention integration, so their attention timings alone do not justify deleting their general implementations.

## Suggested later ablations

1. **Small HD128 query/vector bundle:** N1024/2048/4096 are the clearest current regressions. Compare legal query-off/math variants with the current common policy before changing global defaults. Keep HD64 and large HD128 visible.
2. **Direct O forwarding:** the isolated historical benefit is only about 0.9–1.0%, although confirmed. Measure the current parent with this one control off while retaining its required parent features.
3. **FMA and VF:** FMA has modest historical gains; measure it separately. VF's isolated large-HD128 control was neutral, but removing it requires a supported forwarding/memory configuration and full correctness checks.
4. **Cache/Cube group and remat/layout/fusion/dense components:** fill missing attribution with legal off/on contrasts. A group gain does not justify retaining every constituent, and an unknown constituent gain does not justify deletion.

For future discussion, use **less than 1%** as a candidate “small gain” label, not an automatic deletion threshold. The original plan requires an independent repeat for gains or margins below 2%. Report each shape, paired rounds and variability. Keep repeat cohorts separate; do not select the best repeat. Consider code complexity, dependency removal, correctness and target regressions together with time saved.

Retained optimization removal remains a later decision after the timing screen. Cache aliases and the five compatibility-only options have no distinct kernel optimization to credit: their obsolete spellings/disabled fields are removed in the canonical-only follow-up.

## Evidence map

All S entries refer to exact `result_key` records in the preserved [experiment/evidence index](/home/ebraheema/dev/windows-orders/no-stage6-handoff-win/documentation-20261108/EXPERIMENT_AND_EVIDENCE_INDEX.json), revision660. Selected original records, evidence hashes and the index SHA256 are copied to [feature-gain-source-excerpts.json](/home/ebraheema/bz-workflow/fa-02-cleanup-cde9-20261111/canonical-only-20261112/feature-gain-source-excerpts.json). The source's status reflects the historical experiment, not this cleanup goal.

| Label | Exact record or file |
| --- | --- |
| S18 | `without-stage6-input-copy-forwarding-health-rejected-v1` — rejected timing |
| S34 | `without-stage6-source236-focused-forwarding-six-round-timing-v1` |
| S35 | `without-stage6-source238-dense-two-chain-six-round-timing-v1` |
| S37 | `without-stage6-direct-o-two-independent-cohorts-v1` |
| S41 | `without-stage6-cvp-scalar-all11-b060-v1` |
| S46 | `source247-n1024-hd128-query-schedule-ablation-v1` |
| S51 | `source250-hd128-n16384-vf-math-query-causal-comparison-v1` |
| S52 | `source250-hd128-n16384-fused-vector-fma-discovery-v1` |
| S53 | `source250-hd128-fused-vector-independent-full5-validation-v1` |
| S54 | `source250-ordinary-cvp-boundary-fma-small3-independent-validation-v1` |
| S55 | `source250-n1024-hd128-vector-family-no-promoted-candidate-v1` |
| S64 | `source257v3-wheel19-full11-normal-jit-hardware-comparison-v1` |
| P | [Original integration design](/home/ebraheema/dev/windows-orders/no-stage6-handoff-win/docs/original-integration-plan/DESIGN.md), section2; [simple guide](/home/ebraheema/dev/windows-orders/no-stage6-handoff-win/docs/original-integration-plan/OPTIMIZATIONS_SIMPLE_GUIDE.md), Cube-tail measurement caveat |
| BZ10 | [November 10 paired full-eleven summary](/home/ebraheema/bz-workflow/fa-0.2-build-timing-20261110T0815Z/results-server113-timing05-final-20261110T1429Z/audit/SUMMARY.md) and its retained raw/audit files |
| BZ11 | [Cleanup preservation and timing record](cvpipeline_cleanup.md) |
