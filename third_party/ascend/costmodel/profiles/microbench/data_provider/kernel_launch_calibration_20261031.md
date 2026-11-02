# Ascend950PR kernel launch calibration

Target: `Ascend950PR_9579`. TA base: `b695c198ad3782caea64e742a79d1b65f4154c8d`.
BiShengIR: `9756b309751b`, CANN 9.1. The calibration used profiler task
duration from exact saved binaries; compilation, correctness, warmup and active
measurement were separate phases.

The independently fitted launch envelope is:

```text
C_launch(domain, physical_programs)
  = base(domain)
  + (physical_programs - 1) * incremental(domain)
```

Complete-kernel time does not add that envelope to the Stage body. Simulator
request/retire logs and the physical AIC P28 grid-dot control show that parent
program distribution overlaps useful body work. Ordinary routes therefore use:

```text
D = (physical_programs - 1) * incremental(domain)
C_kernel = base(domain) + norm_p(D, C_stage_body)
           + C_completion + C_argument
p = 1.6
```

An `independent_pipelined_loop` whose completion is measured after the parent
distribution uses `D + C_stage_body` instead of the p-norm. Dynamic grid-stride
shells retain the overlapping form. This distinction is selected from Stage
schedule facts, not a kernel or operator name.

| Domain | Base (SYS_CNT cycles) | Incremental (cycles/program) | Base CV | Incremental CV |
|---|---:|---:|---:|---:|
| AIV | 537.9774739135498 | 14.307636363636364 | 2.07% | 0.57% |
| AIC | 584.3611462727271 | 112.0093621399177 | 2.87% | 1.17% |

AIV training consists of six independent 300-ABBA windows using one SIMD and
one pure-SIMT binary (14,400 target tasks). AIC training consists of three
independent 300-ABBA windows using one FP16 16x16x16 Cube binary (3,600 target
tasks). Every window has an exact binary SHA, one ordered stream, matching
`kernel_details.csv` and `task_time.csv`, numerical correctness, and the
expected physical block sequence.

Independent holdouts were not consumed by the fit:

| Holdout | Marginal error | Complete-kernel error |
|---|---:|---:|
| SIMD direct copy | -0.13% | P1 +9.81%, P28 +7.85%, P56 +4.57% |
| AIV SIMD→SIMT→SIMD | +0.46% | resource decomposition not used |
| Cube plus local SIMT | -4.42% | resource decomposition not used |

The prior store-only physical-grid curve was rejected as a generic launch
model. Instruction-level simulator logs show its tail is an actual scalar/UB
to MTE3/BWIF store completion chain. On hardware its P1→P28 growth is about
2.68 us, while SIMD arithmetic and direct-copy holdouts grow by about 0.37 us.
The old unconditional positive `has_global_load` residual was also rejected:
it must be zero for the direct-copy holdout. Its qualified remainder is retained
only as a completion reference for a nontrivial single-wave load path, dynamic
dot loop, or capped atomic completion. Runtime accounting subtracts the
effective launch contribution before applying that remainder, so it neither
duplicates launch nor applies to pure unit-stride copies.

The symbolic loop-carried dependency schedule has an independently measured
completion envelope whose interval overlaps launch and pipeline fill. That
path marks launch as included and uses its completion envelope instead of
adding `C_launch` again.

Fresh strict validation on `triton-ascend-costmodel-tests@15b71d2` reduced the
profiler-gated failures from 63/100 with the additive v155 accounting to 29/100
with the overlap-aware v161 accounting. All grid-stride dot shapes P56/112/224/
336/448 and all gather-dot controls passed. A follow-up v162 probe passed all
seven launch-boundary controls: masked copy, grid-stride gather, continuous and
discrete P56 atomics, sqrt, cosine, and non-unit-stride block-pointer load.
The four histogram cases were explicitly excluded from this calibration by
user direction; no histogram-specific formula or lowering change is present.
The mechanically filtered 96-case manifest has SHA256
`f37aa5469b62ee191693c9953dbe43f61705d0a353fa910cf3eacc223a292716`.

The subsequent v171 three-process rerun passed 16/18 historical non-histogram
failures. v172 corrected the bounded native indirect-store completion, and v173
passed both `softmax_original` and the structurally identical masked-softmax
control with three independent profiler processes. The final v177 candidate
passes all 96 non-histogram cases on healthy physical device 1 with three fresh
profiler processes per case. Mean absolute error is 3.45% and maximum absolute
error is 9.997%. Evidence is retained at
`/home/h00921337/kernel_launch_20261031/nonhist96_v177_m3_device1`.

Physical device 0 entered an HBMA/HWTS alarm state during final reruns and
produced bimodal timings for the same softmax binary. Those late measurements
are rejected. Healthy device 1 reproduced `softmax_power2` at 50.229--50.441 us
over five fresh processes. The prior large-working-set rule counted logical
tensor bytes and therefore ignored overlap between 256-wide rows at stride
150. v177 evaluates the active affine access footprint instead, removing an
incorrect 14.895 us full-physical completion. Prediction changes from 65.200 us
to 50.305 us; adjacent row-reduction and masked-softmax controls still pass.

The symbolic periodic dependency schedule applies an independently calibrated
initiation-interval efficiency of `0.8818552829255477`. Three original-shape
processes give 0.39% cross-process CV and 0.81% maximum training error. Fresh
v177 validation on healthy device 1 keeps one binary across K=32/64/96/128:
the original QK64/V64 geometry has 0.55% maximum complete-kernel error and the
independent QK32/V64 geometry has 1.29% maximum error.
Evidence archive `stage_dependency_efficiency_evidence.tar.gz` has SHA256
`d1c8e9a3eec2bc13ae075d5fb31b5f8df03773739dce8814f0ed1ec4f9ab30e4`.
Both geometries also pass a dynamic-CV-requested correctness smoke. Because
the selected route is all-SIMD, SSBUFFER returns 2 and the backend correctly
falls back to an effective disabled CV pipeline.

Raw evidence is retained under
`/home/h00921337/kernel_launch_20261031` on the calibration host. Archives:

- `b695_launch_domain_evidence.tar.gz`:
  `0ca051cd4bbed7ec30b5b5661c0a921486ef16e76061f58a1f6275ce2c803378`
- `b695_e2_joint_evidence.tar.gz`:
  `a9093fc4bf5431166c7d36c982ae30d8a43d11081d135c7c2bc5f37a3e5a6b49`
- structural simulator evidence `p1_all_evidence.tar.gz`:
  `115e7be967d33593993c2a52442d0eb4d7a1998a37ea2c932bb603269c09c1fd`

The 9589 simulator was used only to identify instruction/request ownership and
overlap. Its 32-AIC/64-AIV timing was not fitted to the 9579 hardware profile.
The production parameters above were collected from physical 9579 hardware;
final cross-operator and Flash Attention acceptance uses healthy physical
device 1. The later faulted-device-0 measurements are not part of the fit.
