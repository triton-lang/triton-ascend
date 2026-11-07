# Existing indirect Load formulas

`OrdinaryLoadSelected.h` exports the existing TRAIN-fitted SIMD and SIMT Load
coefficients without refitting. `OrdinaryLoadMain.cpp` derives separable axes,
payload types, static rectangular masks and visible strides from registered
post-transform TTIR. It prices one matched target Load and retains the auxiliary
index/address/mask/output workload. Multiple eligible targets, nested scopes,
F>1, unsupported types and unproved geometry retain the existing model.

The selected path is enabled by default. Set the backend option
`ordinary_load_selected_enabled=False` to disable it. Host strides require the
actual entry-argument attributes and matching runtime-scalar option records.
Device strides stay opaque; the native caller must additionally declare their
positive/no-wrap domain before that path is used. Runtime/data-dependent masks
cannot reuse static K. An unknown tensor's contents are never read by the model.

Routing uses storage width and mask presence, then a visible prefix-mask fact
for SIMD or the number of structured axes for SIMT. Every leaf has at most eight
terms. The two frozen models have 60 and 64 coefficients in total; there is no
case-name or kernel-name lookup. Their source-file identities are preserved in
`profiles/indirect_load_selected/`.

## Actual results and remaining scope

Recomputing the existing frozen formulas on their recorded development sources
gave:

| Mode | Unmasked inputs / MAPE | Masked inputs / MAPE | Overall development MAPE |
| --- | --- | --- | --- |
| SIMD | 1160 / 14.8132% | 1224 / 17.7338% | 16.3127% |
| SIMT | 1456 / 19.3568% | 1504 / 14.9182% | 17.1015% |

These are development results, not newly collected final measurements. The
exported C++ arithmetic matched 9384 real metadata rows and 375360 feature
comparisons exactly; the maximum prediction difference was 2.91e-11.
The checked-in Python API independently reproduced all 9384 frozen predictions
with difference zero. The combined Load/Store Python regression suite passed
60 tests. A native build of the current cost-model sources and their real
OffsetAnalysis/Utils dependencies passed 104 positive IR checks, 416 guard
rejections, 104 target-only Stage compositions, 16 Host/Device binding checks
and four FP8 native/byte transport checks. Same-width pointer bitcasts preserve
logical FP8 provenance; SIMD native and SIMT byte transports are distinguished.
Those registered-IR unit fixtures are synthetic; they are not measured
performance evidence. The complete current-upstream Triton extension has not
been rebuilt.

The separately frozen sparse-GQA source panel has 54 configurations / 432
independent inputs and MAPE 17.7052% (SIMD 17.1964%, SIMT 18.2140%). That model
uses an explicit Host producer contract for a causal valid prefix. Its result
does not permit a generic Load estimator to inspect hidden runtime K; the
generic static-mask integration falls back on that runtime-mask case.

The newly evaluated generic final matrix contains 886 Load configurations,
7088 input identities and two sessions. This check **did not pass**. There are
164 nonpositive two-session median labels; the remaining 6924 positive-label
inputs have MAPE 70.2104%. The protocol diagnoses and retains nonpositive
differential labels without including them in MAPE; there is no APE-based
filtering. Repeat-quality diagnostics flag 386 configurations. The SIMD
subset has positive labels throughout and MAPE 30.6038%. No observations were
removed or replaced to declare acceptance. All retained quartet/window/label
reductions were independently reproduced, and 352368 original-feature
comparisons matched exactly. Existing-formula reuse therefore remains a
candidate; these results do not establish final Load accuracy.

An explicitly authorized post-observation re-split assigns 734 configurations
to TRAIN and 152 to holdout by a fixed geometry hash. Initial refits use only
the assigned TRAIN labels, with complete three-fold geometry validation and
scalar recomputation. Among the completed candidates, the best combined
two-session holdout MAPE was 42.0580% on 1187 positive inputs; 29 nonpositive
inputs are retained and reported separately. This is re-split validation of previously
observed data, not an unseen-test result. It also **does not pass** the overall
20% target. These unsuccessful candidates have not replaced the checked-in
coefficient profiles.

Four complete observation rounds have since been verified: 3544 files and
850560 retained quartets. Their four-round medians contain 173 nonpositive
labels, retained and diagnosed outside MAPE. Further TRAIN-only candidates,
including the permitted original-TRAIN anchor, improve the best combined
holdout MAPE to 33.6309%, still above 20%. Each model in that combination
completed the geometry folds and scalar check; holdout coefficients were not fitted.
Discrete formula-bank screening reached 23.2816%, but that composite has not
completed its own fold verification and is not an accepted model. A separate
64-configuration admission-quality comparison also failed to establish final
accuracy. All earlier measurements and failures remain retained. These results
have not changed the checked-in coefficients.

All 1772 existing A/B binaries had verified hashes and complete ELF resource
descriptors, including 30 existing FP8 byte-transport alternatives. The current
generic formula domain is 26 mixed layouts ending in D, F=1, rank 2--5.
Regular-fragment and dedicated-pattern models are separate work, outside this
PR's responsibility. Existing structured and indirect fallback models remain
available outside the generic domain.

Direction, group and repeat-quality metrics are reports under the revised
overall-MAPE policy. The checked-in Load profiles remain the original TRAIN
banks; their development accuracy is not substituted for final acceptance.

## Reference API

The pure Python reference is `python/indirect_load_selected/predictor.py`:

```python
result = predict(model, case, explicit_host_coefficients,
                 stage_ownership=True, matched_increment=True)
```

The caller provides the correct frozen mode model. A false `eligible` result or
a rejected metadata contract requires the original fallback. Arithmetic
eligibility does not certify arbitrary kernels or whole-kernel accuracy.
