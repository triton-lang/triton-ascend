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
57 tests. A native build of the current cost-model sources and their real
OffsetAnalysis/Utils dependencies passed 104 positive IR checks, 416 guard
rejections, 104 target-only Stage compositions and 16 Host/Device binding
checks. Those registered-IR unit fixtures are synthetic; they are not measured
performance evidence. The complete current-upstream Triton extension has not
been rebuilt.

The separately frozen sparse-GQA source panel has 54 configurations / 432
independent inputs and MAPE 17.7052% (SIMD 17.1964%, SIMT 18.2140%). That model
uses an explicit Host producer contract for a causal valid prefix. Its result
does not permit a generic Load estimator to inspect hidden runtime K; the
generic static-mask integration falls back on that runtime-mask case.

The generic final matrix contains 886 Load configurations, but no matching
measurement files were present in the inspected existing measurement sessions.
It must not be represented as an evaluated final set. The present generic
formula domain covers 26 mixed layouts ending in D, F=1, rank 2--5. The original
52-layout plan, all factors, page/pointer-table structures and distribution
pressure panels remain a broader acceptance task. Existing structured and
indirect fallback models are retained outside this domain.

Direction and group metrics are reports under the revised overall-MAPE policy.
No Load coefficients were fitted during this verification or export.

## Reference API

The pure Python reference is `python/indirect_load_selected/predictor.py`:

```python
result = predict(model, case, explicit_host_coefficients,
                 stage_ownership=True, matched_increment=True)
```

The caller provides the correct frozen mode model. A false `eligible` result or
a rejected metadata contract requires the original fallback. Arithmetic
eligibility does not certify arbitrary kernels or whole-kernel accuracy.
