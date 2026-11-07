# Indirect-store selected short formulas

The optional ordinary indirect-store estimator uses 88 visible geometry and
stride features and linear formulas with at most eight terms each. Routing uses
storage width, presence of a mask, and whether the tile has at most 128 elements.
It does not use kernel names, case IDs, or a performance lookup table.

`OrdinaryStoreRound4Main.cpp` extracts facts from the actual StoreOp and its entry
arguments. Known coefficients are read from constants; Host coefficients require
explicit runtime argument mapping. Device-hidden coefficients are never read.
Logical FP8 is tracked separately from the one-byte transport type. Ownership,
first-store and matched-increment contracts are required before a formula can
replace the target contribution. Unsupported cases retain the existing model.

The selected estimator and owned-session API are enabled by default. Explicit
disable remains supported. A formula does not grant a fresh-output lease or
bypass event synchronization: the compiler-issued certificate, exact model and
freeze identities, and per-launch buffer/event checks remain mandatory. The
guarded owned runtime currently supports rank-2 SD, F=1, precompiled float32/int32
kernels. Other shapes retain the existing route unless the explicit native
cost-analysis contract proves their eligibility.

## Validation result and limits

The frozen final validation contains 256 configurations, two sessions and 2048
independent input identities (512 observation files). Its overall MAPE is
**8.2125679121%**, with **5.3364549858% SIMD** and **11.0886808385% SIMT**.
Reporting covers an ordinary store geometry panel: ranks 2--5, **26 mixed S/D
layouts ending in D**, unmasked and rectangular box masks, storage widths 1/2/4/8,
and Known/Host/Device coefficient visibility. Eligibility remains bounded by the
actual metadata guards; the result is not a claim about arbitrary kernels.

The final set was used, with explicit user authorization, to select existing
TRAIN-fitted short leaves. Coefficients were copied exactly from those models.
This is a validation-selected result, **not an unseen independent-test result**.
The source candidates completed three-fold TRAIN validation and independent
scalar arithmetic checks; the newly assembled model did not run a new CV search.
Per-group accuracy and direction are reported, not acceptance gates under the
revised overall-MAPE criterion. Two original repeat-quality failures are retained;
the requested follow-up collection was subsequently cancelled. No observations
were removed or replaced to obtain the reported MAPE.

The standalone C++ check compared 15128 metadata rows and 1331264 feature values:
feature difference zero, maximum prediction difference 5.82077e-11. Three native
translation units, eight report checks, one full Triton incremental build and
four AST-to-TTIR/public-Select frontend checks passed on the validated 70f770c
integration. After porting to current upstream b8b6a2d, all five changed native
translation units passed syntax checks and the 15128-row parity check passed
again. The complete current-upstream extension has not been rebuilt.
After enabling the guarded production API and binding its selected-model/freeze
identities, the two affected native translation units and 49 Python regression
tests passed again.

Real-device buffer ownership/lifetime, asynchronous event integration and the
F>1 whole-kernel path remain unaccepted. F=1 formula accuracy does not establish
those properties. The default-enabled API retains its runtime checks; this change
does not claim that those hardware checks have been performed.

## Original matrix coverage

This change is a partial implementation of the original Load/Store plan, not a
claim that the complete plan has passed. The original mixed-axis matrix contains
52 layouts; the other 26 layouts ending in S are not covered by the final Store
validation above. Page and multiple-base structures, data-dependent masks and
whole-kernel factor coverage also require their own evidence. Existing fallback
models remain responsible for cases outside the selected estimator's guard.
Load validation and integration are tracked separately; this PR's Store MAPE
must not be presented as a Load result.

## Reproduction

Exact model JSON files are in `profiles/indirect_store_selected/`. Their SHA256:

* SIMD: `345cfe338c438661c8ccdab7a138114f8d8a94e87492084a6f4739f96d5b16ca`
* SIMT: `88159dbe346d4c9fd152a164d017528386958071c013ef6c27267958ed29e534`

The pure Python reference is in `python/indirect_store_selected/` and the C++
feature bank and exact coefficients are `OrdinaryStoreSelected.h` and
`OrdinaryStoreSelectedFormulas.inc`. CPU regression tests run with:

```sh
python -m pytest python/test/unit/ascend/test_indirect_store_selected.py
```

The checked-in metadata fixtures are synthetic arithmetic examples and contain
no measured labels. The large real-metadata fixture and original measurements
are retained in the separately delivered validation archive.
