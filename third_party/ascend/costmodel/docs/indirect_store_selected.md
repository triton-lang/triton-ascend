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

This change contains compiler-side formula integration only. The explicit native
contract still binds ownership, first-store, visibility and accounting facts.
No new owned-session, event, launcher or standalone Python API is installed.

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
Those checks describe the earlier integration revision. The scoped compiler-only
refactor is validated separately; historical API tests are not evidence for it.
Real-device ownership, asynchronous events and F>1 remain outside this change.

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

The complete readable coefficient tables are in
`OrdinaryStoreSelectedFormulas.inc`. The bound source-model SHA256 values remain
in that table. Duplicate JSON exports are retained in the validation archive
and prior revision, rather than shipped as a second unused coefficient copy.

The C++ feature bank and exact coefficients are `OrdinaryStoreSelected.h` and
`OrdinaryStoreSelectedFormulas.inc`. The removed standalone Python reference is
retained in the validation archive as an arithmetic oracle, not a public API.
