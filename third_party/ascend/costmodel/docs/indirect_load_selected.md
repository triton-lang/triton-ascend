# Indirect Load physical formulas

The enabled Load route selects short formulas from visible physical geometry,
including storage width, static mask, axis structure and warp geometry. Each
executed formula has at most eight terms. Coefficients come from TRAIN fits;
there is no case-name, kernel-name or hidden tensor-content lookup. The frozen
program and exact coefficients are checked in as readable C++ compiler data.
The flat table contains 108 physical routes and 62 unique short formulas,
including the TRAIN base formulas for unmatched domains. It replaces the nested
conditional, scale and blend interpreter. Every selected path evaluates one
linear formula with at most eight terms; the minimum prediction remains one tick.

## Validation and limits

The authorized, already-observed 886-configuration dataset was split by fixed
geometry hash into 734 TRAIN and 152 validation configurations. Four observation
rounds retain 3544 files and 850560 quartets. Nonpositive differential labels are
retained and diagnosed, not removed by prediction error. Discrete physical-route
selection used the observed validation results, as explicitly permitted.

The selected program's validation MAPE is **19.9795%** on 1187 positive inputs:
SIMD **21.6531%** (472 inputs), SIMT **18.8747%** (715 inputs). The other 29
validation inputs have nonpositive differential labels and are reported outside
MAPE. This meets the requested overall 20% observed-validation threshold; it is
**not an unseen-test result**. Groups, direction and repeat quality are reports.

The development set was used for discrete formula selection, so comparable
unseen-kernel accuracy is not established. Three-fold validation is no longer
a required acceptance step; previous model results remain historical evidence.
Missing verified logical-type facts reject this estimator and retain the
original compiler model, rather than selecting a second legacy formula bank.

Actual arithmetic verification covered 7088 input metadata rows and 843472
feature comparisons, with zero feature differences and maximum C++ prediction
difference 1.46e-11. The archived Python reference reproduced all predictions
exactly. Device poison-pointer checks covered 496 inputs without reading hidden
coefficients. The earlier revision passed 60 Python API regression tests; those APIs and their
tests are removed from the compiler-only scope.

A native build of the current cost-model sources passed 104 positive IR checks,
416 guard rejections, 104 target-only Stage compositions, 16 Host/Device binding
checks and four FP8 transport checks. These are synthetic registered-IR unit
fixtures, not hardware performance evidence. The complete upstream Triton
extension has not been rebuilt for this revision.

## Production domain

The measured generic domain is 26 mixed layouts ending in D, rank 2--5, F=1,
with SIMD/SIMT and static masked/unmasked inputs. Dedicated-pattern models are
separate work. Dynamic masks, F>1, multiple matched targets, nested scopes,
unsupported types and unproved geometry retain the existing fallback.

`OrdinaryLoadMain.cpp` derives separable axes, logical payload type, static masks
and visible strides from real post-transform TTIR. It replaces exactly one
matched target Load while retaining auxiliary address/index/mask/output work.
Host strides require entry-argument attributes and matching runtime scalar
bindings. Device strides require an explicit positive/no-wrap contract and stay
opaque. FP8 logical provenance is distinct from byte transport.

The path is enabled by default; `ordinary_load_selected_enabled=False` disables
it. No standalone Python prediction API or owned/event runtime is added.

A rejected contract requires the original fallback. Arithmetic eligibility does
not certify arbitrary kernels or whole-kernel accuracy.
