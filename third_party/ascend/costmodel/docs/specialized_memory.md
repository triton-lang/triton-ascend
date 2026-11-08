# Specialized segments and paged memory model

The 950PR profile estimates complete F1 memory fragments in microseconds. It
covers segments/paged access, Load/Store, SIMD/SIMT and structured masks. A caller
supplies facts proved from target-pointer/mask SSA and explicitly held Host
scalars. The model does not inspect loaded index/page-table contents, addresses,
random seeds, sampled mask ratios, kernel names or source paths.

`estimateSpecializedMemory(geometry)` returns an estimate, its profile hash and a
supported/reason result. `completeFragmentBoundary` must be proved by the caller:
the label includes index/address preparation and the fragment's source/sink. It
must not be added as a per-instruction cost to an arbitrary Stage estimate.
Unknown geometry stays unknown. F>1, tail-D generic patterns, `int1`, SIMD f64,
unproved masks/page origins and inconsistent payload storage fail closed.

For a proved three-pointer direct fragment, `directThreePointerEnvelope` selects
one shared SIMD Load startup term. The proof requires a rank1 i32/i64 index Load,
one rank2 DS target access and its contiguous source/sink, without other payload
computation or memory operations. The admitted envelope is float32, unmasked,
literal unit regular step, F1/W1. It is a structural boundary, not a kernel ID.

## Frozen formula and reported development results

The formula uses public element/byte work, logical segments, affine spans and
contiguous runs, structured mask bounds, logical page geometry and SIMT warp
work. Each mode/operation keeps a budget of 60 numeric parameters. Coefficients
are compiled into `SpecializedMemoryParameters.inc`; inference performs no file
access or fitting. Shared weighted sums replace repeated expanded expressions,
and unused feature variants and zero terms have been removed.

The frozen composition hash is
`c079ef9fccbbe623193a1ee26871816fe9f744c57d077dc39d5b6c748ac0213b`.
Its parent coefficients are unchanged. The direct-envelope term was fitted on
48 TRAIN inputs from six geometries, excluding the regression shape. The
original 991 DEV configurations/7928 observations and two new DEV geometries/16
observations were retained without error-based deletion. The parent working set
excluded 542 TRAIN configurations using a previously frozen TRAIN-error rule;
that exclusion did not remove any DEV observation.

Acceptance uses complete DEV per-input MAPE for four mode/operation groups and
eight mode/operation/family groups (<20%), plus the direct-fragment DEV set.
There is no three-fold or other cross-validation acceptance method.

| Group | Overall DEV MAPE | Segments | Paged |
|---|---:|---:|---:|
| SIMD Load | 16.84% | 19.52% | 12.92% |
| SIMD Store | 16.39% | 18.39% | 13.48% |
| SIMT Load | 14.21% | 13.77% | 14.78% |
| SIMT Store | 18.89% | 18.85% | 18.93% |

The new shape-disjoint DEV MAPE is 6.31%. Mask-subgroup results are diagnostic,
not an assertion that every subgroup passes 20%: SIMD Load/segments/masked is
31.53%, SIMD Store/segments/unmasked 21.81%, and SIMT Store/paged/masked 22.58%.
The local verifier retains these failure metrics. These are development results,
not new independent acceptance results.

Two complete-fragment candidates from the 100-case matrix were replayed offline.
Using the same measured targets, the aggregate MAPE of the rounded predictions
published in [PR2528](https://github.com/triton-lang/triton-ascend/pull/2528) is
19.58%, versus 9.21% for this model. This is not a rerun of the PR2528 binary or
a reproduction of its historical percentages. Neutral TTIR and fixed inputs
were retained, but current F1/W1/grid1 builds and fresh zeroed outputs differ
from the historical lowering/Torch wrapper. All four contexts and eight repeated
input blocks were included. These regression labels were not fitted. The other
98 kernels were not claimed as specialized online hits.

## Public checks and local verification

Only the implementation, frozen coefficients and aggregate metrics are
distributed. Per-case DEV timing labels and golden prediction fixtures stay
local and are absent from this branch and its new commit history. Consequently,
public CI can check synthetic geometry contracts; it cannot reproduce the
reported DEV accuracy without separately supplied local datasets.

The standalone target needs C++17 and CMake, without LLVM, Torch or an NPU:

```sh
cmake -S third_party/ascend/unittest/costmodel_ut/specialized_memory -B build-specialized-memory -DCMAKE_BUILD_TYPE=Release
cmake --build build-specialized-memory --parallel 1
ctest --test-dir build-specialized-memory --output-on-failure
```

The ordinary CostModel GTest target runs the same synthetic structure checks.
The original 3268-case formula replay and 26 boundary fixtures were also run
locally; all 3268 predictions match the previous implementation exactly. The
optional DEV verifier requires explicit local inputs:

```sh
python third_party/ascend/costmodel/tools/validate_specialized_memory.py PREDICTIONS --development DEV_JSON --direct-development DIRECT_DEV_JSON --output DEV_REPORT
```

## Integration boundary

This change provides a callable C++ model and build/test registration. It does
not automatically route production Stages through the complete-fragment cost.
The full TA build and final native-instruction decoding were not performed for
this change. Generic memory models and production routing are unchanged.
