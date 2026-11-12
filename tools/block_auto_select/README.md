# CPU block auto selection (CV and VV)

These tools select a block grouping from compiled candidates without executing
an NPU kernel or reading operator timing samples. They are explicit, offline
entry points, not a change to the default JIT or an operator-name lookup table.
Apply the returned factor and axis through the backend's existing guarded block
batching path. A resource check is not a numerical-correctness proof.

| Path | Candidates | Ranking |
| --- | --- | --- |
| VV | F1, one prepared axis for F2, one for F4 | Optimized LLVM instruction dependencies, issue resources, DMA and SIMD/SIMT helpers |
| CV | F1 plus at most two explicitly shortlisted axis/factor pairs | Cube/vector work and synchronization event graph; optional FP32 short-batch rule |

The CV and VV models share LLVM analysis and UB proof code. They use different
execution models. CV's legacy result `choice=0` means `compiler_factor=1`.
Native IR detection of dot operations chooses the CV lowering; it is separate
from selecting a factor with these tools.

## Inputs and CPU preparation

Use Python 3.10 or newer (3.12 was used for device-artifact replay). Python stdlib
is sufficient for scoring; tests additionally require pytest. Compilation also
requires a matching Triton-Ascend extension and BiSheng toolchain.

A manifest contains a `cases` list. Each case supplies `id`, `name`, an explicit
launch `contract` (`grid`, argument shape/dtype/stride/scalars, and options), and
`candidates`. VV additionally supplies a `fixture` JSON whose contract must match.
Candidate keys are `unmerged` or `a<axis>_f<factor>`; each candidate declares its
axis, factor, status and resource proof. Names and IDs label output only.

Successful candidates provide `files` records with `path` and `sha256` for the
binary, metadata, resource result, TTIR, TTAdapter, final allocation IR, compiler,
and LLVM (`llvm` for VV; `llvm_aic`/`llvm_aiv` for CV). MIX and optimized paths
need the additional hash-bound library/default/optimized proof records emitted
by their preparation tools. Rejected candidates retain the actual diagnostic.
Changed artifacts, unknown resource usage and failed compilation are not scores.

`cpu_resource_gate.py` compiles a backend request and checks its allocation
evidence. `prepare_cv_all_axes.py` is a CPU candidate-preparation utility that can
enumerate all eligible axes; this enumeration is **not** the three-candidate
shortlist. Supply the chosen shortlist to the CV entry below. M/N roles must come
from compiler metadata, not from treating every grid axis as a matrix axis.
`prepare_vv_coverage.py` attaches pinned library and compiler-default evidence.
`vv_optimized_ir.py` produces optimized LLVM and checks the generated binary's
executable sections against the original candidate before using that evidence.

## VV: the F1/F2/F4 route

With prepared optimized artifacts:

```bash
python -B tools/block_auto_select/run_vv_skill_selector.py \
  --manifest /path/to/optimized/manifest.json \
  --cost-table /path/to/AVECostModelInfo.def \
  --output /path/to/new-vv-selection \
  --runtime-core-floor 56 --require-optimized-evidence --lowered-entry \
  --experimental-mixed-ranking
```

The core floor 56 is the frozen A5 profile used for the 14-case run, not a device
query or a universal hardware constant. Native sequential-iteration grouping
preserves the launch grid; program grouping reduces it. The selector validates
that contract. `static-predictions.json` contains the decision and selected
artifact, while `static-complete.json` seals its hashes before device evaluation.
The audit disallows runtime imports, device access, subprocesses, network
connections and undeclared input files during scoring.

Incomplete model coverage falls back to a resource-verified F1; if F1 is also
unconfirmed the result has no selectable artifact. It does not call that fallback
optimal. Data-dependent control flow, eliminated helper evidence and compiler
versions outside pinned MIX coverage remain explicit limitations.

## CV: the prepared three-candidate route

Provide a manifest containing F1 plus at most two merged candidates per case:

```bash
python -B tools/block_auto_select/run_cv_spec_experiment.py \
  --manifest /path/to/cv-shortlist.json \
  --hardware /path/to/ascend_davidv100.json \
  --cost-table /path/to/AVECostModelInfo.def \
  --output /path/to/new-cv-selection \
  --factors 2 4 --ranking-model latency --prepared-shortlist \
  --empirical-short-batch
```

`decisions.json` records the factor, axis, comparisons, exclusions and coverage.
The short-batch preference only considers admitted FP32 independent-batch F2
candidates with one spatial tile and an original grid fitting one core wave.
Other cases retain latency ranking. This is the rule used in the existing
12-case CV experiment; it was informed by development results and does not
establish held-out accuracy. Scoring does not read those timings. Omit
`--empirical-short-batch` to use only latency ranking. Work ranking and all-axis
enumeration remain available for diagnostics, separately from the shortlist.

## Validation and limits

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tools/block_auto_select
```

The imported VV implementation was exercised on 14 A5 cases with F1/F2/F4 CPU
selection, unchanged shapes, correctness checks, 5 warmups and the arithmetic
mean of 15 device launches. Eight cases had no other process observed; six were
intentionally measured nonexclusively. Three selected F1 because the model was
incomplete. These measurements do not prove optimality or all-VV coverage.
CV has a separate older 12-case device validation using five-sample medians;
it is not the same build or measurement protocol as the VV run.

Instruction throughput, bandwidth, dependency and queue parameters include
uncalibrated assumptions. Scores are ranking estimates, not measured latency.
Pinned assets include source revisions and hashes in their provenance files.
Publication removes machine-local paths from descriptive provenance; numerical
parameters and executable model logic are retained. No benchmark output,
operator timings, device binaries or personal workspace paths are required in
the repository.
