# CPU block selection (draft)

This module scores already compiled CV and VV candidates using host-side LLVM
analysis and backend resource evidence. The shortlist contains F1 and at most
two F2/F4 candidates. No device timing or tensor contents are used.

The Ascend backend routes its existing JIT continuation through
`block_auto_dispatch.get_launch_handler()`. On current `main-dev`, no candidate
provider exists: the handler delegates unchanged to the ordinary JIT or enabled
dynamic-reuse handler. It does not generate candidates, collect artifacts, run
the model, change options, or alter the launched kernel. Warmup also uses this
original path. This draft therefore provides no block-merging speedup.

A future backend-owned `run_block_auto` provider can receive the CPU
`select_candidates` callback and the original `continuation`. That provider must
own candidate generation, same-compilation evidence collection, cache identity,
graph-capture handling and guarded dispatch; it must preserve the original
launch when a candidate cannot be proved. It must not run additional device
measurements. No such provider or block-merging transformation is included here.

The independent model entry points are:

- `block_auto_cv.candidate_plan(contract, probe=...)`: choose at most two merged
  candidates from frontend-proved M/N/batch roles. The supplied probe reports a
  supported role or raises `ValueError`; the model does not rewrite IR.
- `block_auto_model.select_candidates(candidates, contract)`: verify preserved
  artifacts and resource limits, then select an available candidate or F1.
- `ascend.analyze_block_candidate(request_json)`: analyze candidate LLVM on the
  CPU with the C++ scalar, instruction, pipeline and contention model.

The fixed Ascend950PR_9579 instruction and bandwidth assumptions are compiled
into the C++ profile; no external instruction JSON or INI is loaded. These
constants preserve the previous model and are not device measurements.
CV numeric coverage is FP32;
VV MIX and the pow helper require the matching recorded compiler/library
profiles. Missing evidence, unsupported operations and unmodeled transformations
remain explicit reasons to keep F1. A model decision is not proof of runtime
memory safety or numerical correctness; the future provider must retain the
corresponding execution guards.
