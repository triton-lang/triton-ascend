# Optional Ascend source dependencies

This page describes the metadata files behind the optional Ascend source
dependencies (the distributed backend) and what the wheel-build CI
(Continuous Integration) jobs do with each selection.

`cmake/triton-distributed-info.json` owns the optional distributed backend
repository and revision. The distributed-backend setup step in the Ascend
setup hook (`third_party/ascend/build/setup_patch.py`,
`_ensure_distributed_submodule`) reads this metadata and verifies the
checked-out commit, including outside a Git source checkout. It runs only
when `TRITON_BUILD_TD` is enabled (e.g. `ON`); the distributed backend is
otherwise not built.

The AscendNPU-IR compiler source keeps its existing selection: workflow and
image builds track the upstream `stable` branch through
`.github/workflows/build/fetch-npuir-stable.sh`, which stays at its main-dev
version in this change. AscendNPU-IR source metadata is a follow-up once the
repository/branch selection needs to vary by target.

Wheel build, test and publication share the selected source ref and artifact
tag. The ref is the head that triggered the run, which comes from one of three
sources: the `push` commit, the pull request's head commit, or a ref picked by
a manual `workflow_dispatch` trigger. The `Build-Wheels` job in
`.github/workflows/ascend-wheels.yml` selects this ref via the checkout step's
`ref` expression, without preparing an extra Python interpreter. Pull requests
explicitly select the head commit. Post-submit throttling queries the current
branch and CI workflow filename.

These settings do not define a complete runtime compatibility matrix or certify
compiler ABI (Application Binary Interface) compatibility. A moving watch
branch is not a frozen release lock, and source identity and artifact
provenance have separate validation requirements.
