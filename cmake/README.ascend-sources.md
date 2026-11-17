# Optional Ascend source dependencies

The source-policy migration in
[PR #3161](https://github.com/triton-lang/triton-ascend/pull/3161) targets
`main-dev`. Until that build change lands, `main` retains the distributed
repository and revision embedded in `third_party/ascend/build/setup_patch.py`.
The proposed `cmake/triton-distributed-info.json` is not yet a `main` input.

The proposal moves those same values to metadata without upgrading the
dependency. It is read only with `TRITON_BUILD_TD` enabled. Git checkouts verify
their origin and revision and can reuse an existing target commit offline;
switching revisions must not overwrite local changes. Source archives retain
the existing vendored-source contract: populated sources without Git metadata
are used as supplied, without a mandatory new provenance manifest. This does
not authenticate the archive's provenance.

AscendNPU-IR keeps its existing policy. Its repository URL is in `.gitmodules`;
local builds use the native gitlink, while the existing wheel and payload
workflows follow `stable` through their original fetch script. The proposal
does not migrate that policy, add another pin or modify those workflows.
Nested dependencies remain gitlinks in the selected NPU-IR tree. The existing
`NPUIR_REPO_URL` and `NPUIR_REPO_BRANCH` diagnostic overrides remain available.

Source identity, payload provenance and runtime compatibility are separate
validation concerns. Historical release snapshots need not follow current
development pins.
