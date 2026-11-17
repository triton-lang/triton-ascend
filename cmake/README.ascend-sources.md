# Optional Ascend source dependencies

`triton-distributed-info.json` owns the distributed repository and immutable
revision. It is read only with `TRITON_BUILD_TD` enabled. A Git checkout is
verified against the metadata and can reuse the target revision offline.
Changing the remote requires an explicit local checkout update; local changes
are never overwritten to switch revisions. Source archives retain the existing
vendored-source contract: a populated source tree without Git metadata is used
as supplied, without network access. Such archives do not authenticate their
own provenance; this refactor introduces no new required archive manifest.

The AscendNPU-IR source keeps its existing policy. Its repository URL is in
`.gitmodules`; local builds use the native gitlink, and the existing wheel and
payload workflows follow `stable` through their original fetch script.
This PR does not migrate that policy, add another pin or modify those workflows.
Nested dependency revisions remain gitlinks in the selected NPU-IR tree.
The existing `NPUIR_REPO_URL` and `NPUIR_REPO_BRANCH` diagnostic overrides
remain available. A separate source-policy migration must validate its own
external compiler and payload builds before it is proposed for merge.

These inputs define source selection, not a complete runtime compatibility
matrix or certification of payload provenance. Existing artifact fallback and
publication behavior is preserved. Historical release snapshots need not
follow current development pins.
