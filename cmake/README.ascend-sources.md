# Optional Ascend source dependencies

`triton-distributed-info.json` owns the distributed repository and immutable
revision. It is read only with `TRITON_BUILD_TD` enabled. A Git checkout is
verified against the metadata and can reuse the target revision offline.
Changing the remote requires an explicit local checkout update; local changes
are never overwritten to switch revisions. Source archives retain the existing
vendored-source contract: a populated source tree without Git metadata is used
as supplied, without network access. Such archives do not authenticate their
own provenance; this refactor introduces no new required archive manifest.

`npuir-info.json` identifies the native submodule and the CI tracking ref.
`third_party/ascend/build/npuir_source.py` reads its repository from
`.gitmodules`. Local setup still uses the native gitlink; ordinary CI still
tracks the existing `stable` branch. Pinned-wheel verification still uses the
gitlink SHA. The wheel workflow resolves its tracking ref once per matrix job
and passes that SHA to the Docker build, including the in-image build fallback.
Historical matrix branches without the reader use this workflow revision's
policy checkout; no fallback URL/branch literal is maintained in the workflow.
Their existing build recipes continue receiving the selected SHA.

`NPUIR_REPO_URL` and `NPUIR_REPO_BRANCH` remain explicit diagnostic overrides.
An exact SHA avoids ref resolution. Missing/invalid policy fails before a
download. Nested dependency revisions remain native gitlinks in the selected
NPU-IR tree, rather than copies in Triton's metadata. An upstream branch build
failure remains visible; the metadata change does not silently select an older
working commit.

These inputs define source selection, not a complete runtime compatibility
matrix or certification of payload provenance. Existing artifact fallback and
publication behavior is preserved. Historical release snapshots need not
follow current development pins.
