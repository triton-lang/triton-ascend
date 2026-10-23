# Dependency BOM (Bill of Materials)

This directory is the single source of truth for the external dependencies
pinned by Triton-Ascend builds and CI. Consumers — the `setup.py` hooks,
build scripts and GitHub Actions workflows — read their pinned versions from
the files below instead of hardcoding them, mirroring how other backends
(e.g. Intel XPU) keep their dependency pins in one place.

## Files

| File | Contents | Read by |
| ---- | -------- | ------- |
| `llvm-info.json` | LLVM commit hash, `llvm_patch_*.patch` manifest, prebuilt-package checksums | `third_party/ascend/build/setup_patch.py` (`json.load` — prebuilt LLVM download), `scripts/build-llvm-project.sh` (`jq` — local LLVM builds), `.github/workflows/llvm-build.yml` (`jq` — prebuilt LLVM publishing) |
| `toolchain.json` | Bisheng compiler run packages per arch (with sha256), nlohmann-json version | Bisheng install steps in the CI workflows (`jq`) |
| `runtime.json` | Supported CANN runtimes and their matching torch / torch-npu versions | CI matrix values and container image tags |

The CI workflows read the BOM directly: the Bisheng install steps parse
`toolchain.json` with `jq`, and the container image tags in `wheels.yml` /
`build-docker-image.yml` resolve `runtime.json`'s `default_cann` in a
lightweight `deps` job. Values that cannot read JSON at their point of use —
the `requirements.txt` / `docker/Dockerfile` pins and the static matrix
entries — remain hardcoded but are covered by `scripts/check_bom_consistency.py`.

## Relation to the `cmake/` mirrors

Two legacy single-value files under `cmake/` are still consumed by code shared
with upstream Triton and are kept as mirrors of this BOM:

- `cmake/llvm-hash.txt` mirrors `llvm-info.json` -> `llvm_hash`
- `cmake/json-version.txt` mirrors `toolchain.json` -> `json`

`scripts/check_bom_consistency.py` (run by the *BOM Consistency Check* CI
workflow) fails if a mirror drifts from the BOM, so a version bump that
forgets one of the copies is caught at PR time.

## Updating a dependency

1. Edit the entry in the JSON file in this directory. Never edit the `cmake/`
   mirrors by hand.
2. Update the mirrors (`cmake/llvm-hash.txt`, `cmake/json-version.txt`) and
   the still-hardcoded values (`requirements.txt`, `docker/Dockerfile`,
   static matrix entries) to the same version.
3. Run `python scripts/check_bom_consistency.py` locally; it must pass.
4. Commit everything in one change so cache keys derived from these files
   rotate atomically.

## Out of scope

- The `Triton-distributed-ascend` submodule (pinned via git like all
  submodules).
- Release-line docker images under `docker/3.x.x-*/` (frozen per release).
