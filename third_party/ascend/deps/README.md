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

`toolchain.json` and `runtime.json` are currently read by
`scripts/check_bom_consistency.py` only; the workflows still carry their own
copies of some of these values and are migrated to read the BOM directly in
follow-up changes. The consistency check keeps the two in sync in the
meantime.

## Relation to the `cmake/` mirrors

Two legacy single-value files under `cmake/` are still consumed by code shared
with upstream Triton and are kept as mirrors of this BOM:

- `cmake/llvm-hash.txt` mirrors `llvm-info.json` -> `llvm_hash`
- `cmake/json-version.txt` mirrors `toolchain.json` -> `json`

`scripts/check_bom_consistency.py` (run by the *BOM Consistency Check* CI
workflow) fails if a mirror drifts from the BOM, so a version bump that
forgets one of the copies is caught at PR time.

`cmake/bisheng-version.txt` is the pre-BOM location of the Bisheng pins and is
still consumed by several workflows; it will be removed once those workflows
read `toolchain.json` directly.

## Updating a dependency

1. Edit the entry in the JSON file in this directory. Never edit the `cmake/`
   mirrors by hand.
2. Update the mirrors (`cmake/llvm-hash.txt`, `cmake/json-version.txt`) and —
   until the remaining migrations land — the still-hardcoded workflow values
   to the same version.
3. Run `python scripts/check_bom_consistency.py` locally; it must pass.
4. Commit everything in one change so cache keys derived from these files
   rotate atomically.

## Out of scope

- The `Triton-distributed-ascend` submodule (pinned via git like all
  submodules).
- Release-line docker images under `docker/3.x.x-*/` (frozen per release).
