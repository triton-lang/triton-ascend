# Ascend build dependency inputs

Each logical dependency has one authoritative input. Preserve upstream text
pins and native Git submodule metadata instead of maintaining JSON mirrors.

| Input | Consumers |
| --- | --- |
| `llvm-hash.txt` | Existing upstream reader/wrapper and Ascend setup hook, LLVM build workflow and installation examples |
| `json-version.txt` | Ascend `setup_patch.py` JSON package reader |
| `bisheng-version.txt` | integration-tests-ascend, sync-upstream, DynamicCVPipeline and TLA (`triton.language.extra.tla`) installers; filename and mandatory SHA256 per architecture |
| `triton-distributed-info.json` | Optional distributed source preparation; repository and full commit SHA |
| `nvidia-toolchain-version.json` | Existing upstream NVIDIA toolchain mechanism; no Ascend copy |

LLVM patch files remain in `third_party/ascend/patch/llvm_patch_*.patch`.
The existing sorted patch-content hash and prebuilt LLVM naming stay unchanged,
including the AlmaLinux ABI (Application Binary Interface) exception. No dependency version is upgraded here.
Core upstream build entry points (`setup.py`, `python/build_helpers.py`,
`CMakeLists.txt`, `Makefile`, `scripts/build-llvm-project.sh`) are unchanged
relative to this PR (Pull Request)'s main-dev merge base; Ascend overrides live under
`third_party/ascend/build/`.

BiSheng readers use existing shell tools. Packaging uses Python's standard
library and does not require jq. The integration installer keeps `.run` and
`.tar.gz` support; sync-upstream retains `.run` support. These two migrated
readers validate filenames and checksums before downloading. The unchanged
DynamicCVPipeline and TLA workflows retain their original readers of the same
text pin. Switching all consumers to tar payloads is a separate behavior change.
Source metadata and patch bytes enter the migrated cache keys; documentation
does not.

Python/build prerequisites stay in existing requirements/project declarations.
Runtime CANN/torch combinations, historical release recipes and image matrices
keep their existing roles: different supported environments are not duplicate
pins of the same logical source dependency. This change adds no release-image
product and changes no CI (Continuous Integration) throttling policy.
The AscendNPU-IR source-policy migration is deferred. Its native Git metadata,
fetch script and wheel/payload workflows remain unchanged from the merge base.
No compiler dependency, device test or workflow trigger is modified to hide a
failure in those existing pipelines.

Run from the repository root:

```bash
python3 third_party/ascend/build/check_bom_consistency.py
python3 -m pytest -q third_party/ascend/build_tests
```

The audit validates the migrated schemas, documentation references and cache
paths in integration-tests-ascend and sync-upstream. Unchanged workflows keep
their baseline cache policy. It does not certify binaries or NPU runtime
compatibility. Build equivalence requires
before/after wheels with the same toolchain, source dependencies, payload,
build path and flags, followed by file-content comparison and runtime checks.
