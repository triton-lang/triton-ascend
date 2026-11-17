# Ascend build dependency inputs

Each logical dependency has one authoritative input. Preserve upstream text
pins and native Git submodule metadata instead of maintaining JSON mirrors.

| Input | Consumers |
| --- | --- |
| `llvm-hash.txt` | Existing upstream reader/wrapper and Ascend setup hook, LLVM build workflow and installation examples |
| `json-version.txt` | Ascend `setup_patch.py` JSON package reader |
| `bisheng-version.txt` | integration-tests-ascend, sync-upstream, DynamicCVPipeline and TLA installers; filename and mandatory SHA256 per architecture |
| `triton-distributed-info.json` | Optional distributed source preparation; repository and full commit SHA |
| `npuir-info.json` | NPU-IR CI tracking policy; repository address comes from `.gitmodules`, local pinned revision stays in the gitlink |
| `nvidia-toolchain-version.json` | Existing upstream NVIDIA toolchain mechanism; no Ascend copy |

LLVM patch files remain in `third_party/ascend/patch/llvm_patch_*.patch`.
The existing sorted patch-content hash and prebuilt LLVM naming stay unchanged,
including the AlmaLinux ABI exception. No dependency version is upgraded here.
Core upstream build entry points (`setup.py`, `python/build_helpers.py`,
`CMakeLists.txt`, `Makefile`, `scripts/build-llvm-project.sh`) are unchanged
relative to this PR's main-dev merge base; Ascend overrides live under
`third_party/ascend/build/`.

BiSheng readers use existing shell tools. Packaging uses Python's standard
library and does not require jq. The integration installer keeps `.run` and
`.tar.gz` support; the other three retain `.run` support and reject unsupported
formats before downloading. Switching all consumers to tar payloads is a
separate behavior change. Source metadata and patch bytes enter relevant cache
keys; documentation does not.

Python/build prerequisites stay in existing requirements/project declarations.
Runtime CANN/torch combinations, historical release recipes and image matrices
keep their existing roles: different supported environments are not duplicate
pins of the same logical source dependency. This change adds no release-image
product and changes no CI throttling policy.

Run from the repository root:

```bash
python3 third_party/ascend/build/check_bom_consistency.py
python3 -m pytest -q third_party/ascend/build_tests
```

The audit validates schemas, documentation references and cache paths. It does
not certify binaries or NPU runtime compatibility. Build equivalence requires
before/after wheels with the same toolchain, source dependencies, payload,
build path and flags, followed by file-content comparison and runtime checks.
