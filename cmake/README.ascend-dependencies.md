# Ascend build dependency inputs

This page describes the existing pin layout and the reader migration proposed
in [PR #3161](https://github.com/triton-lang/triton-ascend/pull/3161), targeting
`main-dev`. The proposed metadata and audit are not yet part of `main`; do not
assume that every proposed file or reader below exists in a standard `main`
checkout. The proposal preserves dependency versions and native upstream pins.

| Input | Role and availability |
| --- | --- |
| `llvm-hash.txt` | Existing LLVM revision pin; Ascend prebuilt artifact identity also includes sorted local patch bytes |
| `json-version.txt` | Existing nlohmann JSON tag file; the Ascend setup reader migration to this file is proposed in #3161 |
| `bisheng-version.txt` | Existing installer filename and SHA256 per architecture, shared by the integration, sync, DynamicCVPipeline and TLA (`triton.language.extra.tla`) workflows |
| `triton-distributed-info.json` | Proposed in #3161 to replace the repository and full revision embedded in `third_party/ascend/build/setup_patch.py` |
| `nvidia-toolchain-version.json` | Existing upstream NVIDIA toolchain metadata; no Ascend mirror is proposed |

LLVM patches remain in `third_party/ascend/patch/llvm_patch_*.patch`. The
proposal retains their ordering, content hash and the existing AlmaLinux ABI
(Application Binary Interface) exception. It does not replace text pins with
JSON mirrors or change core upstream build entry points. Python build
prerequisites remain in existing requirements/project declarations.

The proposed readers stay under `third_party/ascend/build/`. Packaging uses
Python's standard library; shell installers keep their existing tools and
package-format support. CI (Continuous Integration) cache auditing covers the
two migrated consumers, integration-tests-ascend and sync-upstream; unchanged
workflows keep their existing policies. The proposal adds no release-image
product and changes no CI scheduling or device tests.

Only in a checkout that includes #3161, validate its build inputs with:

```bash
python3 third_party/ascend/build/check_bom_consistency.py
python3 -m pytest -q third_party/ascend/build_tests
```

These commands are validation instructions for the proposed branch, not
installation prerequisites on `main`. The audit checks migrated metadata and
cache paths; it does not certify binaries or runtime compatibility. Build
equivalence requires the same toolchain, source dependencies, payload, build
path and flags, followed by file-content comparison and runtime checks.
Historical release recipes and distinct runtime matrices retain their roles.
