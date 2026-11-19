# Ascend build dependency pins

| Dependency | Pin and consumer |
| --- | --- |
| LLVM | `llvm-hash.txt` supplies the revision to the build and installation guides. Ascend patches live in `third_party/ascend/patch/llvm_patch_*.patch`. |
| nlohmann/json | `json-version.txt` supplies the release tag to the Ascend setup hook. |
| BiSheng | `bisheng-version.txt` contains a package filename and SHA-256 checksum for each runner architecture. The integration and upstream-sync workflows verify the checksum before installing. |
| Triton-Distributed | `triton-distributed-info.json` supplies the clone URL and full Git revision when `TRITON_BUILD_TD` is enabled. Git checkouts use the pinned revision, and source archives can use a vendored checkout. |
| AscendNPU-IR | The repository URL and pinned revision for local Git checkouts come from `.gitmodules` and the Git submodule entry. |

Run the dependency audit and CPU regression tests from the repository root.

```bash
python3 .github/scripts/check_bom_consistency.py
python3 -m pytest -q third_party/ascend/build/tests
```
