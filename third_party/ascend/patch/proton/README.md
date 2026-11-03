# Ascend Proton Build Patch

The `proton_patch` branch stores Ascend changes to `third_party/proton` in
`proton.patch`. Its tracked Proton sources match the upstream baseline recorded
in `manifest.json`. Ascend backend code and tests remain ordinary source files.

## Build

Use the existing build command with `TRITON_BUILD_PROTON=ON`. The Ascend
`CMakeBuild.run()` hook applies the patch before compilation, including the
Python files subsequently copied into the wheel. No CUDA dependency download is
introduced. With `TRITON_BUILD_PROTON=OFF`, the hook leaves pristine sources
unchanged or safely reverses a previously applied patch.

Repeated builds are supported. Conflicting local edits or a partially applied
patch stop the build without resetting files. A failed build leaves the applied
patch available for inspection. Do not run ON and OFF builds concurrently in
the same worktree. Switching OFF does not uninstall an already-installed wheel.

To restore only Proton after building:

```bash
python third_party/ascend/build/proton_patch.py restore
```

The existing `python restore_sources.py` also restores this patch before
restoring the other Ascend build patches. Direct CMake users must explicitly run
the helper's `apply` command first; automatic injection belongs to the Python
package build entry point.

## Keep Both Branches Synchronized

1. Implement and commit feature fixes on
   `codex/proton-upstream-boundaries-20261103`, where Proton remains reviewable as
   ordinary source code.
2. On `proton_patch`, restore the generated Proton sources and port the matching
   Ascend-side changes. Keep this branch's patch-loading build hook.
3. Regenerate and verify the complete Proton delta:

```bash
python third_party/ascend/build/proton_patch.py export --source codex/proton-upstream-boundaries-20261103
python third_party/ascend/build/proton_patch.py check --source codex/proton-upstream-boundaries-20261103
```

4. Commit the patch, manifest and matching Ascend-side changes, then push both
   branches. Do not commit build-generated changes under `third_party/proton`.

`export` uses committed source changes. `check` verifies the patch checksum and
exact Proton Git tree equality without modifying the working tree or real Git
index, and rejects staged Proton sources that differ from the baseline. The
manifest records both source and baseline revisions. For a Triton
upgrade, update the pristine baseline and pass its commit using `export --base`.

## Host Tests

```bash
python third_party/ascend/unittest/pytest_ut/test_proton_patch.py
```

These tests cover build gating, repeated application, restoration, conflict
protection, source archives without `.git`, and synchronization checks. They do
not require Torch or an NPU. Runtime tests remain in `unittest/proton_ut` under
the Ascend directory.
