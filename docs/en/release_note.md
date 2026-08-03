# Triton-Ascend Release Notes

Triton-Ascend releases provide stable codebase snapshots packaged into binary distributions installable via PyPI. Additionally, releases allow the development team to formally announce feature availability, completed improvements, and changes that may affect users (e.g., breaking changes).

## Release Compatibility Matrix

The following is the Triton-Ascend release compatibility matrix:

| Triton-Ascend Version | Python Version | Manylinux Version | Hardware Platform | Hardware Product |
| --- | --- | --- | --- | --- |
| 3.2.0 | >=3.9, <=3.11 | glibc 2.27+, x86-64, aarch64  | Ascend NPU | Atlas A2/A3|

## Release Schedule

The following is the Triton-Ascend release schedule. Note that patch releases are optional.

| Major Version | Branch Cut Date | Release Date | Patch Release Date |
| --- | --- | --- | --- |
| 3.2.0 | 2025-12-08 | 2026-01 | --- |

## Release Highlights

### Triton-Ascend 3.2.0

**Initial Release: Ascend NPU Support**

Triton-Ascend 3.2.0 is the first version to officially support Huawei Ascend NPUs. This release is based on the Triton 3.2.0 community version and is specifically adapted for the Ascend NPU hardware architecture.

#### Key Features

1. **Full-Stack Ascend NPU Support**
   - Complete compilation pipeline from Triton IR to NPU instruction set
   - Support for all Triton Ops

2. **Performance Optimizations**
   - NPU-specific kernel optimizations
   - CV computation optimizations

3. **Developer Tools**
   - Comprehensive debug output support
   - Compilation intermediate artifact dumping

#### Known Limitations

1. **Data Types**: Support for some data types is still being improved
2. **Op Coverage**: The supported op set is being continuously expanded

#### Migration Guide

For existing Triton GPU users migrating to Ascend NPU, see [GPU Triton Operator Migration](./migration_guide/migrate_from_gpu.md)

## Release Versioning Strategy

### Version Numbering

Triton-Ascend follows the [PEP 440](https://peps.python.org/pep-0440/) version specification, with version numbers aligned with upstream Triton: `vMAJOR.MINOR.PATCH[rcN][.postN]`

- **MAJOR.MINOR**: Corresponds one-to-one with the upstream Triton version. For example, Triton-Ascend `3.2` is based on Triton `3.2`
- **PATCH**: The Triton-Ascend `PATCH` version may be higher than the upstream Triton version, used for `MAJOR.MINOR` level bug fixes or improvements. For example, both Triton-Ascend `3.2.0` and `3.2.1` are based on Triton `3.2.0`
- **rcN**: Release candidate versions, published as needed for early community testing and feedback
- **postN**: Subsequent patches to a published release, published as needed to fix issues in stable versions

### Branch Strategy

- The `main` branch is the latest development branch, tracking the latest upstream Triton version
- Each release version creates a corresponding release development branch (e.g., `release/3.2.x`), sharing the same commit id as the community release
- Feature development should be conducted in fork repositories and merged into the Triton-Ascend repository via PRs

**`main` Branch Mapping:**

| Triton-Ascend | Triton commit hash                                           | Python    | CANN  | PyTorch | LLVM commit hash                                             | Patch                                                        |
| ------------- | ------------------------------------------------------------ | --------- | ----- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `main`        | [85400f8](https://github.com/triton-lang/triton-ascend/commit/85400f8) | `3.9~3.13` | `9.0.0` | `2.10.0`   | [f6ded0b](https://github.com/llvm/llvm-project/commit/f6ded0b) | [llvm_patch_f6ded0b.patch](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/patch/llvm_patch_f6ded0b.patch) |

### Maintenance Branches and Lifecycle

Maintenance branch status includes:

- **Active**: Continuously accepts bug fixes, feature improvements, and security patches; will continue to evolve features or publish new versions
- **Maintenance**: Only accepts critical bug fixes and security patches; no further feature improvements
- **End of Life**: No longer accepts any fixes; branch maintenance has ceased

| Branch              | Status     | Triton Version | Triton-Ascend Releases              | Maintenance End |
| ----------------- | -------- | ------------ | ----------------------------------- | -------- |
| `main`            | `Active`   | `3.6.0`      | /                                   | /        |
| `release/3.2.1` | `Active`   | `3.2.0`      | `3.2.1`                             | /        |
| `release/3.2.x` | `Maintenance`   | `3.2.0`      | `3.2.0rc2`, `3.2.0rc3`, `3.2.0rc4`, `3.2.0` | /        |

## Release Cadence

- **Stable Releases**: Published according to the project release rhythm; not every upstream Triton version will have a corresponding stable release
- **rc Releases**: Published in sync with the upstream Triton version rhythm for early user testing
- **post Releases**: Published as needed to fix issues in existing stable versions

### Release Timeline

| Date       | Event                     |
| ---------- | ------------------------ |
| 2026-05-06 | Stable release `3.2.1`     |
| 2026-01-21 | Stable release `3.2.0`     |
| 2025-11-14 | Preview release `3.2.0rc4`  |
| 2025-11-12 | Preview release `3.2.0rc3`  |
| 2025-05-26 | Preview release `3.2.0rc2`  |

## Version Compatibility Matrix

| Triton-Ascend | Triton | Python              | CANN  | PyTorch | LLVM commit hash | LLVM Patch |
| ------------- | ------ | ------------------- | ----- | ------- | ---------------- | --------- |
| `3.2.1`       | `3.2.0` | `3.9`(x86), `3.10-3.13` | `9.0.0` | `2.7.1`   | `b5cc222`        | -         |
| `3.2.0`       | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `b5cc222`        | -         |
| `3.2.0rc4`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `b5cc222`        | -         |
| `3.2.0rc3`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `86b69c3`        | -         |
| `3.2.0rc2`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `86b69c3`        | -         |
