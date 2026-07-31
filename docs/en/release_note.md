# Triton-Ascend Releases

Triton-Ascend releases provide a stable snapshot of the codebase, packaged as binary distributions that can be easily installed via PyPI. In addition, a release represents the formal announcement by the development team to the community of the availability of new features, completed improvements, and changes that may affect users (e.g., breaking changes).

## Release Compatibility Matrix

The following is the release compatibility matrix for Triton-Ascend versions:

| Triton-Ascend Version | Python Version | Manylinux Version | Hardware Platform | Hardware Product |
| --- | --- | --- | --- | --- |
| 3.2.0 | >=3.9, <=3.11 | glibc 2.27+, x86-64, aarch64  | Ascend NPU | Atlas A2/A3|

## Release Schedule

The following is the Triton-Ascend release schedule. Note that patch releases are optional.

| Major Version | Release Branch Cut Date | Release Date | Patch Release Date |
| --- | --- | --- | --- |
| 3.2.0 | December 08, 2025 | January 2026 | --- |

## Release Highlights

### Triton-Ascend 3.2.0

**First Release: Ascend NPU Support**

Triton-Ascend 3.2.0 is the first Triton release with official support for Huawei Ascend NPUs. This release is based on the community Triton 3.2.0 version and is specifically adapted to the Ascend NPU hardware architecture.

#### Key Features

1. **Full-Stack Ascend NPU Support**
   - A complete compilation pipeline from Triton IR to the NPU instruction set
   - Support for all Triton ops

2. **Performance Optimizations**
   - NPU-specific kernel optimizations
   - CV compute optimizations

3. **Developer Tools**
   - Comprehensive debug output support
   - Dump of intermediate compilation artifacts

#### Known Limitations

1. **Data Types**: Support for some data types is still being improved
2. **Op Coverage**: The set of supported ops is being continuously expanded

#### Migration Guide

For existing Triton GPU users migrating to Ascend NPU, see [GPU Triton Operator Migration](./migration_guide/migrate_from_gpu.md)

## Release Strategy

### Version Numbering

Triton-Ascend follows the [PEP 440](https://peps.python.org/pep-0440/) versioning specification, with version numbers aligned with upstream Triton: `vMAJOR.MINOR.PATCH[rcN][.postN]`

- **MAJOR.MINOR**: Corresponds one-to-one with upstream Triton versions; for example, Triton-Ascend `3.2` is based on Triton `3.2`
- **PATCH**: Triton-Ascend's `PATCH` version may be higher than upstream Triton, used for bug fixes or improvements at the `MAJOR.MINOR` level; for example, both Triton-Ascend `3.2.0` and `3.2.1` are based on Triton `3.2.0`
- **rcN**: Release candidate versions, published as needed for early community testing and feedback
- **postN**: Post-releases of already published versions, published as needed to fix issues in stable releases

### Branch Strategy

- The `main` branch is the latest development branch, tracking the latest upstream Triton version
- A release development branch is created for each release (e.g., `release/3.2.x`), sharing the same commit IDs as the community releases
- Feature development should be done in a fork of the repository and merged into the Triton-Ascend repository via `PR`

**`main` branch mapping:**

| Triton-Ascend | Triton commit hash                                           | Python    | CANN  | PyTorch | LLVM commit hash                                             | Patch                                                        |
| ------------- | ------------------------------------------------------------ | --------- | ----- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `main`        | [85400f8](https://github.com/triton-lang/triton-ascend/commit/85400f8) | `3.9~3.13` | `9.0.0` | `2.10.0`   | [f6ded0b](https://github.com/llvm/llvm-project/commit/f6ded0b) | [llvm_patch_f6ded0b.patch](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/patch/llvm_patch_f6ded0b.patch) |

### Maintenance Branches and Lifecycle

Maintenance branch statuses include:

- **Active**: Continuously accepts bug fixes, feature improvements, and security patches; features will continue to evolve or new releases will be published
- **Maintenance**: Only accepts critical bug fixes and security patches; no feature improvements are released
- **End of Life**: No fixes are accepted; maintenance of the branch has stopped

| Branch              | Status     | Triton Version | Triton-Ascend Release              | End of Maintenance |
| ----------------- | -------- | ------------ | ----------------------------------- | -------- |
| `main`            | `Active`   | `3.6.0`      | /                                   | /        |
| `release/3.2.1` | `Active`   | `3.2.0`      | `3.2.1`                             | /        |
| `release/3.2.x` | `Maintenance`   | `3.2.0`      | `3.2.0rc2`，`3.2.0rc3`，`3.2.0rc4`，`3.2.0` | /        |

## Release Cycle

- **Stable releases**: Released according to the project's version cadence; not every upstream Triton version has a corresponding stable release
- **rc releases**: Released in sync with the upstream Triton version cadence, for early user testing
- **post releases**: Published as needed to fix issues in existing stable releases

### Release Timeline

| Date       | Event                     |
| ---------- | ------------------------ |
| 2026-05-06 | Release stable version `3.2.1`     |
| 2026-01-21 | Release stable version `3.2.0`     |
| 2025-11-14 | Release preview version `3.2.0rc4`  |
| 2025-11-12 | Release preview version `3.2.0rc3`  |
| 2025-05-26 | Release preview version `3.2.0rc2`  |

## Version Compatibility Matrix

| Triton-Ascend | Triton | Python              | CANN  | PyTorch | LLVM commit hash | LLVM Patch |
| ------------- | ------ | ------------------- | ----- | ------- | ---------------- | --------- |
| `3.2.1`       | `3.2.0` | `3.9`(x86), `3.10-3.13` | `9.0.0` | `2.7.1`   | `b5cc222`        | -         |
| `3.2.0`       | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `b5cc222`        | -         |
| `3.2.0rc4`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `b5cc222`        | -         |
| `3.2.0rc3`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `86b69c3`        | -         |
| `3.2.0rc2`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `86b69c3`        | -         |
