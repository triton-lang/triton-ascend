# Version Release Policy

## Version Numbering

Triton-Ascend follows [PEP 440](https://peps.python.org/pep-0440/) version specification, with version numbers aligned with upstream Triton: `vMAJOR.MINOR.PATCH[rcN][.postN]`

- **MAJOR.MINOR**: Corresponds one-to-one with upstream Triton versions, e.g., Triton-Ascend `3.2` is based on Triton `3.2`
- **PATCH**: Triton-Ascend's `PATCH` version may be higher than upstream Triton, used for issue fixes or improvements at the `MAJOR.MINOR` level, e.g., both Triton-Ascend `3.2.0` and `3.2.1` are based on Triton `3.2.0`
- **rcN**: Release Candidate, published as needed for early testing and feedback from the community
- **postN**: Post-release patches for already released versions, published as needed to address issues in stable versions

## Branching Strategy

- The `main` branch is the latest development branch, tracking the latest upstream Triton version
- Each release version creates a corresponding release development branch (e.g., `release/3.2.x`), which has the same commit id as the community release
- Feature development should be done in fork repositories and merged into the Triton-Ascend repository via `PR`

**`main` Branch Mapping:**

| Triton-Ascend | Triton commit hash                                           | Python    | CANN  | PyTorch | LLVM commit hash                                             | Patch                                                        |
| ------------- | ------------------------------------------------------------ | --------- | ----- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `main`        | [cfc0a9d](https://github.com/triton-lang/triton-ascend/commit/cfc0a9d) | `3.9~3.13` | `9.0.0` | `2.7.1`   | [fad3272](https://github.com/llvm/llvm-project/commit/fad3272) | [llvm_patch_fad3272.patch](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/patch/llvm_patch_fad3272.patch) |

## Maintenance Branches and Lifecycle

Maintenance branch statuses include:

- **Active**: Continuously accepts bug fixes, feature improvements, and security patches; will continue to evolve features or release new versions
- **Maintenance**: Only accepts critical bug fixes and security patches; no longer releases feature improvements
- **End of Life**: No longer accepts any fixes; branch maintenance has stopped

| Branch              | Status     | Triton Version | Triton-Ascend Release              | End of Maintenance |
| ----------------- | -------- | ---------------- | ----------------------------------- | ------------ |
| `main`            | `Active`   | `3.5.0`          | /                                   | /            |
| `release/3.2.1` | `Active`   | `3.2.0`          | `3.2.1`                             | /            |
| `release/3.2.x` | `Maintenance`   | `3.2.0`          | `3.2.0rc2`，`3.2.0rc3`，`3.2.0rc4`，`3.2.0` | /            |

## Release Cycle

- **Stable releases**: Released according to project version cadence, not every upstream Triton version will have a corresponding stable release
- **rc releases**: Released in sync with upstream Triton version cadence for early user testing
- **post releases**: Released as needed to address issues in existing stable versions

### Release Timeline

| Date       | Event                     |
| ---------- | ------------------------ |
| 2026-05-06 | Released stable version `3.2.1`   |
| 2026-01-21 | Released stable version `3.2.0`   |
| 2025-11-14 | Released preview version `3.2.0rc4` |
| 2025-11-12 | Released preview version `3.2.0rc3` |
| 2025-05-26 | Released preview version `3.2.0rc2` |

## Version Compatibility Matrix

| Triton-Ascend | Triton | Python              | CANN  | PyTorch | LLVM commit hash | LLVM Patch |
| ------------- | ------ | ------------------- | ----- | ------- | ---------------- | ---------- |
| `3.2.1`       | `3.2.0` | `3.9`(x86), `3.10-3.13` | `9.0.0` | `2.7.1`   | `b5cc222`        | -          |
| `3.2.0`       | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `b5cc222`        | -          |
| `3.2.0rc4`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `b5cc222`        | -          |
| `3.2.0rc3`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `86b69c3`        | -          |
| `3.2.0rc2`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `86b69c3`        | -          |
