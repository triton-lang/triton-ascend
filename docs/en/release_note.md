# Triton-Ascend Releases

Triton-Ascend releases provide a stable snapshot of the codebase, packaged as binary distributions that can be easily installed via PyPI. In addition, a release represents the formal announcement by the development team to the community of the availability of new features, completed improvements, and changes that may affect users (e.g., breaking changes).

## Release Schedule

The following is the Triton-Ascend release schedule. Note that patch releases are optional.

| Major Version | Release Branch Cut Date | Release Date     | Version Nature   |
|-------------|-------------------------|------------------|------------------|
| 3.6.0       | September 30, 2026      | October 15, 2026 | Official release |
| 3.2.2.post2 | September 30, 2026      | October 15, 2026 | Patch version    |
| 3.2.2       | July 15,2026            | July 31,2026     | Official release          |
| 3.2.1       | April 15,2026           | May 6,2026       | Official release          |
| 3.2.0       | December 8,2025         | January 21,2026  | Official release          |

Note: Version 3.2.0 is archived on [gitcode (AtomGit)](https://gitcode.com/Ascend/triton-ascend/releases), while version 3.2.1 and later have been migrated to [GitHub](https://github.com/triton-lang/triton-ascend/releases).

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

### Maintenance Branches and Lifecycle

Maintenance branch statuses include:

- **Active**: Continuously accepts bug fixes, feature improvements, and security patches; features will continue to evolve or new releases will be published
- **Maintenance**: Only accepts critical bug fixes and security patches; no feature improvements are released
- **End of Life**: No fixes are accepted; maintenance of the branch has stopped

| Branch          | Status     | Triton Version | Triton-Ascend Release                    | End of Maintenance |
|-----------------| -------- | ------------ |------------------------------------------| -------- |
| `main`          | `Active`   | `3.6.0`      | /                                        | /        |
| `release/3.6.0` | `Active`   | `3.6.0`   | `3.6.0`                                  | /        |
| `release/3.2.2` | `Maintenance`   | `3.2.0`      | `3.2.2`                                  | /        |
| `release/3.2.1` | `Maintenance`   | `3.2.0`      | `3.2.1`                                  | /        |
| `release/3.2.x` | `Maintenance`   | `3.2.0`      | `3.2.0rc2`, `3.2.0rc3`, `3.2.0rc4`, `3.2.0` | /        |

## Release Cycle

- **Stable releases**: Released according to the project's version cadence; not every upstream Triton version has a corresponding stable release
- **rc releases**: Released in sync with the upstream Triton version cadence, for early user testing
- **post releases**: Published as needed to fix issues in existing stable releases

<a id="version-compatibility-matrix"></a>

## Recommended Version Combinations

| Triton-Ascend | PyTorch  | TorchNPU Package | CANN    | SHMEM  | LLVM commit hash | Python              | LLVM Patch |
|---------------|----------|------------------|---------|--------|------------------|---------------------|------------|
| `main`        | `2.10.0` | `2.10.0.post7`   | `9.2.0` | `v1.7.0` | `f6ded0b`        | `3.9~3.13`          | [llvm_patch_f6ded0b.patch](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/patch/llvm_patch_f6ded0b.patch) |
| `3.6.0`       | `2.10.0` | `2.10.0.post7`   | `9.2.0` | `v1.7.0` | `f6ded0b`        | `3.10~3.13`         | - |
| `3.2.2.post2` | `2.7.1`  | `2.7.1.post11` | `9.2.0` | `v1.7.0` | `b5cc222`        | `3.9`(x86), `3.10-3.13` | -         |
| `3.2.2`       | `2.7.1`  | `2.7.1.post8`    | `9.1.0` | -      | `b5cc222`        | `3.9`(x86), `3.10-3.13` | -          |
| `3.2.1`       | `2.7.1`  | `2.7.1.post8`    | `9.0.0` | -      | `b5cc222`        | `3.9`(x86), `3.10-3.13` | -          |
| `3.2.0`       | `2.6.0`  | `2.6.0`          | `8.5.0` | -      | `b5cc222`        | `3.9-3.11`          | -          |

## Version Compatibility Notes

> In the following tables, "/" means not compatible, and "Y" means compatible.

**Table 1** Triton-Ascend and TorchNPU Package Version Compatibility

<table class="table">
  <thead>
    <tr>
      <th rowspan="2">Triton-Ascend</th>
      <th colspan="8" style="text-align:left">TorchNPU Package Version</th>
    </tr>
    <tr>
      <td>2.6.0</td>
      <td>2.7.1</td>
      <td>2.8.0</td>
      <td>2.9.0</td>
      <td>2.10.0</td>
      <td>2.13.0</td>
    </tr>
  </thead>
  <tbody>
    <tr><td>3.2.0</td><td>2.6.0</td><td>/</td><td>/</td><td>/</td><td>/</td><td>/</td></tr>
    <tr><td>3.2.1</td><td>/</td><td>2.7.1</td><td>2.8.0</td><td>2.9.1</td><td>2.10.0</td><td>/</td></tr>
    <tr><td>3.2.2</td><td>/</td><td>2.7.1.post8</td><td>/</td><td>2.9.0.post6</td><td>2.10.0.post4</td><td>/</td></tr>
    <tr><td>3.2.2.post2</td><td>/</td><td>2.7.1.post11</td><td>/</td><td>2.9.0.post9</td><td>2.10.0.post7</td><td>/</td></tr>
    <tr><td>3.6.0</td><td>/</td><td>/</td><td>/</td><td>/</td><td>2.10.0.post7</td><td>2.13.0</td></tr>
  </tbody>
</table>

**Table 2** Triton-Ascend and CANN Version Compatibility

<table class="table">
  <thead>
    <tr>
      <th rowspan="3">Triton-Ascend</th>
      <th colspan="8" style="text-align:left">CANN Version</th>
    </tr>
    <tr>
      <th colspan="4" style="text-align:left">Atlas A2 series products, Atlas A3 series products</th>
      <th colspan="4" style="text-align:left">Ascend 950PR&950DT series products</th>
    </tr>
    <tr>
      <td>8.5.0</td>
      <td>9.0.0</td>
      <td>9.1.0</td>
      <td>9.2.0</td>
      <td>8.5.0</td>
      <td>9.0.0</td>
      <td>9.1.0</td>
      <td>9.2.0</td>
    </tr>
  </thead>
  <tbody>
    <tr><td>3.2.0</td><td>Y</td><td>/</td><td>/</td><td>/</td><td>/</td><td>/</td><td>/</td><td>/</td></tr>
    <tr><td>3.2.1</td><td>/</td><td>Y</td><td>/</td><td>/</td><td>/</td><td>Y</td><td>/</td><td>/</td></tr>
    <tr><td>3.2.2</td><td>/</td><td>Y</td><td>Y</td><td>/</td><td>/</td><td>Y</td><td>Y</td><td>/</td></tr>
    <tr><td>3.2.2.post2</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>/</td><td>/</td><td>Y</td><td>Y</td></tr>
    <tr><td>3.6.0</td><td>Y</td><td>Y</td><td>Y</td><td>Y</td><td>/</td><td>/</td><td>Y</td><td>Y</td></tr>
  </tbody>
</table>
