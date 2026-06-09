# Triton-Ascend Release

The Triton-Ascend version provides a stable code base snapshot, which is encapsulated into a binary package that can be easily installed through PyPI. In addition, the release represents that the development team can officially announce the availability of new functions, completed improvements, and changes that may affect users (such as destructive changes) to the community.

## Release Compatibility Matrix

The release compatibility matrix of the Triton-Ascend version is as follows.

| Triton-Ascend Version| Python Version| Manylinux Version| Hardware Platform| Hardware Product|
| --- | --- | --- | --- | --- |
| 3.2.0 | 3.9 to 3.11| glibc 2.27+, x86-64, AArch64  | Ascend NPU | Atlas A2/A3|

## Release Date

The following is the release plan of Triton-Ascend. Note: The patch version is optional.

| Major Version| Release Branch Cut-Out Time| Release Date| Patch Release Date|
| --- | --- | --- | --- |
| 3.2.0 | 2025-12-08| 2026-01| --- |

## Highlights

### Triton-Ascend 3.2.0

**First release: Ascend NPU is supported.**

Triton-Ascend 3.2.0 is the first Triton version that officially supports Huawei Ascend NPU. This version is based on the Triton 3.2.0 community version and is specially adapted to the Ascend NPU hardware architecture.

#### Main Features

1. **Full-stack support for Ascend NPU**
   - The instruction set compilation pipeline from Triton IR to NPU is complete.
   - All Triton Ops are supported.

2. **Performance optimization**
   - NPU-specific kernel optimization
   - CV compute optimization

3. **Developer tools**
   - Comprehensive debug output is supported.
   - Intermediate compilation products are dumped.

#### Known Limitations

1. **Data type**: Some data types are still being improved.
2. **Operator coverage**: The supported operator set is being continuously expanded.

#### Migration Guide

For details about how to migrate existing Triton GPU users to Ascend NPU, see [Migrating Triton Operators from GPUs](./migration_guide/migrate_from_gpu.md).
