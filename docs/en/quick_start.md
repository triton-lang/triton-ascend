# Quick Start

## Project Introduction

**Triton-Ascend** is an optimized version of Triton adapted for Huawei Ascend processors. It is used for efficient automatic kernel tuning, operator compilation, and deployment. By remaining compatible with Triton's core syntax while being deeply optimized for Ascend NPU features, it helps users quickly develop and deploy high-performance computing tasks on the Ascend platform.

This document uses the online installation and running of the vector addition example via the software package deployment method in an **Ubuntu 22.04** environment as an example, guiding users to quickly get started with **Triton-Ascend**. For more installation methods, please refer to the [Installation Guide](./installation_guide.md).

## Environment Preparation

**Hardware Requirements**

Supported operating systems: Linux (aarch64/x86_64)

Supported Ascend products: Atlas A2/A3/950 series

Minimum hardware configuration: 32 GB memory per card (recommended)

**Software Dependencies**

Determine and install the CANN, Python, and TorchNPU software versions. For the driver and firmware installation, you can refer to [CANN Quick Installation](https://www.hiascend.com/cann/download) on the Ascend community official website.

- CANN version: 9.1.0
- Python version: python3.11
- TorchNPU version: 2.7.1.post8

Note: For more compatibility relationships, please refer to the [Version Description Table](./release_note.md#version-compatibility-matrix).

## Quick Installation

```bash
pip install triton-ascend --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## Quick Start

**Run the vector addition example in tutorials to verify the results**

Vector addition example: [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)
By comparing the output of the Triton kernel with that of native PyTorch computation, it proves that the Ascend NPU device can correctly call the Triton kernel and ensure computational accuracy.

> ⚠️ The following commands must be run in a bash environment. If using POSIX sh, replace `source` with `.`.

```bash
# Set CANN environment variables (taking the root user's default installation path `/usr/local/Ascend` as an example)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Clone the triton-ascend source repository and examples
git clone https://github.com/triton-lang/triton-ascend.git
# Run the tutorials example
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

If you observe similar output, the environment is configured correctly.

```shell
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

## Next Steps: From Examples to Full Development

This page helps you finish installation and verify the first example. To continue through the full operator development workflow, follow the path below:

1. **Operator development**: Read the Vector / Cube / fusion tutorials for your operator type, then modify an example kernel to complete the develop–compile–verify loop.
2. **Autotune**: Use Autotune to select suitable tiling configs.
3. **Debug and profiling**: Learn print-based debugging and performance collection.
4. **Migration and advanced topics**: Migrate from GPU Triton to Ascend, or check environment variables and compiler options.

### Development Guide

| Document | Description |
|----------|-------------|
| [Triton-Ascend Operator Programming](./programming_guide/index.md) | General principles: multi-core split, on-chip memory, access patterns, and tiling |
| [Triton-Ascend Operator Migration](./migration_guide/index.md) | Migrate GPU Triton operators to Ascend NPU |
| [Triton-Ascend Operator Debugging and Profiling](./debug_guide/index.md) | Debugging, profiling, and accuracy troubleshooting |
| [Environment Variables and Compiler Options](./environment_variable_and_compiler_options_reference.md) | Runtime and compilation configuration reference |

### Tutorials & Examples

| Document | Description |
|----------|-------------|
| [Vector Operator Development](./programming_guide/vector_operator.md) | Element-wise, reduction, gather/scatter, and other Vector Core operators |
| [Cube Operator Development](./programming_guide/cube_operator.md) | Cube operators centered on `tl.dot` and matrix multiplication |
| [CV Fusion Operator Development](./programming_guide/cv_fusion_operator.md) | Kernels that combine Cube compute with Vector post-processing |
| [Triton-Ascend Autotune](./autotune_guide.md) | Recommended Autotune usage and automatic tiling scope |
| [Example Operators](./examples/index.md) | End-to-end examples such as Softmax, LayerNorm, Attention, and Matmul |
