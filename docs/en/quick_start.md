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

- CANN version: 9.0.0
- Python version: python3.11
- TorchNPU version: 2.7.1.post4

Note: For more compatibility relationships, please refer to the [Version Description Table](./release_note.md#version-compatibility-matrix).

## Quick Installation

```bash
pip install triton-ascend --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## Quick Start

**Run the vector addition example in tutorials to verify the results**

Vector addition example: [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)
By comparing the output of the Triton kernel with that of native PyTorch computation, it proves that the Ascend NPU device can correctly call the Triton kernel and ensure computational accuracy.

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
