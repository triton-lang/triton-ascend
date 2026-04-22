# Quick Start

## Project Overview

Triton-Ascend is an optimized version of Triton that adapts to Huawei Ascend chips. It provides efficient automatic optimization of kernel functions, operator compilation, and deployment capabilities, and supports products such as Ascend Atlas A2/A3.
While being compatible with the core syntax of Triton, Ascend is optimized for features of Ascend NPUs, including automatic parsing of kernel function parameters, memory access logic optimization, and security deployment mechanism optimization.

## Online Documents

Complete online documents and network materials are provided, covering environment setup, operator development, optimization practices, and FAQ, to help you get started quickly. For details, see the [online documents](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html).

## Environment Requirements

### Hardware Requirements

Supported OS: Linux (AArch64/x86_64)

Supported Ascend products: Atlas A2/A3 series

Minimum hardware configuration: single-device 32 GB graphics memory (recommended)

### Software Dependency

Python (Python 3.9 to Python 3.11), CANN_TOOLKIT, CANN_OPS, [requirements.txt](../../requirements.txt), and [requirements_dev.txt](../../requirements_dev.txt)

For details about the CANN installation and configuration script, see [CANN installation description](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu). Developers can select the CANN version, product series, CPU architecture, operating system, and installation method to find the corresponding installation commands.

You need to select the required version (8.5.0 is recommended) based on the Ascend card model you use. The CANN installation takes about 5 to 10 minutes. Wait until the installation is complete.

- Note: If the user does not specify an installation path, the software will be installed to the default path as follows.
For root user: `/usr/local/Ascend`
For non-root user: `${HOME}/Ascend`, where `${HOME}` is the home directory of the current user.

The environment variable configuration above takes effect only in the current terminal session.

Users can add the command source `${HOME}/Ascend/ascend-toolkit/set_env.sh` to an environment variable configuration file (such as .bashrc) as needed.

You can run the following command to install the requirements:

```shell
pip install -r requirements.txt -r requirements_dev.txt
```

## Environment Setup

You can set up the Triton-Ascend environment by referring to section "Preparing the Environment" in [Installation Guide](installation_guide.md).

### Obtaining the Triton-Ascend Software Package

You can install the latest stable version package using the CLI.

```shell
pip install triton-ascend
```

- Note: Starting from version 3.5, Triton-Ascend mitigates the installation overwrite issue by declaring Triton as an installation dependency. When Triton-Ascend is installed, community Triton is installed first, and then Triton-Ascend overwrites the shared package directory. This helps avoid a later Triton reinstallation overwriting Triton-Ascend when other software packages that depend on Triton are installed. Different community Triton package versions are used for x86 and arm because arm installation packages are only available in the community starting from Triton 3.5: x86 depends on `triton==3.2.0`, and arm depends on `triton==3.5.0`.

- Note 1: This solution mitigates the installation overwrite issue, but it does not completely eliminate the conflict caused by community Triton and Triton-Ascend sharing the same top-level `triton` package directory. If a later installation explicitly reinstalls or upgrades community Triton, the installed Triton-Ascend may still be affected. In this case, uninstall both community Triton and Triton-Ascend first, and then reinstall Triton-Ascend.

You can also download the nightly package from the [download link](https://test.pypi.org/project/triton-ascend/#history) and install it locally.

- Note 2: If you download the nightly package for installation, select the Python version and architecture (AArch64/x86_64) of your server when selecting the Triton-Ascend package.
- Note 3: The nightly package is built every day. Developers submit MRs frequently. Note that if the package does not pass the stable test, function bugs may exist.

## Quick Environment Setup with Docker

We provide a Dockerfile to help you build a Docker environment image. During installation, the corresponding CANN Toolkit and Kernel packages will be automatically downloaded from the official CANN website. You need to specify CANN-related parameters for your machine using `--build-arg`.

| Parameter Name | Default Value | Available Options |
|----------------|---------------|----------------------------------------|
| CHIP_TYPE      | A3            | A3, 910B |
| CANN_VERSION   | 8.5.0 (Recommended) | 8.5.0, 8.3.RC1, 8.3.RC2, 8.2.RC1, 8.2.RC2 |

You can check the NPU model on your system using the `npu-smi` command.

For the machines corresponding to different `CHIP_TYPE` options, refer to the table below:

| Option No. | **CHIP_TYPE Value** | Corresponding Server/Product Series | Typical Server Model |
|:----------:|:-------------------:|:----------------------------------:|:-----------------------------------:|
| 1 | `A3` | Atlas A3 Training Series | Atlas 900 A3 SuperPoD |
| 2 | `A2` | Atlas A2 Training Series | Atlas 800T A2 |

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend
docker build \
--build-arg CHIP_TYPE=A3 \
--build-arg CANN_VERSION=8.5.0 \
-t triton-ascend-image:latest -f ./docker/Dockerfile .
```

To start a container from this image, you can use the following command as a reference:

```bash
docker run -u 0 -dit --shm-size=512g --name=triton-ascend_container --net=host --privileged \
--security-opt seccomp=unconfined \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /home:/home \
-v /etc/ascend_install.info:/etc/ascend_install.info \
triton-ascend-image:latest \
/bin/bash

# Enter the container
docker exec -u root -it triton-ascend_container /bin/bash
```

## Running Triton Examples

Run the example: [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)

```bash
# Set CANN environment variables (using the default root installation path `/usr/local/Ascend` as an example)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Clone the triton-ascend repository and examples (optional; required for running examples if not installed from source)
git clone https://gitcode.com/Ascend/triton-ascend.git
# Run the tutorials example:
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

Output similar to the following indicates that your environment is correctly configured.

```shell
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```
