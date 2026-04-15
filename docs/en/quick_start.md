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

- Note: Community Triton and Triton-Ascend cannot coexist. When you install other software that depends on Triton, community Triton will be automatically installed, which will overwrite the installed Triton-Ascend directory.
In this case, you need to uninstall community Triton first and then install Triton-Ascend.

You can also download the nightly package from the [download link](https://test.pypi.org/project/triton-ascend/#history) and install it locally.

- Note 1: If you download the nightly package for installation, select the Python version and architecture (AArch64/x86_64) of your server when selecting the Triton-Ascend package.
- Note 2: The nightly package is built every day. Developers submit MRs frequently. Note that if the package does not pass the stable test, function bugs may exist.

## Example for Running Triton

Run the [01-vector-add.py](../../ascend/examples/tutorials/01-vector-add.py) instance.
```bash
# Set the CANN environment variables (for example, as the root user and with the default installation path /usr/local/Ascend).
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Run the tutorials example.
python3 ./triton-ascend/ascend/examples/tutorials/01-vector-add.py
```
